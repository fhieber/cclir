#include <algorithm>
#include <climits>
#include <unordered_map>
#include <unordered_set>
#include "ranker.h"
#include "vwbindings.h"

#include "sampler.h"
boost::shared_ptr<MT19937> rng;

const string BASE_VW_CONF = " --adaptive --invariant --loss_function quantile --quantile_tau 1.0 --noconstant --min_prediction -2000 --max_prediction 2000";
// other suggested options: -l 0.001 --power_t 0 --initial_t 0 -b 22
// learning rate much smaller than default
// no decaying learning rate
// set sufficient bit size if many features!

using namespace CLIR;

unsigned n = 0; // number of input examples
bool hold_out = false;
unsigned hold_out_every_jth=0;
bool freeze;
vector<int> frozen_features;
bool use_vw; // if to use Vowpal Wabbit
// ## ADADELTA (Zeiler, 2012) variables ##
double decay_rate = 0.95;
double epsilon = 1e-6;
SparseVector<weight_t> G; // accumulate squared gradients
SparseVector<weight_t> X; // accumulate squared updates


Decoder* setupDecoder(const po::variables_map& cfg) {
	register_feature_functions();
	SetSilent(true);
	ReadFile ini_rf(cfg["decoder_config"].as<string>());
	return new Decoder(ini_rf.stream());
}

bool init_params(int argc, char** argv, po::variables_map* conf) {
	po::options_description opts;
	opts.add_options()
			("input,i", po::value<string>()->default_value("-"), "input (file or '-' for STDIN)")
			("decoder_config,c", po::value<string>(),"cdec configuration")
			("weights,w", po::value<string>(),"initial smt weights")
			("ir_weights", po::value<string>(), "initial ir weights")
			("iterations,I", po::value<unsigned>()->default_value(1), "# of iterations over input (cannot be STDIN if >1)")
			("default_ir_weight", po::value<double>()->default_value(0.0),"default ir weight for term match")
			("word_classes", po::value<string>()->default_value(""), "a file containing word2class mappings for extended matching")
			("dftable", po::value<string>(), "* df table")
			("stopwords,s",po::value<string>(), "stopword file")
			("vwconfig,v",po::value<string>(), "Vowpal Wabbit config, use quotes\nsuggested options: '-l 0.01 --power_t 0 --initial_t 0 -b 22'\nSet sufficient bit size for many features!")
			("epsilon,e", po::value<double>()->default_value(1e-6), "constant added to RMS if not VW")
			("decay_rate", po::value<double>()->default_value(0.95), "decay_rate for sliding window of squared gradients if not VW")
			//("sync_plan,p", po::value<unsigned>()->default_value(2), "when to sync updated VW model with decoder weights: (1) each new query (2) +each new relevant document")
			("tune_dense", po::value<bool>()->default_value(0), "tune a dense IR feature instead of sparse")
			("tune_smt", po::value<bool>()->default_value(1), "tune SMT features jointly with IR feature(s)")
			("freeze", po::value<vector<string> >()->multitoken(), "specify set of features to freeze (excluded from the gradient sent to VW)")
			("output,o", po::value<string>(),"output directory to write learned weights to.")
			("hold_out", po::value<double>(),"percentage of input to hold out for loss calculation")
			("perceptron,P", po::value<bool>()->zero_tokens(), "use only a perceptron (zero margin required)")
			("fixed_margin,M", po::value<bool>()->zero_tokens(), "require a fixed margin of 1 for each example")
			("quiet", po::value<bool>()->zero_tokens(), "quiet ranker");

	po::options_description clo("Command line options");
	clo.add_options()("config", po::value<string>(),
			"Configuration file (cdec.ini)")("help,H",
			"Print this help message and exit");
	po::options_description dconfig_options, dcmdline_options;
	dconfig_options.add(opts);
	dcmdline_options.add(opts).add(clo);
	po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
	if (conf->count("config")) {
		ifstream config((*conf)["config"].as<string>().c_str());
		po::store(po::parse_config_file(config, dconfig_options), *conf);
	}
	po::notify(*conf);
	if (conf->count("help") 
		|| !conf->count("decoder_config") 
		//|| !conf->count("vwconfig") 
		|| !conf->count("dftable")
		|| !conf->count("output")) {
		cerr << dcmdline_options << endl;
		return false;
	}
	if ((*conf)["iterations"].as<unsigned>() > 1 && (*conf)["input"].as<string>() == "-") {
		cerr << "\nCannot iterate multiple times over STDIN!\n";
		return false;
	}
	return true;
}

void writeWeights(const string& fname, const WeightVector& w, const string& msg="") {
	WriteFile out(fname);
	ostream& o = *out.stream();
	assert(o);
	o.precision(12);
	if (!msg.empty()) {
		o << msg << "\n";
	}
	for (WeightVector::const_iterator i = w.begin(); i != w.end(); ++i) {
		if (i->second != 0.0)
			o << FD::Convert(i->first) << " " << i->second << "\n";
	}
	cerr << "Weights written to '" << fname << "'.\n";
}

int main(int argc, char** argv) {
	rng.reset(new MT19937);
	po::variables_map cfg;
	if (!init_params(argc, argv, &cfg)) return 1;

	unsigned I = cfg["iterations"].as<unsigned>();  // Iterations

	if (cfg["input"].as<string>() == "-") {
		cerr << "Reading from STDIN...\n";
		I = 1;
	} else {
		std::ifstream in(cfg["input"].as<string>()); 
  		n = std::count(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>(), '\n');
  		cerr << n << " input pairs; running " << I << " iterations \n";
	}

	freeze = cfg.count("freeze");
	if (freeze) {
		const vector<string>& ffstrs = cfg["freeze"].as<vector<string> >();
		string ffs = "frozen features: ";	
		for (vector<string>::const_iterator ffit=ffstrs.begin();ffit!=ffstrs.end();++ffit) {
			frozen_features.push_back(FD::Convert(*ffit));
			ffs += *ffit + " ";
		}
		cerr << ffs << endl;
	}

	// load stopwords
	if (cfg.count("stopwords")) {
		sw::loadStopwords(cfg["stopwords"].as<string>());
		cerr << sw::stopwordCount() << " stopwords loaded from '"
				<< cfg["stopwords"].as<string>() << "'\n";
	}
	// setup decoder
	Decoder* decoder = setupDecoder(cfg);
	if (!decoder) {
		cerr << "error while loading decoder with"
				<< cfg["decoder_config"].as<string>() << "!\n";
		return 1;
	}
	// get reference to decoder weights
	vector < weight_t > &decoder_weights = decoder->CurrentWeightVector();
	WeightVector w_smt;
	// the SMT weights
	if (cfg.count("weights")) {
		Weights::InitFromFile(cfg["weights"].as<string>(), &decoder_weights);
		Weights::InitSparseVector(decoder_weights, &w_smt);
	}

	WeightVector w_ir;
	if (cfg.count("ir_weights")) {
		vector < weight_t > dirw;
		Weights::InitFromFile(cfg["ir_weights"].as<string>(), &dirw);
		Weights::InitSparseVector(dirw, &w_ir);
	}
	DfTable dft(cfg["dftable"].as<string>());
	cerr << "DF table loaded (" << dft.size() << " entries)." << endl;
	IRFeatureExtractor extractor(&w_ir, cfg["default_ir_weight"].as<double>(), cfg["word_classes"].as<string>());
	IRFeatureIndicator indicator(&extractor, &dft);
	TrainingRanker ranker(decoder,
						  &indicator,
						  &w_smt,
						  &w_ir,
						  cfg["tune_dense"].as<bool>(),
						  cfg.count("quiet"));
	ranker.extract_smt = cfg["tune_smt"].as<bool>();
	ranker.extract_sparse = !cfg["tune_dense"].as<bool>();
	ranker.extract_dense = cfg["tune_dense"].as<bool>();

	// initialize VW model
	use_vw = cfg.count("vwconfig");
	vw* vwm = NULL;
	if (use_vw) {
		cerr << "VW config: " << cfg["vwconfig"].as<string>()+BASE_VW_CONF << endl;
		vwm = VW::initialize(cfg["vwconfig"].as<string>()+BASE_VW_CONF);
		if (!vwm) {
			cerr << "could not initialize vw model!\n";
			exit (1);
		}
		// set weights in VW model
		VWBindings::set_vw_weights(*vwm, w_smt);
		VWBindings::set_vw_weights(*vwm, w_ir);

		//VWBindings::compare_vw_weights(*vwm, w_smt, w_ir);

		cerr << "VW model initialized with smt and ir weights.\n";
	} else {
		cerr << "I am using my own ADADELTA updates.\n";
		decay_rate = cfg["decay_rate"].as<double>();
		epsilon = cfg["epsilon"].as<double>();
		cerr << "epsilon="<<epsilon<<" decay_rate="<<decay_rate<<"\n";
	}

	// hold out some examples for loss calculation
	hold_out = cfg.count("hold_out");
	if (hold_out) { 
		hold_out_every_jth = n / (n*cfg["hold_out"].as<double>());
		cerr << "holding out every " << hold_out_every_jth << " example.\n";
	}
	// perceptron?
	const bool perceptron = cfg.count("perceptron");
	if (perceptron) cerr << "using perceptron (0 margin required)\n";
	const bool fixed_margin = cfg.count("fixed_margin");
	if (fixed_margin) cerr << "using a fixed margin of 1\n";

	// OPTIMIZATION
	for ( unsigned i = 0; i < I; ++i ) {

		if (use_vw)
			cerr << "LR=" << vwm->eta << endl;
		//else
		//	cerr << "LR=" << eta << endl;

		WordID prev_qid = -1;
		WordID prev_d_rel_id = -1;
		string qid, qstr, d_rel_id, d_rel_tfstr, d_irel_id, d_irel_tfstr;
		int d_rel_len, d_irel_len;
		float margin;
		FeatureVector f_rel, f_irel;
		double s_rel=0, s_irel=0; // scores
		unsigned cp = 0;  // # examples
		unsigned cco = 0; // # examples correctly ordered
		unsigned cne = 0; // # negative examples
		unsigned cgp = 0; // # of pairs with sufficient margin
		unsigned che = 0; // # of held-out examples
		double loss = 0;
		double hold_out_loss = 0;
		unsigned nonzero_gradients = 0;
		unsigned grad_nonzero_sum = 0;
		double gradient_l2norm_sum = 0;
		ReadFile input(cfg["input"].as<string>());
		unsigned j=0;
		CLIR::FDocument* d_rel=NULL;
		while ( *input >> qid ) {
			// parse input
			input->ignore(1, '\t');
			getline(*input, qstr, '\t');
			*input >> margin;
			input->ignore(1, '\t');
			*input >> d_rel_id;
			input->ignore(1, '\t');
			*input >> d_rel_len;
			input->ignore(1, '\t');
			getline(*input, d_rel_tfstr, '\t');
			*input >> d_irel_id;
			input->ignore(1, '\t');
			*input >> d_irel_len;
			input->ignore(1, '\t');
			getline(*input, d_irel_tfstr);
			if (qid.empty() || qstr.empty() || margin == 0 || d_rel_id.empty()
					|| d_rel_len == 0 || d_rel_tfstr.empty() || d_irel_id.empty()
					|| d_irel_len == 0 || d_irel_tfstr.empty()) { cerr << "Warning! Format Error!\n"; continue; }

			if (TD::Convert(qid) != prev_qid) { // new query, produce hypergraph, apply irff
				prev_qid = TD::Convert(qid);
				decoder_weights.clear();
				// update decoder weights with current SMT weights
				w_smt.init_vector(&decoder_weights);
				// update global IR weights (w_ir) with the latest version from the ranker/previous query
				for (WeightVector::iterator it=ranker.ir_active().begin(); it!=ranker.ir_active().end(); ++it)
						w_ir.set_value(it->first, it->second);
				ranker.translate(qid, qstr);
				ranker.applyIndicator(); // fill qspace_.ir_active() with used feature weights (& IR default_weight)
				// update w_ir weights in vw model if we saw some new ir features using the default weight.
				// Uses ranker.ir_active() since this is smaller and only contains relevant features. everything else is already in VW model
				if (use_vw)
					VWBindings::set_vw_weights(*vwm, ranker.ir_active()); 
			} else {
				ranker.pushSMTWeightsToHypergraph(); // only reweight the hypergraph with current SMT weights
			}

			if (TD::Convert(d_rel_id) != prev_d_rel_id) { // new relevant document, read it
				prev_d_rel_id = TD::Convert(d_rel_id);
				delete d_rel;
				d_rel = new CLIR::FDocument(d_rel_id, d_rel_len, extractor.convert(vutils::read_vector(d_rel_tfstr)));
			}
			// read irrelevant document
			CLIR::FDocument d_irel(d_irel_id, d_irel_len, extractor.convert(vutils::read_vector(d_irel_tfstr)));

			// scoring
			s_rel  = log( ranker.score(*d_rel, f_rel) ); // score relevant document and get feature vector
			s_irel = log( ranker.score(d_irel, f_irel) ); // score irrelevant document and get feature vector

			//cerr << f_rel << endl;
			//cerr << f_irel << endl;
			
			//cerr << "w_smt x f_rel = " << f_rel.dot(w_smt) << endl;
			//cerr << "w_ir x f_rel = " << f_rel.dot(ranker.ir_active()) << endl;

			if (perceptron) margin = 1e-06; // if perceptron we only require correct ordering (margin very close to 0)
			if (fixed_margin) margin = 1;
			FeatureVector gradient = (f_rel - f_irel); // this is the INVERTED gradient so we need to ADD as an update!
			if (freeze) {
				for (unsigned x=0;x<frozen_features.size();++x)
					gradient.set_value(frozen_features[x], .0);
			}

			//cerr << "Gradient " << gradient << endl;
			// gradient.dot(w_smt + w_ir) can be a prediction with outdated pathes, but with updated weights.
			// Thus we use difference in scores directly. VW makes its own prediction anyway
			double prediction = s_rel - s_irel;

			//double comp_pred_rel = f_rel.dot(w_smt + ranker.ir_active());
			//double comp_pred_irel = f_irel.dot(w_smt + ranker.ir_active());

			/*if (!VWBindings::close_enough(comp_pred_rel, s_rel)) {
				cerr << "WARNING REL scores differ: " << comp_pred_rel << " " << s_rel << endl;
			}
			if (!VWBindings::close_enough(comp_pred_irel, s_irel)) {
				cerr << "WARNING IREL scores differ: " << comp_pred_irel << " " << s_irel << endl;
			}*/

			// CURRENT BUG STATUS: f_rel.dot(w) for relevant doc is correct! s_rel is incorrect. for irel doc it seems to be ok :/

			cout << "Q(" << qid << ") ||| D+(" << d_rel_id << "): " << s_rel 
			     << " ||| D-(" << d_irel_id << "):" << s_irel << " ||| Desired Margin: " 
			     << margin << " ||| Prediction: " << prediction << " ||| ";

			j++;
			if (hold_out && hold_out_every_jth!=0 && j%hold_out_every_jth==0) { // if this is a hold-out example 
				che++;
				if (prediction < margin) hold_out_loss += margin - prediction;
				cout << " held out ||| " << endl;
				continue;
			}
			cp++;
			if (s_rel > s_irel) cco++; // D+ > D-?
			if (prediction < margin) {
				loss += margin - prediction; // compute loss
				cne++;
				cout << "*update* ||| ";
			} else {
				cout << " ||| ";
				cgp++;
			}

			//if (!gradient.empty()) {
			unsigned grad_num_nonzero = gradient.num_nonzero();
			if (grad_num_nonzero > 0) {

				nonzero_gradients++;
				gradient_l2norm_sum += gradient.l2norm();
				grad_nonzero_sum += grad_num_nonzero;
				cout << "|G|=" << grad_num_nonzero << " ||| ";

				if (use_vw) {

					// Vowpal Wabbit
					example* x = VWBindings::build_vwexample(*vwm, gradient);
					VW::add_label(x, margin);
					vwm->learn(x);
					double vw_prediction = (((label_data*)x->ld)->prediction);
					cout << " VW Prediction: " << vw_prediction;
					VW::finish_example(*vwm, x);
					// get updated weights
					//if (vw_prediction < margin) { // if there was an update, we need to get the updated weights for SMT and the current active IR features
					VWBindings::get_vw_weights(*vwm, gradient, w_smt, ranker.ir_active(), false);
					//}
					//cout << vwm->sd->t << endl << endl; // time t

				} else {

					if (prediction < margin) {
						// ADADELTA (Zeiler, 2012) update
						// NOTE: we do not update the squared gradients and updates for values not present in the gradient.
						// This is probably wrong but more efficient since we have two parameter vectors)
						for (auto& gi : gradient) {
							double& G_i = G[gi.first];
							double& X_i = X[gi.first];
							G_i = decay_rate * G_i + (1.0 - decay_rate) * (gi.second*gi.second);
							double ui = (sqrt(X_i + epsilon) / sqrt(G_i + epsilon)) * gi.second;
							X_i = decay_rate * X_i + (1.0 - decay_rate) * (ui*ui);
							if (ui) {
								if (VWBindings::is_ir_feature(gi.first)) {
									ranker.ir_active()[gi.first] += ui;
								} else {
									w_smt[gi.first] += ui;
								}
							}
						}
					}
				
				}

			}
			
			cout << endl;

		} // end of input loop

		// end of input: synchronize weights
		decoder_weights.clear();
		// update decoder weights with current SMT weights
		w_smt.init_vector(&decoder_weights);
		// update global IR weights (w_ir) with the latest version from the ranker/previous query
		for (WeightVector::iterator it=ranker.ir_active().begin(); it!=ranker.ir_active().end(); ++it)
				w_ir.set_value(it->first, it->second);

		if (cp>0) {
			cerr    << "ITERATION " << i << endl
					<< "|W_SMT|=" << w_smt.size() << " |W_IR|=" << w_ir.size() << endl
					<< "# of total pairs seen:        " << cp << endl
					<< "# of pairs correctly ordered: " << cco << " ("
					<< (cco / (double) cp) * 100 << "%)" << endl
					<< "# of pairs /w suff. margin:   " << cgp << " ("
					<< (cgp / (double) cp) * 100 << "%)" << endl 
					<< "# of pairs to learn from:     " << cne << " ("
					<< (cne / (double) cp) * 100 << "%)" << endl
					<< "# of nonzero gradients:       " << nonzero_gradients << endl;
			if (nonzero_gradients!=0) {
			cerr	<< "avg. ||G||=" << gradient_l2norm_sum/nonzero_gradients << endl
					<< "avg.  |G| =" << grad_nonzero_sum/nonzero_gradients << endl
					<< "LOSS " << loss/(double)cp << endl;
			}
			
			//VWBindings::compare_vw_weights(*vwm, w_smt, w_ir);
		}
		if (hold_out && che>0)
			cerr << "TEST (" << che << ") " << hold_out_loss/che << endl;
		cerr << endl;


		// write weights to output directory
		int node_id = rng->next() * 100000;
		cerr << " Writing weights to " << node_id << endl;
		ostringstream oss;
		oss << cfg["output"].as<string>() << "/weights." << i << "." << node_id;
		string msg = "# DECODE_AND_VWTRAIN tuned weights ||| " + boost::lexical_cast<std::string>(node_id) + " ||| " + boost::lexical_cast<std::string>(cp);
		writeWeights(oss.str() + ".smt", w_smt, msg);
		writeWeights(oss.str() + ".ir", w_ir, msg);
		//vwm->eta = vwm->eta / 2; // TODO

	} // end of epochs 

	if (use_vw)
		VW::finish(*vwm);

	delete decoder;

	cerr << "DONE.\n";

	return 0;

}
