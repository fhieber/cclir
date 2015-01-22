#include "train_bowfd.h"

int main(int argc, char** argv) {
	po::variables_map cfg;
	if (!init_params(argc, argv, &cfg)) return 1;

	rng.reset(new MT19937);

	unsigned I = cfg["iterations"].as<unsigned>();  // Iterations

	unsigned n = 0; // number of input examples
	if (cfg["input"].as<string>() == "-") {
		cerr << "Reading from STDIN...\n";
		I = 1;
	} else {
		std::ifstream in(cfg["input"].as<string>()); 
  		n = std::count(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>(), '\n');
  		cerr << n << " input pairs; running " << I << " iterations \n";
	}

	bool freeze = cfg.count("freeze");
	vector<int> frozen_features;
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

	// Adadelta
	SparseVector<weight_t> G; // accumulate squared gradients
	SparseVector<weight_t> X; // accumulate squared updates
	double decay_rate = cfg["decay_rate"].as<double>();
	double epsilon = cfg["epsilon"].as<double>();
	cerr << "Adadelta: epsilon="<<epsilon<<" decay_rate="<<decay_rate<<"\n";

	// hold out some examples for loss calculation
	bool hold_out = cfg.count("hold_out");
	unsigned hold_out_every_jth=0;
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
		int qlen;
		CLIR::FDocument* d_rel=NULL;
		while ( *input >> qid ) {
			// parse input
			input->ignore(1, '\t');
			*input >> qlen;
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
							if (ISIRFID(gi.first)) {
								ranker.ir_active()[gi.first] += ui;
							} else {
								w_smt[gi.first] += ui;
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

	} // end of epochs 

	delete decoder;

	cerr << "DONE.\n";

	return 0;

}
