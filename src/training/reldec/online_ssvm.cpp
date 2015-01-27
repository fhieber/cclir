#include <vector>
#include <numeric>
#include <cfloat>
#include <sys/time.h>

#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/foreach.hpp>

#include "config.h"

#include "sentence_metadata.h"
#include "verbose.h"
#include "viterbi.h"
#include "hg.h"
#include "hg_io.h"
#include "hg_sampler.h"
#include "hg_union.h"
#include "forest_writer.h"
#include "prob.h"
#include "ff_register.h"
#include "decoder.h"
#include "filelib.h"
#include "stringlib.h"
#include "fdict.h"
#include "time.h"
#include "sampler.h"

#include "weights.h"
#include "feature_vector.h" // typedefs FeatureVector,WeightVector, DenseWeightVector

using namespace std;
namespace po = boost::program_options;

boost::shared_ptr<MT19937> rng;

double lr, lambda;
bool l1, adagrad, ramploss, freeze, perceptron, rescoring_model, no_redecode, conservative;
int eval, dump;
vector<int> frozen_features;

// this way lr for all features is equal to the commandline specified lr in 1st iter
const double beta = 1.0; // AdaGrad beta parameter;
double delta = -1.0; // mira conservative update step size

bool init_params(int argc, char** argv, po::variables_map* conf) {
	po::options_description opts("Configuration options:");
	opts.add_options()
		("decoder_config,c",po::value<string>(),"Decoder configuration file")
		("weights,w", po::value<string>(), "* initial weights")
		("firstpass-model", po::value<string>(), "learn a model on top the weights given here. Use result with --weights2 during testing.")
		("rweights,r", po::value<string>(), "* relevance weights for hope derivations")
		("ramploss", po::value<bool>()->zero_tokens(), "use ramp loss")
		("perceptron", po::value<bool>()->zero_tokens(), "use perceptron (fear := viterbi)")
		("weights_output,o",po::value<string>(),"Directory to write weights to")
		("learningrate,n", po::value<double>()->default_value(0.0001), "learning rate")
		("reg,l", po::value<double>()->default_value(0.0), "l1 regularization w/ clipping (Tsuoroka et al). specify regularization strength")
		("adagrad,a", po::value<bool>()->zero_tokens(), "use per-coordinate learning rate (Duchi et al, 2010 / Green et al, ACL'13)")
		("random_seed,S", po::value<uint32_t>(), "Random seed (if not specified, /dev/random will be used)")
		("freeze", po::value<vector<string> >()->multitoken(), "specify set of features to freeze (excluded from the gradient)")
		("sample", po::value<int>(), "sample x times fear and hope from forest for each sentence.")
		("conservative", po::value<bool>()->zero_tokens(), "use conservative (MIRA) updates.")
		("no-redecode", po::value<bool>()->zero_tokens(), "use initial weights for each input sentence.")
		("forest-output,O",po::value<string>(),"Directory to write forests to");
	po::options_description clo("Command line options");
	clo.add_options()
		("config", po::value<string>(), "Configuration file (cdec.ini)")
		("help,H", "Print this help message and exit");
	po::options_description dconfig_options, dcmdline_options;
	dconfig_options.add(opts);
	dcmdline_options.add(opts).add(clo);
	po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
	if (conf->count("config")) {
		ifstream config((*conf)["config"].as<string>().c_str());
		po::store(po::parse_config_file(config, dconfig_options), *conf);
	}
	po::notify(*conf);
	if(conf->count("help") || !conf->count("decoder_config") || !conf->count("weights") || !conf->count("rweights") || !conf->count("weights_output")) {
		cerr << dcmdline_options << endl;
		return false;
	}
	return true;
}
	
Decoder* setupDecoder(const po::variables_map& cfg) {
	register_feature_functions();
	SetSilent(true); 
	ReadFile ini_rf(cfg["decoder_config"].as<string>());
	return new Decoder(ini_rf.stream());
}

/*
 * converts sparse to dense while keeping size of dense intact (contrary to feature_vector.h::sparse_to_dense())
 */
int sparse2dense(const WeightVector& v, DenseWeightVector* dv) {
	int nonzero=0;
	for(WeightVector::const_iterator i=v.begin();i!=v.end();++i) {
		assert(i->first < dv->size());
		(*dv)[i->first] = i->second; 
		if (i->second != 0)
			nonzero++;
	}
	return nonzero;
}

void loadRelevanceWeights(const string& fname, WeightVector& rw) {
	rw.clear();
	ReadFile in_file(fname);
	istream& in = *in_file.stream();
	assert(in);
	string f;
	double v;
	while(in>>f) {
		in>>v;
		if (f.empty()) continue;
		rw.set_value(FD::Convert(f), v);
	}
}

inline void setRelevanceToZero(FeatureVector& f, const WeightVector& rw) {
	//f.erase(rf3); f.erase(rf2); f.erase(rf1); f.erase(rf0);
	for (WeightVector::const_iterator it=rw.begin(); it!=rw.end(); ++it)
		f.set_value(it->first, .0);
}

double getFeaturesAndRelevance(Hypergraph& hg, const WeightVector& rw, FeatureVector& f) {
	f.clear();
	f = ViterbiFeatures(hg);
	double rel = rw.dot(f);
	setRelevanceToZero(f, rw);
	return rel;
}

inline void getGradient(FeatureVector& grad, const FeatureVector& bad, const FeatureVector& good, const WeightVector& rw, const bool freeze) {
	grad = bad - good;
	setRelevanceToZero(grad, rw);
	if (grad.l2norm()) grad /= grad.l2norm();
	if (freeze) {
		for (int ff=0;ff<frozen_features.size();++ff)
			grad.set_value(frozen_features[ff], .0);
	}
}

void update(WeightVector& w, const FeatureVector& g, FeatureVector& G) {
	if (adagrad) {
		/*
		 * Green et al, Fast and Adaptive Online Training of
		 * Feature-Rich Translation Models (ACL'13)
		 * AdaGrad applies per-coordinate (j) learning rate to the gradient:
		 * lr_j = 1 / (beta + sqrt(G^t)
		 *
		 * FOBOS (1) [eq. 13,14]
		 * gradient = gradient .* (lr / (beta + sqrt(G^t)))
		 * beta is a fixed parameter (>lr) so that denominator is not 0 for t=0;
		 *
		 * w is updated with the per coordinate learning rate
		 *
		 * FOBOS (2) [eq. 15]
		 * l1 regularization with w/ adaptive learning rate
		 */
		for (FeatureVector::const_iterator it=g.begin();it!=g.end();++it)
				w[it->first] += ( -lr / (beta + sqrt(G.get(it->first)))) * it->second;
		if (l1) {
			for (WeightVector::iterator it=w.begin();it!=w.end();++it) {
				if (it->second > 0)
					it->second = max(.0, it->second - lambda * ( lr / (beta + sqrt(G.get(it->first)))));
				else if (it->second < 0)
					it->second = min(.0, it->second + lambda * ( lr / (beta + sqrt(G.get(it->first)))));
			}
		}
		for (FeatureVector::const_iterator it=g.begin();it!=g.end();++it)
			G[it->first] += it->second * it->second;

	} else {
		// regular FOBOS 2-step update with constant learning rate
		// update weight vector: w = w - g*lr
		w.plus_eq_v_times_s(g, -lr);
		if (l1) {
			for (WeightVector::iterator it=w.begin(); it!=w.end();++it) {
				if (it->second > 0)
					it->second = max(.0, it->second - lambda * lr );
				else if (it->second < 0)
					it->second = min(.0, it->second + lambda * lr );
			}
		}
	}
}

struct TrainingObserver : public DecoderObserver {
	TrainingObserver() {}
	virtual void NotifyTranslationForest(const SentenceMetadata& smeta, Hypergraph* hg) {
		forest = *hg;
		cur_sent = smeta.GetSentenceID();
	}
	Hypergraph& GetCurrentForest() {
		return forest;
	}
	int GetCurrentSent() const {
		return cur_sent;
	}
	Hypergraph forest;
	int cur_sent;
};

struct Stats {
	Stats() : loss(0), cost(0), margin(0), rel_fear(0), rel_viterbi(0), rel_hope(0), uc(0), fc(0) {}
	double loss, cost, margin, rel_fear, rel_viterbi, rel_hope;
	int uc, fc;
	void update(const string& id, const int cur_sent, const double c,
			const double m, const double rf,
			const double rv, const double rh,
			const double nonzero) {
		cost += c;
		margin += m;
		loss += max(c+m,.0);
		rel_fear += rf;
		rel_viterbi += rv;
		rel_hope += rh;
		uc++;
		fc=nonzero;
		cerr << id << '\t' << cur_sent << '\t' << max(c+m,.0) << '\t' <<
				c << '\t' << m << '\t' << rf <<
				'\t' << rv<< '\t' << rh <<
				'\t' << fc << endl;
	}
	void show() {
		if (uc==0) uc=1;
		cerr << uc << "\t0\t" <<
				loss/uc << '\t' <<
				cost/uc << '\t' <<
				margin/uc << '\t' <<
				rel_fear/uc << '\t' <<
				rel_viterbi/uc << '\t' <<
				rel_hope/uc << '\t' <<
				fc << endl;
	}
};

int main(int argc, char** argv) {
	po::variables_map cfg;
	if (!init_params(argc,argv,&cfg)) return 1;

	if (cfg.count("random_seed"))
		rng.reset(new MT19937(cfg["random_seed"].as<uint32_t>()));
	else
		rng.reset(new MT19937);

	// set variables
	lr = cfg["learningrate"].as<double>();
	lambda = cfg["reg"].as<double>();
	l1 = (lambda != 0.0);
	adagrad = cfg.count("adagrad");
	ramploss = cfg.count("ramploss");
	perceptron = cfg.count("perceptron");
	freeze = cfg.count("freeze");
	rescoring_model = cfg.count("firstpass-model");
	no_redecode = cfg.count("no-redecode");
	conservative = cfg.count("conservative");
	if (freeze) {
		const vector<string>& ffstrs = cfg["freeze"].as<vector<string> >();
		stringstream ffss;
		ffss << "frozen features: ";
		for (vector<string>::const_iterator ffit=ffstrs.begin();ffit!=ffstrs.end();++ffit) {
			frozen_features.push_back(FD::Convert(*ffit));
			ffss << *ffit << " ";
		}
		cerr << ffss.str() << endl;
	}

	// setup decoder
	Decoder* decoder = setupDecoder(cfg);
	if (!decoder) {
		cerr << "error while loading decoder with" << cfg["decoder_config"].as<string>() << "!\n";
		return 1;
	}
	TrainingObserver observer;
	// get reference to decoder weights
	vector<weight_t>& decoder_weights = decoder->CurrentWeightVector();
	// setup weights
	WeightVector w, w_hope, w_fear;
	// the SMT weights (to be optimized)
	Weights::InitFromFile(cfg["weights"].as<string>(), &decoder_weights);
	Weights::InitSparseVector(decoder_weights, &w);
	if (rescoring_model) Weights::InitFromFile(cfg["firstpass-model"].as<string>(), &decoder_weights);
	// the weight vector that gives you hope
	loadRelevanceWeights(cfg["rweights"].as<string>(), w_hope);
	cerr << "|RW|=" << w_hope.size() << endl;
	cerr << "|W| =" << w.size() << endl;
	cerr << w_hope << endl;
	if (ramploss) cerr << "RAMPLOSS\n";
	if (perceptron) cerr << "PERCEPTRON\n";

	const string input = decoder->GetConf()["input"].as<string>();
	cerr << "Reading input from " << ((input == "-") ? "STDIN" : input.c_str()) << endl;
	ReadFile in_read(input);
	istream *in = in_read.stream();
	assert(*in);
	string id, sentence;
	int cur_sent = 0;
	Stats stats;
	double rel_viterbi, rel_hope, rel_fear;
	FeatureVector f_viterbi, f_hope, f_fear; // feature vectors
	FeatureVector g, G; // gradient, squared gradient coordinates over time
	vector<WordID> t_viterbi, t_hope, t_fear; // translations
	unsigned uc = 0; // update count
	unsigned lc = 0; // line count
	std::vector<HypergraphSampler::Hypothesis> samples;

	while(*in >> id) {

		in->ignore(1, '\t');
		getline(*in, sentence);
		if (sentence.empty() || id.empty()) continue;

		decoder->SetId(cur_sent);
		decoder->Decode(sentence, &observer); // decode with decoder_weights
		cur_sent = observer.GetCurrentSent();
		Hypergraph hg = observer.GetCurrentForest();

		if (rescoring_model) hg.Reweight( w ); // apply current model to forest

		/*
		 * SAMPLE PATHS FROM FOREST
		 */
		if (cfg.count("sample")) {
			
			ViterbiESentence(hg, &t_viterbi);
			rel_viterbi = w_hope.dot(ViterbiFeatures(hg));
			FeatureVector acc_g;
			for (unsigned s=0; s<cfg["sample"].as<int>(); ++s) {
				HypergraphSampler::sample_hypotheses(hg, 2, &(*rng), &samples);
				double rel0 = w_hope.dot(samples[0].fmap);
				double rel1 = w_hope.dot(samples[1].fmap);
				if (rel0 > rel1) {
					rel_hope = rel0;
					rel_fear = rel1;
					t_hope = samples[0].words;
					t_fear = samples[1].words;
					getGradient(g, samples[1].fmap, samples[0].fmap, w_hope, freeze);
				} else if (rel1 > rel0) {
					rel_hope = rel1;
					rel_fear = rel0;
					t_hope = samples[1].words;
					t_fear = samples[0].words;
					getGradient(g, samples[0].fmap, samples[1].fmap, w_hope, freeze);
				} else { continue; }

				acc_g += g;
				//stats.update(id, cur_sent, acc_g.dot(w), rel_hope-rel_fear, rel_fear, rel_viterbi, rel_hope, w.num_nonzero());
			}

			cout << id << " ||| "
				 << "SAMPLED" << " ||| "
				 << TD::GetString(t_viterbi) << " ||| "
				 << "SAMPLED" << endl;
			cout.flush();
			
			update( w, g, G );

			if (!rescoring_model || !no_redecode) w.init_vector(&decoder_weights);
			++uc;

		/*
		 * SHORTEST PATH HOPE & FEAR
		 */
		} else { // get hope and fear based on shortest path w.r.t relevance weights
		
			// get viterbi
			rel_viterbi = getFeaturesAndRelevance(hg, w_hope, f_viterbi);
			ViterbiESentence(hg, &t_viterbi);

			// get hope
			hg.Reweight( ramploss ? (w_hope + w) : w_hope );
			rel_hope = getFeaturesAndRelevance(hg, w_hope, f_hope);
			ViterbiESentence(hg, &t_hope);

			// get fear
			hg.Reweight( perceptron ? w : (w - w_hope) );
			rel_fear = getFeaturesAndRelevance(hg, w_hope, f_fear);
			ViterbiESentence(hg, &t_fear);

			cerr << "H " << f_hope << endl;
			cerr << "F " << f_fear << endl;

			// compute loss
			double rloss = rel_hope - rel_fear;
			double cost = (f_hope - f_fear).dot(w);
			double loss = rloss - cost;

			cerr << "ID=" << id << " SENT=" << cur_sent << endl;
			cerr << " rfear=" << rel_fear << " rbest=" << rel_viterbi << " rel_hope=" << rel_hope << endl;
			cerr << " rloss=" << rloss << " cost=" << cost << " loss=" << loss << endl;

			if (conservative) { // PA mira (ultraconservative updates)

				if (loss > 0.0001) {
					cerr << " constraint violated. optimizing." << endl;
					double diffsqnorm = (f_hope - f_fear).l2norm_sq();
					cerr << " normsq=" << diffsqnorm << endl;
					delta = (diffsqnorm > 0) ? (-loss / (lr*diffsqnorm)) : 0;
					cerr << " d1=" << delta;
					delta = max(-1.0, min(delta,.0)); // clip
					cerr << " d2=" << delta;
					cerr << " step=" << delta*lr << endl;
					w += (f_fear) * (delta * lr);
					w -= (f_hope) * (delta * lr);

				} else { cerr << " no violation, no update.\n"; }

			} else { // regular sgd with fixed step size
				w += (f_fear) * -lr;
				w -= (f_hope) * -lr;
			}

			cout << id << " ||| " 
				 << TD::GetString(t_fear) << " ||| " 
				 << TD::GetString(t_viterbi) << " ||| " 
				 << TD::GetString(t_hope) << endl;
			cout.flush();
			cerr << " |w|=" << w.num_nonzero() << endl;
			
			// if not rescoring model or batch update, set weights for decoder to the current model
			if (!rescoring_model || !no_redecode) {
				decoder_weights.clear();
				w.init_vector(&decoder_weights);
			}
			++uc;

		}
		++lc;
	}

	decoder_weights.clear();
	w.init_vector(&decoder_weights);
	//Weights::ShowLargestFeatures(decoder_weights);
	// write weights
	int node_id = rng->next() * 100000;
	cerr << " Writing model to " << node_id << endl;
	ostringstream os;
	os << cfg["weights_output"].as<string>() << "/weights." << node_id;
	string msg = "Online SSVM tuned weights ||| " + boost::lexical_cast<std::string>(node_id) + " ||| " + boost::lexical_cast<std::string>(uc);
	Weights::WriteToFile(os.str(), decoder_weights, true, &msg);
	delete decoder;
	cerr << "\ndone.\n";
	return 0;

}

