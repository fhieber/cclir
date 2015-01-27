#include <vector>
#include <numeric>
#include <cfloat>
#include <sys/time.h>
#include <limits>

#include <boost/shared_ptr.hpp>
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
bool freeze;
vector<int> frozen_features;

double MIN_REL, MAX_REL;
double SMO_EPS = 0.0001;
int SMO_ITER = 10;
int CP_ITER = 10;
WeightVector w, relw, relw_scaled, negrelw; // sparse weights and relevance weights
unsigned hope_select, fear_select, optimizer;
unsigned scaling;
double scalingfactor;

bool init_params(int argc, char** argv, po::variables_map* conf) {
	po::options_description opts("CuttingPlane PA MIRA on full Hyergraphs\nConfiguration options:");
	opts.add_options()
		("decoder_config,c",po::value<string>(),"Decoder configuration file")
		("weights,w", po::value<string>(), "initial weights")
		("rweights,r", po::value<string>(), "* relevance weights for hope derivations")
		("optimizer",po::value<int>()->default_value(3), "Optimizer (SGD=1, PA MIRA w/Delta=2, Cutting Plane MIRA=3, PA MIRA=4)")
		("hope,h", po::value<int>()->default_value(1), "Hope selection max(model-delta)=1, max(-delta)=2")
		("fear,f", po::value<int>()->default_value(1), "only for SGD: Fear selection max(model+delta)=1, max(delta)=2, max(model)=3)")
		("weights_output,o",po::value<string>(),"Directory to write weights to")
		("learningrate,n", po::value<double>()->default_value(0.0001), "learning rate")
		("random_seed,S", po::value<uint32_t>(), "Random seed (if not specified, /dev/random will be used)")
		("freeze", po::value<vector<string> >()->multitoken(), "specify set of features to freeze (excluded from the feature vectors)")
		("scaling,s", po::value<int>()->default_value(1), "scale rw to ||w||: 0 (disabled), 1 (rescale at initializiation), 2 (rescale after each update of w).")
		("scalingfactor", po::value<double>()->default_value(1), "x in rw = (rw / ||w||) * x");
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
	if(conf->count("help") || !conf->count("decoder_config") || !conf->count("rweights") || !conf->count("weights_output")) {
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

//updates relw_scaled by scaling it to the current norm of w, multiplied by the scalingfactor
void scaleRelevanceWeights(const double scalingfactor) {
	relw_scaled /= relw_scaled.l2norm(); // to length 1: relw / ||relw||
	const double cur_w_norm = w.l2norm(); // ||w||
	if (cur_w_norm == 0) return;
	relw_scaled *= cur_w_norm * scalingfactor; // scale by ||w||*scalingfactor
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

inline double relscale(double rel) {
	if (MAX_REL == MIN_REL) return 1;
	return (rel - MIN_REL) / (MAX_REL - MIN_REL);
}

struct HypothesisInfo {
	HypothesisInfo() : rel(), cost(), loss(), alpha() {};
	~HypothesisInfo() { oracle.reset(); features.clear(); hyp.clear(); }
	SparseVector<double> features;
	vector<WordID> hyp;
	double rel; // raw relevance (constant)
	double cost; // scaled relevance delta to oracle
	double loss; // loss w.r.t to oracle and weights
	double alpha; // dual variable of this derivation
	boost::shared_ptr<HypothesisInfo> oracle; // pointer to oracle
	void computeCost() {
		// cost is constant for every hypothesis so this should be called once
		// as soon as we have max_rel and min_rel for scaling.
		assert(oracle);
		//cost = relscale(oracle->rel) - relscale(rel);
		cost = oracle->rel - rel;
	}
	double computeLoss() {
		// compute loss w.r.t to oracle hypothesis and current weights w
		assert(oracle);
		if (hope_select==1)
			loss = (features.dot(w) + cost) - (oracle->features.dot(w) - oracle->cost);
		else
			loss = (features.dot(w) + cost) - (oracle->features.dot(w));
		if (loss < 0) {
			cerr << "Warning! Loss < 0! this_score=" << features.dot(w) << " oracle_score=" << oracle->features.dot(w) << " this_cost=" << cost << " oracle_cost=" << oracle->cost << endl;
			loss = 0;
		}
		return loss;
	}
};
inline std::ostream& operator<<(std::ostream& out, const boost::shared_ptr<HypothesisInfo>& h) {
  out << "[H: score=" << h->features.dot(w) << " rel=" << h->rel << "(" << relscale(h->rel) << "), cost=" << h->cost << ", loss=" << h->loss << ", alpha=" << h->alpha << "]";
  return out;
}

bool lossComp(const boost::shared_ptr<HypothesisInfo>& h1, const boost::shared_ptr<HypothesisInfo>& h2 )  { return h1->loss < h2->loss; };

boost::shared_ptr<HypothesisInfo> MakeHypothesisInfo(Hypergraph& hg) {
	/*
	 * create an HypothesisInfo with feature vector, translation and its relevance
	 * relevance feature values are removed (and optionally any frozen features)
	 */
	boost::shared_ptr<HypothesisInfo> h(new HypothesisInfo);
	h->features = ViterbiFeatures(hg);
	h->rel = h->features.dot(relw);
	// clean relevance weights from feature vector
	for (WeightVector::iterator it=relw.begin(); it!=relw.end(); ++it) { h->features.set_value(it->first, .0); }
	ViterbiESentence(hg, &(h->hyp));
	if (freeze) { for (unsigned x=0;x<frozen_features.size();++x) { h->features.set_value(frozen_features[x], .0); } }
	// for rel scaling:
	if (h->rel > MAX_REL) MAX_REL = h->rel;
	if (h->rel < MIN_REL) MIN_REL = h->rel;
	return h;
}

bool SelectPair(const vector<boost::shared_ptr<HypothesisInfo> >& S, vector<boost::shared_ptr<HypothesisInfo> >& pair) {
	pair.clear();
	for (unsigned u=0;u<S.size();u++) {
		// get maximum loss != S[u] in S
		boost::shared_ptr<HypothesisInfo> max_loss;
		for (unsigned i=0;i<S.size();i++) { // select maximal violator
			if (i!=u && (!max_loss || S[i]->loss > max_loss->loss)) { max_loss = S[i]; }
		}
		if (!max_loss) return false;

		// first heuristic
		if ((S[u]->alpha == 0) && (S[u]->loss > max_loss->loss + SMO_EPS)) {
			for (unsigned i=0;i<S.size();i++) {
				if (i!=u && (S[i]->alpha > 0)) {
					pair.push_back(S[u]);
					pair.push_back(S[i]);
					cerr << " Select by a=0 ";
					return true;
				}
			}
		}
		// second heuristic
		if ((S[u]->alpha > 0) && (S[u]->loss < max_loss->loss - SMO_EPS)) {
			for (unsigned i=0;i<S.size();i++) {
				if (i!=u && (S[i]->loss > S[u]->loss)) {
					pair.push_back(S[u]);
					pair.push_back(S[i]);
					cerr << " Select by a>0 ";
					return true;
				}
			}
		}
	}
	return false;
}

double ComputeDelta(const vector<boost::shared_ptr<HypothesisInfo> >& pair) {
	const double loss0 = pair[0]->features.dot(w) + pair[0]->cost - (pair[0]->oracle->features.dot(w) - pair[0]->oracle->cost);
	const double loss1 = pair[1]->features.dot(w) + pair[1]->cost - (pair[1]->oracle->features.dot(w) - pair[1]->oracle->cost);
	const double num = loss0 - loss1;
	//const double num = pair[0]->loss - pair[1]->loss;
	//cerr << "loss_0=" << pair[0]->loss << " loss_1=" << pair[1]->loss << endl;
	cerr << " ComputeDelta: loss_0=" << loss0 << " loss_1=" << loss1;
	SparseVector<double> diff = pair[0]->features;
	diff -= pair[1]->features;
	double diffsqnorm = diff.l2norm_sq();
	double delta;
	if (diffsqnorm > 0)
		delta = num / (diffsqnorm * lr);
	else
		delta = 0;
	cerr << " delta1=" << delta;
	// clip
	delta = max(-pair[0]->alpha, min(delta, pair[1]->alpha));
	cerr << " delta2=" << delta << endl;
	return delta;

}

void OptimizeSet(const vector<boost::shared_ptr<HypothesisInfo> >& S, const unsigned I=SMO_ITER) {
	cerr << "OptimizeSet with " << S.size() << " constraints:\n";
	// initialize alphas with 0; hope/oracle with 1
	for (unsigned i=0;i<S.size();++i) {
		S[i]->alpha = .0;
		cerr << " S[" << i << "]=" << S[i] << endl;
	}
	S[0]->alpha = 1.0;
	// optimize working set
	unsigned iter = 0;
	double delta,step;
	vector<boost::shared_ptr<HypothesisInfo> > pair;
	while (iter < I) {
		cerr << "SMO " << iter;
		iter++;
		if (!SelectPair(S, pair)) { cerr << endl; return; }
		cerr << " Pair(" << pair[0] << "," << pair[1] << ")\n";
		// compute delta
		delta = ComputeDelta(pair);
		// update alphas
		pair[0]->alpha += delta;
		pair[1]->alpha -= delta;
		for (unsigned i=0;i<S.size();++i) {
			cerr << " S[" << i << "]=" << S[i] << endl;
		}
		step = delta * lr;
		// update weights
		cerr << " step=" << step << " a0=" << pair[0]->alpha << " a1=" << pair[1]->alpha << endl;
		// could stop here when delta=0 => iter = I;
		w += pair[1]->features * step;
		w -= pair[0]->features * step;
		// update losses in working set S
		//for(unsigned i=0;i<S.size();i++) S[i]->computeLoss();
	}
}

int main(int argc, char** argv) {
	po::variables_map cfg;
	if (!init_params(argc,argv,&cfg)) return 1;

	if (cfg.count("random_seed"))
		rng.reset(new MT19937(cfg["random_seed"].as<uint32_t>()));
	else
		rng.reset(new MT19937);

	// set variables
	lr = cfg["learningrate"].as<double>();
	hope_select = cfg["hope"].as<int>();
	fear_select = cfg["fear"].as<int>();
	optimizer = cfg["optimizer"].as<int>();
	freeze = cfg.count("freeze");
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
	scaling = cfg["scaling"].as<int>();
	scalingfactor = cfg["scalingfactor"].as<double>();
	cerr << "scaling="<< scaling << " scalingfactor=" << scalingfactor << endl;

	// setup decoder
	Decoder* decoder = setupDecoder(cfg);
	if (!decoder) {
		cerr << "error while loading decoder with" << cfg["decoder_config"].as<string>() << "!\n";
		return 1;
	}
	TrainingObserver observer;
	// get reference to decoder weights
	vector<weight_t>& decoder_weights = decoder->CurrentWeightVector();
	// the SMT weights (to be optimized)
	if (cfg.count("weights")) {
		Weights::InitFromFile(cfg["weights"].as<string>(), &decoder_weights);
		Weights::InitSparseVector(decoder_weights, &w);
	} else {
		cerr << "starting with EMPTY weights!\n";
	}
	// the weight vector that gives the oracle
	loadRelevanceWeights(cfg["rweights"].as<string>(), relw);
	negrelw -= relw;
	relw_scaled = relw;
	// initial scaling
	if (scaling != 0) scaleRelevanceWeights(scalingfactor);

	// output some vector stats
	cerr << "W_REL=" << relw << endl;
	cerr << "W_REL_SCALED=" << relw_scaled << endl;
	cerr << "|W_REL|=" << relw_scaled.size() << endl;
	cerr << "|W_SMT|=" << w.size() << endl;

	cerr << "hope selection: " << hope_select << endl;
	const string input = decoder->GetConf()["input"].as<string>();
	cerr << "Reading input from " << ((input == "-") ? "STDIN" : input.c_str()) << endl;
	ReadFile in_read(input);
	istream *in = in_read.stream();
	assert(*in);
	string id, sentence;
	int cur_sent = 0;
	unsigned lc = 0; // line count

	double objective=0;
	double tot_loss = 0;
	WeightVector avg_w = w;
	//SparseVector<double> tot;
	//SparseVector<double> oldw = w;
	//tot.clear();
	//tot += w;

	while(*in >> id) {

		in->ignore(1, '\t');
		getline(*in, sentence);
		if (sentence.empty() || id.empty()) continue;

		cerr << "\nID="<<id << endl;
		decoder->SetId(cur_sent);
		decoder->Decode(sentence, &observer); // decode with decoder_weights
		cur_sent = observer.GetCurrentSent();
		Hypergraph hg = observer.GetCurrentForest();

		vector<boost::shared_ptr<HypothesisInfo> > S;
		MAX_REL = std::numeric_limits<double>::lowest();
		MIN_REL = std::numeric_limits<double>::max();

		// get viterbi
		boost::shared_ptr<HypothesisInfo> viterbi = MakeHypothesisInfo(hg);

		// get the true oracle (sets max_rel)
		hg.Reweight(relw);
		boost::shared_ptr<HypothesisInfo> oracle = MakeHypothesisInfo(hg);
		oracle->oracle = oracle;
		oracle->computeCost();

		// get the worst derivation (to get min_rel)
		hg.Reweight(negrelw);
		boost::shared_ptr<HypothesisInfo> worst = MakeHypothesisInfo(hg);
		worst->oracle = oracle;
		worst->computeCost();

		if (hope_select == 1) { // hope
			hg.Reweight(w + relw_scaled);
			S.push_back(MakeHypothesisInfo(hg));
			S[0]->oracle = oracle;
			S[0]->computeCost();
		} else { // true oracle
			S.push_back(oracle);
		}
		// S contains now ONE (hope/oracle) hypothesis
		S[0]->computeLoss();
		boost::shared_ptr<HypothesisInfo> good = S[0];

		viterbi->oracle = oracle;
		viterbi->computeCost();
		viterbi->computeLoss();

		cerr << "min_rel=" << MIN_REL << " max_rel=" << MAX_REL << endl;
		cerr << "S[0]=" << S[0] << endl;

		boost::shared_ptr<HypothesisInfo> fear;

		if (optimizer == 4) { // PA update (single dual coordinate step)
			cerr << "PA MIRA (single dual coordinate step)\n";

			hg.Reweight(w - relw_scaled);
			fear = MakeHypothesisInfo(hg);
			fear->oracle = oracle;
			fear->computeCost();
			fear->computeLoss();
			cerr << "LOSS: " << fear->loss;
			if (fear->loss > 0.0) {
				double diffsqnorm = (good->features - fear->features).l2norm_sq();
				double delta;
				if (diffsqnorm > 0) {
					delta = fear->loss / (diffsqnorm);
					if (delta > lr) delta = lr;
					w += good->features * delta;
					w -= fear->features * delta;
				}
			}

		} else if (optimizer == 1) {// sgd - nonadapted step size
			cerr << "SGD\n";

			if (fear_select == 1) {
				hg.Reweight(w - relw_scaled);
				fear = MakeHypothesisInfo(hg);
			} else if (fear_select == 2) {
				fear = worst;
			} else if (fear_select == 3) {
				fear = viterbi;
			}
			w += good->features * lr;
			w -= fear->features * lr;

		} else if (optimizer == 2) { // PA MIRA with selection from  cutting plane
			cerr << "PA MIRA with Selection from Cutting Plane\n";

			hg.Reweight(w - relw_scaled);
			fear = MakeHypothesisInfo(hg);
			fear->oracle = oracle;
			fear->computeCost();
			fear->computeLoss();
			if (fear->loss < 0) {
				cerr << "FEAR LOSS < 0! THIS SHOULD NOT HAPPEN!\n";
				abort();
			}
			if (fear->loss > good->loss + SMO_EPS) {
				S.push_back(fear);
				OptimizeSet(S, 1); // only one iteration with a set of two constraints
			} else { cerr << "constraint not violated. fear loss:" << fear->loss << "\n"; }

		} else if (optimizer == 3) { // Cutting Plane MIRA
			cerr << "Cutting Plane MIRA\n";

			unsigned cp_iter=0; // Cutting Plane Iteration
			bool again = true;
			while (again && cp_iter<CP_ITER) {
				again = false;
				cerr << "CuttingPlane: " << cp_iter << endl;
				// find a fear derivation
				hg.Reweight(w - relw_scaled);
				fear = MakeHypothesisInfo(hg);
				fear->oracle = oracle;
				fear->computeCost();
				fear->computeLoss();
				if (fear->loss < 0) {
					cerr << "FEAR LOSS < 0! THIS SHOULD NOT HAPPEN!\n";
					//abort();
				}
				// find max loss hypothesis
				double max_loss_in_set = (*std::max_element(S.begin(), S.end(), lossComp))->loss;
				if (fear->loss > max_loss_in_set + SMO_EPS) {
					cerr << "Adding new fear " << fear << " to S\n";
					S.push_back(fear);
					OptimizeSet(S);
					again = true;
				} else { cerr << "constraint not violated. fear loss:" << fear->loss << "\n"; }
				cp_iter++;
				// update losses
				//for(unsigned i=0;i<S.size();i++) S[i]->computeLoss();
			}
		}

		cerr << "|W|=" << w.size() << endl;
		tot_loss += relscale(viterbi->rel);
		//print objective after this sentence
		//double w_change = (w - oldw).l2norm_sq();
		//double temp_objective = 0.5 * w_change;// + max_step_size * max_fear;
		for(int u=0;u!=S.size();u++) {
			cerr << "alpha=" << S[u]->alpha << " loss=" << S[u]->loss << endl;
			//temp_objective += S[u]->alpha * S[u]->loss;
		}
		//objective += temp_objective;
		//cerr << "SENT OBJ: " << temp_objective << " NEW OBJ: " << objective << endl;

		//tot += w;
		++lc;
		avg_w *= lc;
		avg_w = (w + avg_w) / (lc+1);

		// set decoder weights for next sentence
		decoder_weights.clear();
		w.init_vector(&decoder_weights);
		// rescale relevance weights to balance with new model after the update
		if (scaling == 2) {
			scaleRelevanceWeights(scalingfactor);
			cerr << "W_REL_SCALED=" << relw_scaled << endl;
		}

		// viterbi 2 for debugging
		//hg.Reweight(w);
		//boost::shared_ptr<HypothesisInfo> viterbi2 = MakeHypothesisInfo(hg);
		//viterbi2->oracle = oracle;
		//viterbi2->computeCost();
		//viterbi2->computeLoss();
		//fear->computeLoss();
		//viterbi->computeLoss();
		//good->computeLoss();
		cerr << "FEAR : " << fear << " \n" << TD::GetString(fear->hyp) << endl;
		cerr << "BEST : " << viterbi << " \n" << TD::GetString(viterbi->hyp) << endl;
		//cerr << "BEST2: " << viterbi2 << " \n" << TD::GetString(viterbi2->hyp) << endl;
		cerr << "HOPE : " << good << " \n" << TD::GetString(good->hyp) << endl;

		cout << id << " ||| " << TD::GetString(fear->hyp) << " ||| " << TD::GetString(viterbi->hyp) << " ||| " << TD::GetString(good->hyp) << endl;

		S.clear();
		fear.reset();
		viterbi.reset();
		//viterbi2.reset();
		good.reset();
		worst.reset();
		oracle.reset();

	}

    //cerr << "FINAL OBJECTIVE: "<< objective << endl;
    cerr << "Translated " << lc << " sentences\n";
    cerr << " [AVG METRIC LAST PASS=" << (tot_loss / lc) << "]\n";
    //tot_loss = 0;

	decoder_weights.clear();
	w.init_vector(&decoder_weights);
	//Weights::ShowLargestFeatures(decoder_weights);
	// write weights
	int node_id = rng->next() * 100000;
	cerr << " Writing model to " << node_id << endl;
	ostringstream os;
	os << cfg["weights_output"].as<string>() << "/last." << node_id;
	string msg = "HGMIRA tuned weights ||| " + boost::lexical_cast<std::string>(node_id) + " ||| " + boost::lexical_cast<std::string>(lc);
	Weights::WriteToFile(os.str(), decoder_weights, true, &msg);

	//SparseVector<double> x = tot;
	//x /= lc+1;
	ostringstream sa;
	string msga = "HGMIRA tuned weights AVERAGED ||| " + boost::lexical_cast<std::string>(node_id) + " ||| " + boost::lexical_cast<std::string>(lc);
	sa << cfg["weights_output"].as<string>() << "/avg." << node_id;
	avg_w.init_vector(&decoder_weights);
	Weights::WriteToFile(sa.str(), decoder_weights, true, &msga);


	delete decoder;
	cerr << "\ndone.\n";
	return 0;

}

