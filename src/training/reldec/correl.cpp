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
#include "kbest.h"

#include "weights.h"
#include "feature_vector.h" // typedefs FeatureVector,WeightVector, DenseWeightVector

using namespace std;
namespace po = boost::program_options;

boost::shared_ptr<MT19937> rng;

bool init_params(int argc, char** argv, po::variables_map* conf) {
	po::options_description opts("Configuration options:");
	opts.add_options()
		("decoder_config,c",po::value<string>(),"Decoder configuration file")
		("weights,w", po::value<string>(), "* mt model")
		("rweights,r", po::value<string>(), "* relevance weights ")
		("random_seed,S", po::value<uint32_t>(), "Random seed (if not specified, /dev/random will be used)")
		("sample", po::value<int>(), "sample x times fear and hope from forest for each sentence.")
		("kbest", po::value<int>(), "get kbest lists for best/worst of model/relevance score");
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
	if(conf->count("help") || !conf->count("decoder_config") || !conf->count("weights") || !conf->count("rweights") ) {
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

void loadWeights(const string& fname, WeightVector& w) {
	DenseWeightVector dw;
	Weights::InitFromFile(fname, &dw);
	Weights::InitSparseVector(dw, &w);
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


inline double vscale(double v, double vmin, double vmax) { return (v-vmin) / (vmax-vmin); }

int main(int argc, char** argv) {
	po::variables_map cfg;
	if (!init_params(argc,argv,&cfg)) return 1;

	if (cfg.count("random_seed"))
		rng.reset(new MT19937(cfg["random_seed"].as<uint32_t>()));
	else
		rng.reset(new MT19937);


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
	loadWeights(cfg["rweights"].as<string>(), w_hope);
	WeightVector w_inv = w*-1;
	WeightVector w_hope_inv = w_hope*-1;

	//cerr << "W    " << w << endl;
	//cerr << "WINV " << w_inv << endl;
	//cerr << "R    " << w_hope << endl;
	//cerr << "RINV " << w_hope_inv << endl;

	const string input = decoder->GetConf()["input"].as<string>();
	//cerr << "Reading input from " << ((input == "-") ? "STDIN" : input.c_str()) << endl << endl;
	ReadFile in_read(input);
	istream *in = in_read.stream();
	assert(*in);
	string id, sentence;
	std::vector<HypergraphSampler::Hypothesis> samples;

	while(*in >> id) {

		in->ignore(1, '\t');
		getline(*in, sentence);
		if (sentence.empty() || id.empty()) continue;

		//decoder->SetId(id);
		decoder->Decode(sentence, &observer); // decode with decoder_weights
		Hypergraph hg = observer.GetCurrentForest();

		// get max model score
		double max_tscore = ViterbiFeatures(hg).dot(w);
		// get min model score
		hg.Reweight(w_inv);
		double min_tscore = -ViterbiFeatures(hg).dot(w_inv);
		// get max rel score
		hg.Reweight(w_hope);
		double max_rscore = ViterbiFeatures(hg).dot(w_hope);
		// get min rel_score
		hg.Reweight(w_hope_inv);
		double min_rscore = -ViterbiFeatures(hg).dot(w_hope_inv);

		//cerr << max_tscore << " " << min_tscore << " " << max_rscore << " " << min_rscore << endl;

		if (cfg.count("sample")) {

			HypergraphSampler::sample_hypotheses(hg, cfg["sample"].as<int>(), &(*rng), &samples);
			for (unsigned s=0;s<samples.size();++s) {
				const HypergraphSampler::Hypothesis& h = samples[s];
				cout << id << "\t" << "S\t" << vscale(h.fmap.dot(w), min_tscore, max_tscore) <<
						"\t" <<  vscale(h.fmap.dot(w_hope), min_rscore, max_rscore) <<
						"\t" << TD::GetString(h.words) << endl;
			}

		} else if (cfg.count("kbest")) {
			typedef KBest::KBestDerivations<vector<WordID>, ESentenceTraversal,KBest::FilterUnique> K;
			// get kbest model score derivations
			hg.Reweight(w);
			K kbest2(hg,cfg["kbest"].as<int>());
			for (int i = 0; i < cfg["kbest"].as<int>(); ++i) {
			      typename K::Derivation *d = kbest2.LazyKthBest(hg.nodes_.size() - 1, i);
			      if (!d) break;
			      cout << id << "\t" << "KBT\t" << vscale(d->feature_values.dot(w), min_tscore, max_tscore) <<
						"\t" <<  vscale(d->feature_values.dot(w_hope), min_rscore, max_rscore) <<
						"\t" << TD::GetString(d->yield) << endl;
			}

			// get kworst model score derivations
			hg.Reweight(w_inv);
			K kbest3(hg,cfg["kbest"].as<int>());
			for (int i = 0; i < cfg["kbest"].as<int>(); ++i) {
			      typename K::Derivation *d = kbest3.LazyKthBest(hg.nodes_.size() - 1, i);
			      if (!d) break;
			      cout << id << "\t" << "KWT\t" << vscale(d->feature_values.dot(w), min_tscore, max_tscore) <<
						"\t" <<  vscale(d->feature_values.dot(w_hope), min_rscore, max_rscore) <<
						"\t" << TD::GetString(d->yield) << endl;
			}

			// get kbest rel score derivations
			hg.Reweight(w_hope);
			K kbest4(hg,cfg["kbest"].as<int>());
			for (int i = 0; i < cfg["kbest"].as<int>(); ++i) {
			      typename K::Derivation *d = kbest4.LazyKthBest(hg.nodes_.size() - 1, i);
			      if (!d) break;
			      cout << id << "\t" << "KBR\t" << vscale(d->feature_values.dot(w), min_tscore, max_tscore) <<
						"\t" <<  vscale(d->feature_values.dot(w_hope), min_rscore, max_rscore) <<
						"\t" << TD::GetString(d->yield) << endl;
			}

			// get kbest model score derivations
			hg.Reweight(w_hope_inv);
			K kbest(hg,cfg["kbest"].as<int>());
			for (int i = 0; i < cfg["kbest"].as<int>(); ++i) {
			      typename K::Derivation *d = kbest.LazyKthBest(hg.nodes_.size() - 1, i);
			      if (!d) break;
			      cout << id << "\t" << "KWR\t" << vscale(d->feature_values.dot(w), min_tscore, max_tscore) <<
						"\t" <<  vscale(d->feature_values.dot(w_hope), min_rscore, max_rscore) <<
						"\t" << TD::GetString(d->yield) << endl;
			}

		}


	}

	delete decoder;
	return 0;

}

