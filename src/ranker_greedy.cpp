#include "ranker.h"

using namespace CLIR;

Decoder* setupDecoder(const po::variables_map& cfg) {
	register_feature_functions();
	SetSilent(true);
	ReadFile ini_rf(cfg["decoder_config"].as<string>());
	return new Decoder(ini_rf.stream());
}

bool init_params(int argc, char** argv, po::variables_map* conf) {
	po::options_description opts;
	opts.add_options()
		("decoder_config,c",po::value<string>(),"Decoder configuration file")
		("weights,w", po::value<string>(), "initial weights")
		("K,k", po::value<int>(), "* Keep track of K-best documents per query. (Number of results per query)")
		("jobs,j", po::value<int>()->default_value(1), "Number of threads. Default: number of cores")
		("documents,d", po::value<string>(), "* File containing document tf vectors.")
		("dftable", po::value<string>(), "table containing the df values");
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
	if(conf->count("help") || !conf->count("decoder_config")) {
		cerr << dcmdline_options << endl;
		return false;
	}
	return true;
}

int main(int argc, char** argv) {
	po::variables_map cfg;
	if (!init_params(argc,argv,&cfg)) return 1;
	// setup decoder
	Decoder* decoder = setupDecoder(cfg);
	if (!decoder) {
		cerr << "error while loading decoder with" << cfg["decoder_config"].as<string>() << "!\n";
		return 1;
	}
	TrainingObserver observer;
	// get reference to decoder weights
	vector<weight_t>& decoder_weights = decoder->CurrentWeightVector();
	WeightVector w;
	// the SMT weights (to be optimized)
	if (cfg.count("weights")) {
		Weights::InitFromFile(cfg["weights"].as<string>(), &decoder_weights);
		Weights::InitSparseVector(decoder_weights, &w);
	} else {
		cerr << "starting with EMPTY weights!\n";
	}

	// load DfTable
	DfTable dft(cfg["dftable"].as<string>());
	cerr << "CLIR-PARALLEL::DF table loaded (" << dft.size() << " entries)." << endl;
	// load document collection
	vector<CLIR::Document> documents;
	double avgDocLen = CLIR::loadDocuments(cfg["documents"].as<string>(), documents);
	size_t N = documents.size();
	if (N < dft.mMaxDf) {
		cerr << "CLIR-PARALLEL::maxDf=" << dft.mMaxDf << " > N=" << N << ", setting N to maxDf.\n";
		N = dft.mMaxDf;
	}

	GreedyScorer gscorer(&dft, N, avgDocLen);

	int J = cfg["jobs"].as<int>();
	omp_set_num_threads(J);
	int K = cfg["K"].as<int>();
	TIMER::timestamp_t t0, t1;
	double time;
	string id, sentence;
	while(cin >> id) {

		cin.ignore(1, '\t');
		getline(cin, sentence);
		if (sentence.empty() || id.empty()) continue;

		cerr << "\nQ="<<id<<endl;
		decoder->Decode(sentence, &observer); // decode with decoder_weights
		Hypergraph hg = observer.GetCurrentForest();

		t0 = TIMER::get_timestamp();

		//gscorer.ComputeInsideScores(hg);
		//gscorer.ComputeViterbiInsideScores(hg);
		gscorer.ComputeEdgeMarginals(hg);
		//gscorer.ComputeNormalizedInsideScores(hg);

		vector<Scores> scores(J, Scores(K));
		#pragma omp parallel for
		for ( int d=0;d<N;++d ) {
			CLIR::Document& doc = documents[d];
			prob_t s = gscorer.Score(hg, doc);
			scores[omp_get_thread_num()].update( CLIR::Score(doc.id_, s));
		}
		// merge the heaps
		Scores& result = scores[0]; // master thread scores
		for (vector<Scores>::iterator sit=scores.begin()+1;sit!=scores.end();++sit) result.update(*sit);
		CLIR::writeResult(cout, id, result.k_largest(), "0");

		t1 = TIMER::get_timestamp();
		time = (t1-t0) / 1000000.0L;
		cerr << time << "s\n";
	}


	delete decoder;
}
