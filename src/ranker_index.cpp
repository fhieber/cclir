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
		("index,i", po::value<string>(), "* inverted index");
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
	//index
	ReadFile idx_in(cfg["index"].as<string>());
	const Index::Index idx(*idx_in);
	cerr << idx << endl;

	ViterbiScorer vs(&idx);

	// load rank weights
	SparseVector<double> rankweights;

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
		int len = -1;
		prob_t vit = Viterbi<PathLengthTraversal>(hg, &len);

		t0 = TIMER::get_timestamp();

		vector<prob_t> scores(idx.NumberOfDocuments());
		vs.Score(hg, &scores);
		Scores kbest(K);
		unsigned below_lb=0;
		unsigned lb=0;
		unsigned above_lb=0;
		prob_t max = prob_t::Zero();
		prob_t min = prob_t(100000000);
		for (int d=0;d<idx.NumberOfDocuments();++d) {
			if (scores[d] < vit) below_lb++;
			else if (scores[d] == vit) lb++;
			else above_lb++;
			if (scores[d]>max) max = scores[d];
			if (scores[d]<min) min = scores[d];
			kbest.update( CLIR::Score(idx.GetDocID(d), scores[d]));
		}
		CLIR::writeResult(cout, id, kbest.k_largest(), "0");

		t1 = TIMER::get_timestamp();
		time = (t1-t0) / 1000000.0L;
		cerr << time << "s\n";
		cerr << "<V: " << below_lb << " =V: " << lb << " >V: " << above_lb << endl;
		cerr << "max score: " << max << " min score: " << min << " viterbi score: " << vit << endl;
	}


	delete decoder;
}
