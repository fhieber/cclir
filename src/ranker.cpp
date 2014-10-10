#include "ranker.h"
#include "clir-scoring.h"
#include <climits>

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
		("ir_weights", po::value<string>(), "ir weights")
		("default_ir_weight", po::value<double>()->default_value(0.0), "default ir weight for term match")
		("word_classes", po::value<string>()->default_value(""), "a file containing word2class mappings for extended matching")
		("K,k", po::value<int>(), "* Keep track of K-best documents per query. (Number of results per query)")
		("jobs,j", po::value<int>()->default_value(1), "Number of threads. Default: number of cores")
		("documents,d", po::value<string>(), "* File containing document classic BM25 vectors.")
		("dftable", po::value<string>(), "* DF table to load")
		("beam_size,b", po::value<unsigned>()->default_value(10000), "# of edges evaluated at each node during document scoring.")
		("sort_edges", po::value<bool>()->zero_tokens(), "whether to sort incoming edges at each node by upper bound Viterbi score.")
		//("density_prune", po::value<double>(), "Pass 1 pruning: keep no more than this many times the number of edges used in the best derivation tree (>=1.0)")
		//("beam_prune", po::value<double>(), "Pass 1 pruning: Prune paths from scored forest, keep paths within exp(alpha>=0)")
		("bound", po::value<bool>()->zero_tokens(), "use bounding techniques for inference.")
		("ideal", po::value<bool>()->zero_tokens(), "DEBUGGING: compute true upper bounds for each doc (SUPERSLOW. only single-threaded!)")
		("dense", po::value<bool>()->zero_tokens(), "score with dense BM25 feature INSTEAD of sparse features. (should give the same scores)")
		("quiet", po::value<bool>()->zero_tokens(), "quiet ranker")
		("result_separator", po::value<string>()->default_value("\n"), "output results per query with custom separator and not newlines (for parallelize.pl)")
		("stopwords,s",po::value<string>(),	"stopword file")
		("qrels", po::value<string>(), "if specified, will compute IR metrics directly on the output list. No trec_eval required.");

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

	omp_set_num_threads(cfg["jobs"].as<int>());

	// result separator
	const string separator = cfg["result_separator"].as<string>();

	// load stopwords
	if (cfg.count("stopwords")) {
		sw::loadStopwords(cfg["stopwords"].as<string>());
		cerr << sw::stopwordCount() << " stopwords loaded from '"
			 << cfg["stopwords"].as<string>() << "'\n";
	}
	// setup decoder
	Decoder* decoder = setupDecoder(cfg);
	if (!decoder) {
		cerr << "error while loading decoder with" << cfg["decoder_config"].as<string>() << "!\n";
		abort();
	}
	// get reference to decoder weights
	vector<weight_t>& decoder_weights = decoder->CurrentWeightVector();
	WeightVector w_smt;
	// the SMT weights
	if (cfg.count("weights")) {
		Weights::InitFromFile(cfg["weights"].as<string>(), &decoder_weights);
		Weights::InitSparseVector(decoder_weights, &w_smt);
	} else { cerr << "starting with EMPTY weights!\n"; }
	WeightVector w_ir;
	if (cfg.count("ir_weights")) {
		vector<weight_t> dirw;
		Weights::InitFromFile(cfg["ir_weights"].as<string>(), &dirw);
		Weights::InitSparseVector(dirw, &w_ir);
	}

	DfTable dft (cfg["dftable"].as<string>());
	cerr << "DF table loaded (" << dft.size() << " entries)." << endl;
	// IR feature indicator/extractor
	IRFeatureExtractor extractor(&w_ir, cfg["default_ir_weight"].as<double>(), cfg["word_classes"].as<string>());
	IRFeatureIndicator indicator(&extractor, &dft);
	// load documents
	vector<CLIR::FDocument> documents;
	const double avg_len = CLIR::loadDocuments(cfg["documents"].as<string>(), documents, &extractor);

	TestRanker ranker(
			decoder,
			&indicator,
			&w_smt,
			&w_ir,
			cfg.count("dense"),
			cfg.count("quiet"));
	ranker.K = cfg["K"].as<int>();
	ranker.J = cfg["jobs"].as<int>();
	ranker.bound = cfg.count("bound");
	ranker.ideal_bounds = cfg.count("ideal");
	ranker.avg_len = avg_len;
	ranker.beam_size = cfg["beam_size"].as<unsigned>();
	ranker.sort_edges = cfg.count("sort_edges");
	ranker.setDocuments(&documents);

	cerr << "beam per incoming edge set to " << ranker.beam_size << endl;

	string id, sentence;
	vector<CLIR::Score> results;
	double q_start, q_stop, q_time, total_time;
	unsigned Q=0;
	CLIR::IRScorer* irs = NULL;
	if (cfg.count("qrels")) {
		irs = new CLIR::IRScorer(cfg["qrels"].as<string>());
	}
	while(cin >> id) {

		cin.ignore(1, '\t');
		getline(cin, sentence);
		if (sentence.empty() || id.empty()) continue;

		q_start = omp_get_wtime();
		ranker.rank(id, sentence, results);
		q_stop = omp_get_wtime();
		q_time = q_stop - q_start;
		total_time += q_time;
		cerr << "SPEED: Query processing: " << q_time << endl;

		ostringstream ss;
		if (irs) {
			CLIR::writeResult(ss, id, results, "0", true);
			irs->evaluateSegment(ss.str());
		} else {
			CLIR::writeResult(ss, id, results, "0", true, separator);
			cout << ss.str();
			if(separator != "\n") {
				cout << "\n";
			}
		}

		++Q;

	}
	if (irs) {
		cout << "\nEvaluation:\n===========\n";
		cout.precision(4);
		cout << " queries\t" << irs->N() << endl;
		cout << " num_rel_ret\t" << irs->NUMRELRET() << endl;
		cout << " MAP \t\t" << irs->MAP() << endl;
		cout << " NDCG\t\t" << irs->NDCG() << endl;
	}
	delete irs;
	delete decoder;

	cerr << "SPEED: Total time: " << total_time << endl
		 << "SPEED: avg. time/query: " << total_time / Q << endl;

}

