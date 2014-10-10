#include "clir.h"
#include "clir-scoring.h"
#include "distclir.h"

bool init_params(int argc, char** argv, po::variables_map* cfg) {
	po::options_description cl("\nPerforms bm25-based batch retrieval by connecting to a clir-server at the given port.\nCommand Line Options");
	cl.add_options()
			("queries,q", po::value<string>()->default_value("-"), "queries (Format: ID TAB TFV or ID TAB PSQ). Default input: STDIN")
			("psq,p", po::value<bool>()->zero_tokens(), "indicate if queries are Probabilistic Structured Queries (PSQs)")
			("K,k", po::value<int>()->default_value(10), "* Keep track of K-best documents per query. (Number of results per query)")
			("run-id,r", po::value<string>()->default_value("1"), "run id shown in the output")
			("servers,s", po::value<vector<string> >()->multitoken(), "list of server connections, e.g. localhost:5555, donna:5558,...")
			("qrels", po::value<string>(), "if specified, the client will compute IR metrics directly on the output list. No trec_eval required.")
			("text,t", po::value<bool>()->zero_tokens(), "if specified, input is raw text: ID TAB TEXT (NO PSQ!)")
			("help,h",	po::value<bool>()->zero_tokens(),	"print this help message.");			
	po::store(parse_command_line(argc, argv, cl), *cfg);
	po::notify(*cfg);
	
	if (cfg->count("help")) {
		cerr << cl << endl;
		return false;
	}
	return true;
}

int main(int argc, char** argv) {
	// handle parameters
	po::variables_map cfg;
	if (!init_params(argc,argv, &cfg)) exit(1); // something is wrong
	
	vector<string> servers = {"localhost:5555"};
	if (cfg.count("servers")) servers = cfg["servers"].as<vector<string> >();
		
	int k = cfg["K"].as<int>();
	string runid = cfg["run-id"].as<string>();
	bool psq = cfg.count("psq");
	
	CLIR::IRScorer* irs = 0;
	if (cfg.count("qrels")) irs = new CLIR::IRScorer(cfg["qrels"].as<string>());

	if (cfg.count("text")) cerr << "CLIENT::queries as text.\n";
    // prepare our context and socket
    zmq::context_t context (1);
    zmq::socket_t* client = DISTCLIR::setup_socket(context, servers);

	ReadFile input(cfg["queries"].as<string>());
	if (cfg["queries"].as<string>() == "-") cerr << "CLIENT::reading queries from STDIN.\n";
	string line,reply;
	cerr << "CLIENT::sending queries ";
	int i=0;
	TIMER::timestamp_t t0, t1;
	t0 = TIMER::get_timestamp();
	while(getline(*input, line)) {
		if (line.empty()) continue;
		if (cfg.count("text")) {
			CLIR::Document d(line);
			if (!d.parsed_) cerr << "CLIENT::Warning: query is malformed!\n";
			d.computeTfVector(false);
			line = d.asVector();
		}
		DISTCLIR::run_query(DISTCLIR::construct_msg(runid, k, psq, line), client, reply);
		if (cfg.count("qrels")) {
			irs->evaluateSegment(reply);
		} else {
			cout << reply;
		}
		if (i%100==0) cerr << ".";
		++i;
		
	}
	t1 = TIMER::get_timestamp();
	double time = (t1-t0) / 1000000.0L;
	cerr << "\nCLIENT::done (" << time << "s)\n";
	if (cfg.count("qrels")) {
		cout << "\nEvaluation:\n===========\n";
		cout.precision(4);
		cout << " queries\t" << irs->N() << endl;
		cout << " num_rel_ret\t" << irs->NUMRELRET() << endl;
		cout << " MAP \t\t" << irs->MAP() << endl;
		cout << " NDCG\t\t" << irs->NDCG() << endl;
	}
	delete client;
	delete irs;
	return 0;
	
}
