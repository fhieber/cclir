#include "src/core/clir.h"

using namespace CLIR;

bool init_params(int argc, char** argv, po::variables_map* cfg) {
	po::options_description cl("\nPerforms bm25-based batch retrieval of input queries (DT vectors OR PSQs). Queries are loaded into memory. Documents from STDIN.\nCommand Line Options");
	cl.add_options()
			("queries,q", po::value<string>(), "* File containing Queries")
			("psq,p", po::value<bool>()->zero_tokens(), "indicate if queries are Probabilistic Structured Queries (PSQs)")			
			("dftable,d", po::value<string>(), "* table containing the df values")
			("K,k", po::value<int>(), "* Keep track of K-best documents per query. (Number of results per query)")
			("N,n", po::value<int>(), "Number of documents in collection")
			("avg_len,a", po::value<double>(), "* Average length of documents (required for BM25)")
			("run-id,r", po::value<string>()->default_value("1"), "run id shown in the output")
			//("show-empty-results",po::value<bool>()->zero_tokens(), "set this if you want to include dummy output for proper scoring (default false)")
			("metric,m", po::value<string>()->default_value("classicbm25"), "Scoring function (classicbm25, bm25, classicbm25tf, tfidf, stfidf). default is classicbm25. If metric is tfidf/stfidf, cosine is calculated.")
			("no-qtf", po::value<bool>()->zero_tokens(), "do not use query term frequency for bm25 scoring");

	po::store(parse_command_line(argc, argv, cl), *cfg);
	po::notify(*cfg);

	if( !cfg->count("queries") || !cfg->count("K") || !cfg->count("dftable") || !cfg->count("avg_len") || !cfg->count("N") ) {
		cerr << cl << endl;
		return false;
	}

	return true;
}

int main(int argc, char** argv) {

	// handle parameters
	po::variables_map cfg;
	if (!init_params(argc,argv, &cfg)) exit(1); // something is wrong

	// setup DfTable
	DfTable dft (cfg["dftable"].as<string>());
	cerr << "DF table loaded (" << dft.size() << " entries).\n";

	// load queries
	vector<Query> queries;
	cerr << "reporter:status:loading queries...\n";
	CLIR::loadQueries(cfg["queries"].as<string>(), queries, cfg.count("psq"));
	size_t queries_size = queries.size();
	size_t N = cfg["N"].as<int>();
	double avgDocLen = cfg["avg_len"].as<double>();

	Scorer* scorer = setupScorer(cfg["metric"].as<string>(), N, avgDocLen, &dft);

	const bool no_qtf = cfg.count("no-qtf");
	
	string run_id = cfg["run-id"].as<string>();

	// initialize vector of Scores objects for each query
	vector<Scores> scores (queries_size, Scores(cfg["K"].as<int>()));

	// for each doc
	string docid;
	int len;
	string raw;
	TermVector doc;
	int c = 0;

	cerr << "reporter:status:scanned=0\n";

	while (cin >> docid) {
		cin.ignore(1,'\t');
		cin >> len;
		cin.ignore(1,'\t');
		getline(cin, raw);

		if (docid.size() == 0 || len <= 0 || raw.size() == 0)
			continue;

		Document doc(vutils::read_vector(raw), docid, len);

		// for each query compute score between current document and query
		for ( size_t i = 0 ; i < queries_size ; ++i ) {
			scores[i].update( Score(doc.id_, scorer->score(queries[i], doc, no_qtf) ) );
		}

		c++;
		c%10==0 ? cerr << "reporter:status:scanned="<< c << "\n" : cerr ;

	}

	cerr << "\nreporter:status:outputting kbest lists..." << endl;
	for ( size_t i = 0; i < scores.size(); ++i ) {
		CLIR::writeResult(std::cout, queries.at(i), scores[i].k_largest(), run_id);
	}

	cerr << "reporter:status:done.\n";

	delete scorer;

}

