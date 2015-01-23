#include "src/core/clir.h"
#include "src/core/clir-scoring.h"

#ifdef _OPENMP
	#include <omp.h>
#else
	#define omp_get_thread_num() 0
#endif

using namespace CLIR;

bool init_params(int argc, char** argv, po::variables_map* cfg) {
	po::options_description cl("\nPerforms batch retrieval for input queries (DT vectors OR PSQs). Document repository can either come from STDIN (default, requires total number of documents and avg length; suitable for HADOOP), load documents into memory (-documents), or use an inverted index (-index). Uses OpenMP if available.\nCommand Line Options");
	cl.add_options()
			("queries,q", po::value<string>(), "* File containing Queries.")
			("psq,p", po::value<bool>()->zero_tokens(), "if queries are Probabilistic Structured Queries (PSQs)")
			("text-input,t", po::value<bool>()->zero_tokens(), "if queries are 'id TAB text' instead of 'id TAB length TAB vector' (does not work with --psq)")
			("index,i", po::value<string>(), "use inverted index as document repository")
			("documents,c", po::value<string>(), "use in-memory document repository")
			("dftable,d", po::value<string>(), "table containing the df values if -c is specified")
			("K,k", po::value<int>()->default_value(1000), "* Keep track of K-best documents per query. (Number of results per query)")
			("N,n", po::value<int>(), "if documents from STDIN: Number of documents in collection")
			("avg_len,a", po::value<double>(), "if documents from STDIN: Average length of documents (required for BM25)")
			("run-id,r", po::value<string>()->default_value("1"), "run id shown in the output")
			#ifdef _OPENMP
			("jobs,j", po::value<int>(), "Number of threads. Default: number of cores")
			#else
			#endif
			("model,m", po::value<string>()->default_value("classicbm25"), "Scoring function (bm25, classicbm25, classicbm25tf, tfidf, stfidf). default is classicbm25. If metric is tfidf/stfidf, cosine is calculated.")
			("no-qtf", po::value<bool>()->zero_tokens(), "do not use query term frequency for bm25 scoring")
			("qrels", po::value<string>(), "if specified, will compute IR metrics directly on the output list. No trec_eval required.")
			("output,o", po::value<string>()->default_value("-"), "* File for output. Default STDOUT");
	po::store(parse_command_line(argc, argv, cl), *cfg);
	po::notify(*cfg);
	if(!cfg->count("queries")) {
		cerr << cl << "\n\nPlease specify set of queries (-q)\n";
		return false;
	}
	if (!cfg->count("documents") && !cfg->count("index")) {
		if (!cfg->count("N")) {
			cerr << cl << "\n\nPlease specify total number of documents to be read (-n)\n";
			return false;
		}
		if (!cfg->count("avg_len")) {
			cerr << cl << "\n\nPlease specify average length of documents in the repository (-a)\n";
			return false;
		}
	}
	if (cfg->count("documents") && cfg->count("index")) {
		cerr << cl << "\n\nPlease specify either documents (-c) OR an inverted index (-i)!\n";
		return false;
	}
	if(!cfg->count("index") && !cfg->count("dftable")) {
		cerr << cl << "\n\nPlease specify a document frequency table (-d)\n";
		return false;
	}
	return true;
}

int main(int argc, char** argv) {

	// handle parameters
	po::variables_map cfg;
	if (!init_params(argc,argv, &cfg)) exit(1); // something is wrong

	// set number of jobs
	#ifdef _OPENMP
	if (cfg.count("jobs")) {
		omp_set_num_threads(cfg["jobs"].as<int>());
		cerr << "CLIR::Using " << cfg["jobs"].as<int>() << " threads.\n"
	}
	#else
	#endif

	const bool no_qtf = cfg.count("no-qtf");
	string run_id = cfg["run-id"].as<string>();
	const int K = cfg["K"].as<int>();

	// load queries
	vector<Query> queries;
	CLIR::loadQueries(cfg["queries"].as<string>(), queries, cfg.count("psq"), cfg.count("text-input"));
	size_t queries_size = queries.size();
	size_t done = queries_size / 10.0;
	cerr.precision(1);

	// setup result vector
	vector<vector<Score> > results(queries_size);

	if (cfg.count("index")) {

		Index::Index idx;
		Index::LoadIndex(cfg["index"].as<string>(), idx);
		IndexScorer* ranker = CLIR::setupIndexScorer(cfg["model"].as<string>(), &idx);
		cerr << "CLIR::Retrieving from in-memory inverted index ...";
		# pragma omp parallel for
		for ( int i = 0 ; i < queries_size ; ++ i ) {
			vector<double> scores(idx.NumberOfDocuments(), .0);
			ranker->score(queries[i], scores, no_qtf);
			Scores kbest(K);
			for (int d=0;d<idx.NumberOfDocuments();++d) {
				if (scores[d]>0) {
					kbest.update( Score(idx.GetDocID(d), prob_t(scores[d]) ) );
				}
			}
			results[i] = kbest.k_largest();
			i%done == 0 ? cerr << done/queries_size << "%.." : cerr ;
		}
		delete ranker;
		cerr << "ok.\n";

	} else if (cfg.count("documents")) {

		// load DfTable
		DfTable dft(cfg["dftable"].as<string>());
		cerr << "CLIR::DF table loaded (" << dft.size() << " entries)." << endl;
		// load document collection
		vector<CLIR::Document> documents;
		double avgDocLen = CLIR::loadDocuments(cfg["documents"].as<string>(), documents);
		size_t N = documents.size();
		if (N < dft.mMaxDf) {
			cerr << "CLIR::maxDf=" << dft.mMaxDf << " > N=" << N << ", setting N to maxDf.\n";
			N = dft.mMaxDf;
		}

		Scorer* ranker = CLIR::setupScorer(cfg["model"].as<string>(), N, avgDocLen, &dft);
		cerr << "CLIR::Retrieving from in-memory documents ...";
		# pragma omp parallel for
		for ( int i = 0 ; i < queries_size ; ++ i ) {
			const Query& query = queries[i];
			Scores scores(K);
			for ( int d = 0 ; d < N ; ++d ) {
				Document& doc = documents[d];
				scores.update( Score(doc.id_, ranker->score(query, doc, no_qtf) ) );
			}
			results[i] = scores.k_largest();
			i%done == 0 ? cerr << done/queries_size << "%.." : cerr ;
		}
		delete ranker;
		cerr << "ok.\n";

	} else {

		// load DfTable
		DfTable dft(cfg["dftable"].as<string>());
		cerr << "CLIR::DF table loaded (" << dft.size() << " entries)." << endl;
		size_t N = cfg["N"].as<int>();
		if (N < dft.mMaxDf) {
			cerr << "CLIR::maxDf=" << dft.mMaxDf << " > N=" << N << ", setting N to maxDf.\n";
			N = dft.mMaxDf;
		}
		double avgDocLen = cfg["avg_len"].as<double>();
		cerr << "CLIR::Retrieving from STDIN documents ...";
		
		Scorer* ranker = CLIR::setupScorer(cfg["model"].as<string>(), N, avgDocLen, &dft);
		vector<Scores> scores (queries_size, Scores(K)); // score vectors for each query
		string docid, raw;
		int len, c = 0;
		cerr << "reporter:status:scanned="<< c << "\n";
		TermVector doc;
		while (cin >> docid) {
			cin.ignore(1,'\t');
			cin >> len;
			cin.ignore(1,'\t');
			getline(cin, raw);

			if (docid.size() == 0 || len <= 0 || raw.size() == 0)
				continue;

			Document doc(vutils::read_vector(raw), docid, len);

			// for each query compute score between current document and query
			# pragma omp parallel for
			for ( size_t i = 0 ; i < queries_size ; ++i ) {
				scores[i].update( Score(doc.id_, ranker->score(queries[i], doc, no_qtf) ) );
			}

			c++;
			c%10==0 ? cerr << "reporter:status:scanned="<< c << "\n" : cerr ;

		}

		delete ranker;

		// create kbest lists
		for ( size_t i = 0; i < scores.size(); ++i ) {
			results[i] = scores[i].k_largest();
		}
	}

	// retrieval done; output results
	WriteFile out(cfg["output"].as<string>());
	if (cfg.count("qrels")) {
		cerr << "CLIR::Evaluating results ...";
		CLIR::IRScorer irs(queries_size, cfg["qrels"].as<string>());
		# pragma omp parallel for
		for ( int i = 0 ; i < queries_size ; ++ i ) {
			irs.evaluateIthSegment(i, queries[i].id(), results[i]);
		}
		*out << "num_q\t" << irs.N() << "\tnum_rel_ret\t" << irs.NUMRELRET() << "\tMAP\t" << irs.MAP() << "\tNDCG\t" << irs.NDCG() << endl;
	} else {
		cerr << "CLIR::Writing results ...";
		for ( int i = 0 ; i < queries_size ; ++ i ) {
			CLIR::writeResult(*out, queries[i], results[i], run_id);
		}
	}

	cerr << "ok.\nCLIR::done.\n";

}





