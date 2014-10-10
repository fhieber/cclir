#include "clir.h"
#include "clir-scoring.h"
#include <omp.h>

using namespace CLIR;

bool init_params(int argc, char** argv, po::variables_map* cfg) {
	po::options_description cl("\nPerforms bm25-based batch retrieval based on Ture's SIGIR'12 models for a set of queries on a document collection.\nCommand Line Options");
	cl.add_options()
			("queries,q", po::value<string>(), "* File containing Queries.")
			("psq,p", po::value<bool>()->zero_tokens(), "indicate if queries are Probabilistic Structured Queries (PSQs)")
			("text-input,t", po::value<bool>()->zero_tokens(), "indicate if queries are raw text and need to be converted to tf vectors (does not work with psq)")
			("index,i", po::value<string>(), " * Inverted Index for Documents")
			("documents,c", po::value<string>(), "* File containing document tf vectors.")
			("dftable,d", po::value<string>(), "table containing the df values if -c is specified")
			("K,k", po::value<int>()->default_value(1000), "* Keep track of K-best documents per query. (Number of results per query)")
			("run-id,r", po::value<string>()->default_value("1"), "run id shown in the output")
			("jobs,j", po::value<int>(), "Number of threads. Default: number of cores")
			("model,m", po::value<string>()->default_value("classicbm25"), "Scoring function (bm25, classicbm25, classicbm25tf, tfidf, stfidf). default is classicbm25. If metric is tfidf/stfidf, cosine is calculated.")
			("no-qtf", po::value<bool>()->zero_tokens(), "do not use query term frequency for bm25 scoring")
			("qrels", po::value<string>(), "if specified, will compute IR metrics directly on the output list. No trec_eval required.")
			("output,o",	po::value<string>()->default_value("-"), "* File for output. Default STDOUT");
	po::store(parse_command_line(argc, argv, cl), *cfg);
	po::notify(*cfg);
	if ((!cfg->count("documents") && !cfg->count("index")) || (cfg->count("documents") && cfg->count("index"))) {
		cerr << cl << "\n\nError: Specify either documents (-c) OR an inverted index (-i)!\n";
		return false;
	}
	if(!cfg->count("queries")) {
		cerr << cl << endl;
		return false;
	}
	if(cfg->count("documents") && !cfg->count("dftable")) {
		cerr << cl << endl;
		return false;
	}

	return true;
}

int main(int argc, char** argv) {

	// handle parameters
	po::variables_map cfg;
	if (!init_params(argc,argv, &cfg)) exit(1); // something is wrong

	// set number of jobs
	if (cfg.count("jobs")) omp_set_num_threads(cfg["jobs"].as<int>());

	const bool no_qtf = cfg.count("no-qtf");
	string run_id = cfg["run-id"].as<string>();
	const int K = cfg["K"].as<int>();

	// load queries
	vector<Query> queries;
	CLIR::loadQueries(cfg["queries"].as<string>(), queries, cfg.count("psq"), cfg.count("text-input"));
	size_t queries_size = queries.size();

	// setup result vector
	vector<vector<Score> > results(queries_size);

	if (cfg.count("index")) {

		Index::Index idx;
		Index::LoadIndex(cfg["index"].as<string>(), idx);
		IndexScorer* ranker = CLIR::setupIndexScorer(cfg["model"].as<string>(), &idx);
		cerr << "CLIR-PARALLEL::Retrieving...";
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
		}
		delete ranker;
		cerr << "ok.\n";

	} else {

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

		Scorer* ranker = CLIR::setupScorer(cfg["model"].as<string>(), N, avgDocLen, &dft);
		cerr << "CLIR-PARALLEL::Retrieving...";
		# pragma omp parallel for
		for ( int i = 0 ; i < queries_size ; ++ i ) {
			const Query& query = queries[i];
			Scores scores(K);
			for ( int d = 0 ; d < N ; ++d ) {
				Document& doc = documents[d];
				scores.update( Score(doc.id_, ranker->score(query, doc, no_qtf) ) );
			}
			results[i] = scores.k_largest();
		}
		delete ranker;
		cerr << "ok.\n";

	}

	WriteFile out(cfg["output"].as<string>());
	if (cfg.count("qrels")) {
		cerr << "CLIR-PARALLEL::Evaluating results.\n";
		CLIR::IRScorer irs(queries_size, cfg["qrels"].as<string>());
		# pragma omp parallel for
		for ( int i = 0 ; i < queries_size ; ++ i ) {
			irs.evaluateIthSegment(i, queries[i].id(), results[i]);
		}
		*out << "num_q\t" << irs.N() << "\tnum_rel_ret\t" << irs.NUMRELRET() << "\tMAP\t" << irs.MAP() << "\tNDCG\t" << irs.NDCG() << endl;
	} else {
		cerr << "CLIR-PARALLEL::Writing results.\n";
		for ( int i = 0 ; i < queries_size ; ++ i ) {
			CLIR::writeResult(*out, queries[i], results[i], run_id);
		}
	}

	cerr << "CLIR-PARALLEL::done.\n";

}





