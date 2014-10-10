#include "clir.h"
#include "distclir.h"
#include <omp.h>

using namespace CLIR;

bool init_params(int argc, char** argv, po::variables_map* cfg) {
	po::options_description cl("\nruns a clir server on the given port\nCommand Line Options");
	cl.add_options()
			("index,i", po::value<string>(), " * Inverted Index for Documents")
			("documents,c", po::value<string>(), "* File containing document vectors (tf for bm25, tfidf for tfidf)")
			("dftable,d", po::value<string>(), "* table containing the df values. only used for bm25")
			("jobs,j", po::value<int>(), "Number of threads. Default: number of cores")
			("port", po::value<string>()->default_value("5555"), "port this server listens to")
			("no-qtf", po::value<bool>()->zero_tokens(), "do not use query term frequency for bm25 scoring")
			("model,m", po::value<string>()->default_value("classicbm25"), "Scoring function (classicbm25, bm25, classicbm25tf, tfidf, stfidf). default is classicbm25. If metric is tfidf/stfidf, cosine is calculated.")
			("help,h",	po::value<bool>()->zero_tokens(),	"print this help message.");			
	po::store(parse_command_line(argc, argv, cl), *cfg);
	po::notify(*cfg);
	if ((!cfg->count("documents") && !cfg->count("index")) || (cfg->count("documents") && cfg->count("index"))) {
		cerr << cl << "\n\nError: Specify either documents (-c) OR an inverted index (-i)!\n";
		return false;
	}
	if(cfg->count("help") || (cfg->count("documents") && !cfg->count("dftable")) ) {
		cerr << cl << endl;
		return false;
	}

	return true;
}

void runServer(const short& J, const string& port, const vector<CLIR::Document>& documents, const Scorer* ranker, const bool no_qtf) {


	// prepare context and socket
	zmq::context_t context (1);
	zmq::socket_t socket (context, ZMQ_REP);
	socket.bind ( ("tcp://*:" + port).c_str() );
	cerr << "server::LOG::running at port " << port << "\n";

	int K;
	bool psq;
	string msg, query_raw, runid;
	TIMER::timestamp_t t0, t1;
	double time;

	omp_set_num_threads(J);

	while (true) {

		// wait for a query
		DISTCLIR::receive(&socket, msg);
		if (DISTCLIR::parse_msg(msg, runid, K, psq, query_raw)) {

			Query query(psq, query_raw);
			if (!query.empty()) {

				t0 = TIMER::get_timestamp();
				cerr << "server::REQ::query(size="<< query.size() << ",psq=" << psq << ",k=" << K << ")\n";
				// one heap for each thread with the size that the query requested
				vector<Scores> scores(J, Scores(K));
				#pragma omp parallel for
				for (int d=0;d<documents.size(); ++d) {
					const Document& doc = documents[d];
					scores[omp_get_thread_num()].update( Score(doc.id_, ranker->score(query, doc, no_qtf) ) );
				}
				// merge the heaps
				Scores& result = scores[0]; // master thread scores
				for (vector<Scores>::iterator sit=scores.begin()+1;sit!=scores.end();++sit)
					result.update(*sit);
				stringstream rss;
				CLIR::writeResult(rss, query, result.k_largest(), runid);
				DISTCLIR::send(&socket, rss.str());

				t1 = TIMER::get_timestamp();
				time = (t1-t0) / 1000000.0L;
				cerr << "server::LOG::time::" << time << "s\n";

			} else {
				DISTCLIR::invalid(&socket);
			}

		} else {
			DISTCLIR::invalid(&socket);
		}

	}
}

void runIndexServer(const short& /*J*/, const string& port, const Index::Index& idx, const IndexScorer* ranker, const bool no_qtf) {

	// prepare context and socket
	zmq::context_t context (1);
	zmq::socket_t socket (context, ZMQ_REP);
	socket.bind ( ("tcp://*:" + port).c_str() );
	cerr << "server::LOG::running at port " << port << "\n";

	int K;
	bool psq;
	string msg, query_raw, runid;
	TIMER::timestamp_t t0, t1;
	double time;

	while (true) {

		// wait for a query
		DISTCLIR::receive(&socket, msg);
		if (DISTCLIR::parse_msg(msg, runid, K, psq, query_raw)) {

			Query query(psq, query_raw);
			if (!query.empty()) {

				t0 = TIMER::get_timestamp();
				cerr << "server::REQ::query(size="<< query.size() << ",psq=" << psq << ",k=" << K << ")\n";

				vector<double> scores(idx.NumberOfDocuments(), .0);
				ranker->score(query, scores, no_qtf);
				Scores kbest(K);
				for (int d=0;d<idx.NumberOfDocuments();++d) {
					if (scores[d]>0) {
						kbest.update( Score(idx.GetDocID(d), prob_t(scores[d]) ) );
					}
				}
				stringstream rss;
				CLIR::writeResult(rss, query, kbest.k_largest(), runid);
				DISTCLIR::send(&socket, rss.str());

				t1 = TIMER::get_timestamp();
				time = (t1-t0) / 1000000.0L;
				cerr << "server::LOG::time::" << time << "s\n";

			} else {
				DISTCLIR::invalid(&socket);
			}

		} else {
			DISTCLIR::invalid(&socket);
		}

	}

}

int main(int argc, char** argv) {
	// handle parameters
	po::variables_map cfg;
	if (!init_params(argc,argv, &cfg)) exit(1); // something is wrong

	// use query term frequency
	if (cfg.count("no-qtf")) cerr << "server::LOG::not using query term frequencies\n";
	cerr << "server::LOG::ranking_function=" << cfg["model"].as<string>() << "\n";

	if (cfg.count("index")) {

		ReadFile idx_in(cfg["index"].as<string>());
		const Index::Index idx(*idx_in);
		cerr << "server::Inverted Index loaded: " << idx << endl;

		IndexScorer* ranker = CLIR::setupIndexScorer(cfg["model"].as<string>(), &idx);

		runIndexServer(1, cfg["port"].as<string>(), idx, ranker, cfg.count("no-qtf"));

		delete ranker;

	} else {

		// setup DfTable
		DfTable dft (cfg["dftable"].as<string>());
		cerr << "server::DF table loaded (" << dft.size() << " entries)." << endl;

		// load document collection
		vector<Document> documents;
		double avgDocLen = CLIR::loadDocuments(cfg["documents"].as<string>(), documents);
		size_t N = documents.size();
		if (N < dft.mMaxDf) {
			cerr << "server::LOG::maxDf=" << dft.mMaxDf << " > N=" << N << ", setting N to maxDf.\n";
			N = dft.mMaxDf;
		}
		Scorer* ranker = CLIR::setupScorer(cfg["model"].as<string>(), N, avgDocLen, &dft);
		
		runServer(cfg["jobs"].as<int>(), cfg["port"].as<string>(), documents, ranker, cfg.count("no-qtf"));
		
		delete ranker;
		
	}

}

