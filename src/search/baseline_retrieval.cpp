#include "baseline_retrieval.h"

int main(int argc, char** argv) {

	// handle parameters
	po::variables_map cfg;
	if (!init_params(argc,argv, &cfg)) exit(1); // something is wrong

	// load queries
	vector<CLIR::Document> queries;
	loadQueries(cfg["queries"].as<string>(), queries);
	cerr << queries.size() << " queries loaded." << endl;

	// get run id
	string run_id = "RUN1";
	if (cfg.count("run-id"))
		run_id = cfg["run-id"].as<string>();

	// check for xue croft scoring
	Collection* C = NULL;
	double lambda = 0.9;
	double beta = 0.9;
	if (cfg.count("xue-croft-scoring")) {
		C = new Collection(cfg["collection"].as<string>());
		cerr << "Query term back off collection loaded (" << C->size() << " entries)." << endl;
		if (cfg.count("lambda"))
			lambda = cfg["lambda"].as<double>();
		if (cfg.count("beta"))
			beta = cfg["beta"].as<double>();
	}

	// initialize vector of Scores objects for each query
	vector<CLIR::Scores<string> > scores (queries.size(), CLIR::Scores<string>(cfg["K"].as<int>()));

	// for each doc
	string docid;
	int len;
	string raw;
	TermVector doc;
	int c = 0;

	cerr << "reporter:status:scanning documents..." << endl;

	while (cin >> docid) {
		cin.ignore(1,'\t');
		cin >> len;
		cin.ignore(1,'\t');
		getline(cin, raw);

		if (docid.size() == 0 || len <= 0 || raw.size() == 0)
			continue;

		CLIR::Document doc(vutils::read_vector(raw), docid, len);

		// for each query compute score between current document and query
		for ( unsigned int i = 0 ; i < queries.size() ; ++i ) {

			Score<string> score (doc.docid_, prob_t(0));

			if (cfg.count("xue-croft-scoring"))
				score.mS = xue_croft_score( queries[i].wvec_, doc.wvec_, doc.len_, *C, lambda, beta );
			else if (cfg.count("bm25"))
				score.mS = bm25_from_precalculated_weights( queries[i].wvec_, doc.wvec_);
			else
				score.mS = cosine( queries[i].wvec_, doc.wvec_ );

			if(cfg.count("brevity-penalty") && cfg.count("conversion-factor"))
				score.mS *= brevityPenalty(queries[i].len_, doc.len_, cfg["conversion-factor"].as<double>() );
			scores[i].update(score);

		}

		c++;
		//c%1000==0 ? cerr << "[" << c << "]" << endl : cerr ;
		c%1000==0 ? cerr << "reporter:status:scanned="<< c << endl : cerr ;

	}

	if (cfg.count("xue-croft-scoring"))
		delete C;

	cerr << endl;
	cerr << "reporter:status:outputting kbest lists..." << endl;

	// for each query
	for (int j = 0; j < scores.size(); ++j) {
		vector<Score<string> > x  = scores[j].k_largest();

        	// if all scores are 0 for current query, just output a single dummy line if wanted
        	if (cfg.count("show-empty-results") && x.empty()) {
			cout << queries[j].docid_ << "\tQ0\t-1\t1\t0.0\t" << run_id << endl;
            		continue;
		}
		
		vector<Score<string> >::reverse_iterator rit;
		int k = 1;
		// for each document in kbest
		for ( rit=x.rbegin() ; rit < x.rend(); ++rit ) {
			//if (cfg.count("topic-map"))
			//	cout << topic_map[ queries[j].docid_ ] << "\t"; // print topic no
			//else
			cout << queries[j].docid_ << "\t"; // print query id
			cout << "Q0" << "\t";
			cout << rit->mD << "\t"; // docno
			cout << k << "\t"; // rank
			//cout << rit->mS.as_float() << "\t"; // score
			cout << rit->mS << "\t"; // score
			cout << run_id << endl; // run id
			k++;
		}
	}

	cerr << "reporter:status:done." << endl;

}

bool init_params(int argc, char** argv, po::variables_map* cfg) {
	po::options_description cl("\nPerforms batch retrieval for a set of queries on a document collection coming from STDIN.\nCommand Line Options");
	cl.add_options()
			("queries,q", po::value<string>(), "* File containing the query vectors. Documents are read from STDIN")
			("K,k", po::value<int>(), "* Keep track of K-best documents per query. (Number of results per query)")
			//("topic-map,t", po::value<string>(), "File containing the mappings between query ids and topic number. Necessary for Trec output.")
			("run-id,r", po::value<string>(), "RUN id that is printed in the output. Defaults to RUN1.")
			("brevity-penalty,p", po::value<bool>()->zero_tokens(), "penalize short document candidates using brevity penalty")
			("conversion-factor,f", po::value<double>(), "estimated mean length difference factor from query to doc. (1.05158 for de->en, 0.95094 for en->de")
			("xue-croft-scoring,x", po::value<bool>()->zero_tokens(), "Uses scoring method as defined by Jehl,Hieber,Riezler, WMT2012, instead of cosine scoring. Use 'expanded' document vectors for this!")
			("collection,c", po::value<string>(), "only used when -x is set. Specifies the file containing relative frequencies for query terms.")
			("lambda,l", po::value<double>(), "only used when -x is set. Specifies the smoothing parameter between translation mixture model and query collection backoff. Default 0.9")
			("beta,b", po::value<double>(), "only used when -x is set. Specifies the smoothing parameter between translation model and self translation. Default 0.9")
			("show-empty-results",po::value<bool>()->zero_tokens(), "set this if you want to include dummy output for proper scoring (default false)")
			("bm25,m", po::value<bool>()->zero_tokens(), "Uses proper bm25 scoring if document vectors are previously calculated bm25 vectors.");



	po::store(parse_command_line(argc, argv, cl), *cfg);
	po::notify(*cfg);
	if(!cfg->count("queries")) {
		cerr << cl << endl;
		return false;
	}
	if(!cfg->count("K")) {
		cerr << cl << endl;
		return false;
	}
	
	if(cfg->count("xue-croft-scoring")) {
		if (!cfg->count("collection")) {
			cerr << cl << endl;
			return false;
		}
		if (cfg->count("bm25")) {
			cerr << cl << endl;
			return false;
		}
	}
	
	return true;
}

