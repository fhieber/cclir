/*
 * retrieval.cpp
 *
 *  Created on: Apr 24, 2013
 */

#include "retrieval.h"


int main(int argc, char** argv) {
	// handle parameters
	po::variables_map cfg;
	if (!init_params(argc,argv, &cfg)) exit(1); // something is wrong

	// set number of jobs
	if (cfg.count("jobs")) omp_set_num_threads(cfg["jobs"].as<int>());
	
	// setup DfTable
	DfTable dft (cfg["dftable"].as<string>());
	cerr << "DF table loaded (" << dft.size() << " entries)." << endl;

	// setup BM25 scorer
	BM25* scorer = new BM25();
	scorer->setAvgDocLength(cfg["avg_len"].as<double>());
	if (cfg.count("N"))
		scorer->setDocCount(cfg["N"].as<double>());
	else
		scorer->setDocCount(dft.mMaxDf);

	// load instances
	vector<TrainingInstance> instances;
	loadInstances(cfg["queries"].as<string>(), instances);

	const string run_id = cfg["run-id"].as<string>();

	int instances_size=instances.size(), query_count=0;
	for (int i=0; i<instances_size;++i) {
		instances[i].LoadNbestTTables(); // parse nbest_ttable_strings
		query_count += instances[i].size();
	}
	cerr << query_count << " queries to process...\n";

	// loading document collection
	vector<Document> documents;
	loadDocuments(cfg["documents"].as<string>(), documents);
	cerr << "loaded " << documents.size() << " documents into memory\n";
	int documents_size = documents.size();

	// setup output directory
	MkDirP(cfg["output"].as<string>());
	stringstream outss;
	outss << cfg["output"].as<string>() << "/";
	const string out_path = outss.str();

	/*
	 * run complete retrieval for each TrainingInstance
	 */
	# pragma omp parallel for
	for ( int i = 0 ; i < instances_size ; ++ i ) {
		
		const TrainingInstance& instance = instances[i];
		const int instance_size = instance.size();
		vector<Scores<string> > scores (instance_size, Scores<string>(cfg["K"].as<int>()));
		vector<double> instance_bm25_mate_scores(instance_size, 0.0);

		bool is_mate = false;

		// go over document collection
		for ( int d = 0 ; d < documents_size ; ++d ) {
				
			Document& doc = documents[d];
			if (instance.isRelevant(doc.docid_,2))
				is_mate = true; // indicates if cur document is rlevel 2 mate

			// for each query within instance calculate scores between query and current document
			for (unsigned short q = 0; q < instance_size ; ++q) {

				Score<string> score (doc.docid_, prob_t(0));
				int overlap = 0;
				score.mS = crosslingual_bm25(instance.GetNbestTTable(q), doc.wvec_, doc.len_, scorer, dft, overlap);
				scores[q].update(score);
				if (is_mate)
					instance_bm25_mate_scores[q] = score.mS.as_float();

			}

			is_mate = false;

		} // end of document loop
		
		// set instance retrieval permutation according to bm25 mate scores
		//instance.SetGoldPermutation(instance_bm25_mate_scores);

		/*
		 * write out retrieval results and correct relevances to separate files
		 */
		stringstream res_ss, rel_ss;
		res_ss << out_path << instance.GetID() << ".ranks";
		rel_ss << out_path << instance.GetID() << ".rels" ;
		WriteFile rank_out(res_ss.str());
		WriteFile rel_out (rel_ss.str());

		// write out scores for each query within current instance
		for (int q = 0 ; q < instance_size ; ++ q ) {

			vector<Score<string> > x = scores[q].k_largest();

			// write ranked list
			if (x.empty())
				*rank_out << q << "\tQ0\t-1\t1\t0.0\t" << run_id << endl;
			else {
				vector<Score<string> >::reverse_iterator rit;
				int r = 1; // rank
				// for each document in ranked list
				for ( rit=x.rbegin(); rit < x.rend(); ++rit, ++r ) {
					*rank_out << q << "\t" // query id
							  << "Q0" << "\t"
							  << rit->mD << "\t" // docno
							  << r << "\t" // rank
							  << rit->mS.as_float() << "\t" // score
							  //<< rit->mS << "\t" // score
							  << run_id << endl; // run id
				}
			}
			// write relevance file
			*rel_out << instance.GetRelevance().trec_relevance_string(q);

		} // end of query write out

		//cerr << instance.print() << endl;

	} // end of parallelized loop over instances

	cerr << "done\n";

}



