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
	vector<Instance> instances;
	loadInstances(cfg["queries"].as<string>(), instances);

	const string run_id = cfg["run-id"].as<string>();

	int instances_size=instances.size(), query_count=0;
	for (int i=0; i<instances_size;++i) {
		instances[i].GetQueries();
		query_count += instances[i].size;
	}
	cerr << query_count << " queries to process...\n";

	// loading document collection
	vector<Document> documents;
	CLIR::loadDocuments(cfg["documents"].as<string>(), documents);
	int documents_size = documents.size();

	// setup output directory
	MkDirP(cfg["output"].as<string>());
	stringstream outss;
	outss << cfg["output"].as<string>() << "/";
	const string out_path = outss.str();

	/*
	 * run retrieval for each TrainingInstance
	 */
	# pragma omp parallel for
	for ( int i = 0 ; i < instances_size ; ++ i ) {
		
		const Instance& instance = instances[i];
		const int instance_size = instance.size;
		vector<Scores<string> > scores (instance_size, Scores<string>(cfg["K"].as<int>()));

		// go over document collection
		for ( int d = 0 ; d < documents_size ; ++d ) {
			Document& doc = documents[d];
			// for each query within instance calculate scores between query and current document
			for (unsigned short q = 0; q < instance_size ; ++q) {
				Score<string> score (doc.docid_, prob_t(0));
				int overlap = 0;
				score.mS = crosslingual_bm25(instance.derivations[q].query, doc.wvec_, doc.len_, scorer, dft, overlap);
				scores[q].update(score);
			}

		} // end of document loop

		/*
		 * write out retrieval results and correct relevances to separate files
		 */
		stringstream res_ss, rel_ss;
		res_ss << out_path << instance.id << ".ranks";
		rel_ss << out_path << instance.id << ".rels" ;
		WriteFile rank_out(res_ss.str());
		WriteFile rel_out (rel_ss.str());

		// write out scores for each query within current instance
		for (int q = 0 ; q < instance_size ; ++ q ) {

			vector<Score<string> > topk = scores[q].k_largest();
			reverse(topk.begin(), topk.end());
			CLIR::writeResult(*rank_out, instance.derivations[q].query, topk, run_id);

			// write relevance file
			*rel_out << instance.relevance.trec_relevance_string(q);

		} // end of query write out

	} // end of parallelized loop over instances

	cerr << "done\n";

}



