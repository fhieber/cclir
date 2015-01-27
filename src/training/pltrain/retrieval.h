/*
 * retrieval.h
 *
 *  Created on: Apr 24, 2013
 */

#ifndef RETRIEVAL_H_
#define RETRIEVAL_H_

#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <omp.h>

#include "filelib.h"
#include "timing_stats.h"

#include "instance.h"
#include "src/core/clir.h"
#include "src/core/crosslingual_bm25.h"


using namespace std;
namespace po = boost::program_options;


/*
 * loads and parses commandline parameters
 */
bool init_params(int argc, char** argv, po::variables_map* cfg) {
	po::options_description cl("\nRun Re");
	cl.add_options()
			("queries,q", po::value<string>(), "* File containing Instances as queries. For each query there should be an NBestTTable")
			("documents,c", po::value<string>(), "* File containing document tf vectors.")
			("output,o",	po::value<string>(), "* directory to write results to.")
			("dftable,d", po::value<string>(), "* table containing the df values")
			("K,k", po::value<int>(), "* Keep track of K-best documents per query. (Number of results per query)")
			("N,n", po::value<double>(), "Number of documents in collection (for idf calculation)")
			("avg_len,a", po::value<double>(), "* Average length of documents (required for BM25)")
			("run-id,r", po::value<string>()->default_value("1"), "run id shown in the output")
			("jobs,j", po::value<int>(), "Number of threads. Default: number of cores");

	po::store(parse_command_line(argc, argv, cl), *cfg);
	po::notify(*cfg);

	if(!cfg->count("queries") || !cfg->count("output") || !cfg->count("documents") || !cfg->count("K") || !cfg->count("dftable") || !cfg->count("avg_len")) {
		cerr << cl << endl;
		return false;
	}

	if(DirectoryExists(cfg->at("output").as<string>() )) {
		cerr << "output directory exists!\n";
		return false;
	}

	return true;
}

#endif /* RETRIEVAL_H_ */
