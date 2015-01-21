#ifndef WEIGHTED_TERM_VECTORS_H_
#define WEIGHTED_TERM_VECTORS_H_

#include <iostream>
#include <boost/program_options.hpp>

#include "src/core/clir.h"

using namespace std;
using namespace CLIR;
namespace po = boost::program_options;

bool init_params(int argc, char** argv, po::variables_map* cfg) {
	po::options_description cl("\nComputes various term weights for input term vectors (reads from STDIN)\nCommand Line Options");
	cl.add_options()
			("dftable,d", po::value<string>(), "* DF table to load")
			("weight_metric,w", po::value<string>()->default_value("classicbm25"), "Term weighting metric (classicbm25, bm25, classicbm25tf, tfidf, stfidf). default is classicbm25.")
			("verbose,v", po::value<bool>()->zero_tokens(), "verbose output to STDERR")
			("N,n", po::value<int>(), "Number of documents in the collection (for idf calculation). If not specified the maximum df value from the df table is used")
			("avg_len,a", po::value<double>(), "Average length of the documents (needed for BM25")
			("normalize", po::value<bool>()->zero_tokens(), "indicate if whether vectors should be normalized. (do not normalize for BM25)");

	po::store(parse_command_line(argc, argv, cl), *cfg);
	po::notify(*cfg);
	if(!cfg->count("dftable") && cfg->at("weight_metric").as<string>() != "classicbm25tf") {
		cerr << cl << endl;
		return false;
	}

	if(cfg->count("weight_metric") && cfg->at("weight_metric").as<string>() == "bm25") {
		if (!cfg->count("avg_len")) {
			cerr << cl << "\nMissing average document length for BM25 weighting!\n" << endl;
			return false;
		}
	}
	return true;
}

#endif





