/*
 * optimize.h
 *
 *  Created on: Apr 29, 2013
 */

#ifndef OPTIMIZE_H_
#define OPTIMIZE_H_

#include <iostream>
#include <fstream>
#include "filelib.h"
#include "stringlib.h"
#include "weights.h"

#include <boost/program_options.hpp>

#include "TrainingInstance.h"
#include "AdaRank.h"

using namespace std;
namespace po = boost::program_options;


/*
 * loads and parses commandline parameters
 */
bool init_params(int argc, char** argv, po::variables_map* cfg) {
	stringstream ss;
	ss << "\nruns Boosting";
	po::options_description cl(ss.str());
	cl.add_options()
			("input,i",				po::value<string>(),					"Labeled training instances as generated from set-gold-permutation")
			("weights,w",			po::value<string>(),					"current weight vector (same as for generate-instances)")
			("loss,l",  			po::value<string>(),					"loss function: 'map' or 'ndcg' or 'plackett'")
			("output,o",			po::value<string>(),					"output weight vector file")
			("iterations,t",		po::value<int>()->default_value(1000),	"number of iterations for boosting")
			("epsilon,e",			po::value<double>()->default_value(0.00001), "stopping criterion: delta change of loss < epsilon")
			("verbose,v",			po::value<bool>()->zero_tokens(),		"verbose output to STDERR.")
			("help,h",				po::value<bool>()->zero_tokens(),		"print this help message.");

	po::store(parse_command_line(argc, argv, cl), *cfg);
	po::notify(*cfg);

	if (cfg->count("help")) {
		cerr << cl << endl;
		return false;
	}

	if (!cfg->count("input") || !cfg->count("weights") || !cfg->count("loss") || !cfg->count("output")) {
		cerr << cl << endl;
		return false;
	}

	/*if(DirectoryExists(cfg->at("output").as<string>() )) {
		cerr << "output directory exists!\n";
		return false;
	}*/

	return true;
}

ListwiseLossFunction* set_loss(po::variables_map* cfg) {
	string metric = UppercaseString(cfg->at("loss").as<string>());
	if (metric == "MAP")
		return new MAP(true);
	else if (metric == "NDCG")
		return new NDCG(true);
	else if (metric == "PLACKETT")
		return new PlackettLuce(true);
	cerr << "unknown loss function!";
	abort();
}





#endif /* OPTIMIZE_H_ */
