/*
 * write-svmmap-data.h
 *
 *  Created on: Apr 29, 2013
 */

#ifndef WRITE_SVMMAP_DATA_H_
#define WRITE_SVMMAP_DATA_H_

#include <iostream>
#include <fstream>
#include "filelib.h"
#include "weights.h"

#include <boost/program_options.hpp>

#include "TrainingInstance.h"

using namespace std;
namespace po = boost::program_options;

/*
 * loads and parses commandline parameters
 */
bool init_params(int argc, char** argv, po::variables_map* cfg) {
	po::options_description cl("\nWrites SVMMAP training data format...");
	cl.add_options()
			("instances,i", po::value<string>(), "file containing TrainingInstances as queries.")
			("output,o", po::value<string>(), "output directory")
			("weights,w", po::value<string>(), "weight vector file (for feature mapping)")
			("verbose,v",			po::value<bool>()->zero_tokens(),		"verbose output to STDERR.");

	po::store(parse_command_line(argc, argv, cl), *cfg);
	po::notify(*cfg);

	if(!cfg->count("instances") || !cfg->count("output") || !cfg->count("weights")) {
		cerr << cl << endl;
		return false;
	}

	if(DirectoryExists(cfg->at("output").as<string>() )) {
		cerr << "label directory does exist!\n";
		return false;
	}

	return true;
}


#endif /* WRITE_SVMMAP_DATA_H_ */
