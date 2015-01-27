/*
 * generate-instances.h
 *
 *  Created on: Apr 22, 2013
 */

#ifndef INSTANCE_TEST_H_
#define INSTANCE_TEST_H_

#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "decoder.h"
#include "ff_register.h"
#include "verbose.h"
#include "filelib.h"

#include "observer.h"

using namespace std;
namespace po = boost::program_options;


/*
 * loads and parses commandline parameters
 */
bool init_params(int argc, char** argv, po::variables_map* cfg) {
	stringstream ss;
	ss << "\nGenerates TrainingInstances for clirtrain.\n"
	   << "<chunksize> determines the number of derivations used for one NbestTTable.\n"
	   << "Only complete chunks will be used.\n"
	   << "If the <k> is not a multiple of <chunksize>, the last chunk won't be created.\n"
	   << "If the forest is not large enough for given <chunksize>, no chunk will be created";
	po::options_description cl(ss.str());
	cl.add_options()
			("input,i",				po::value<string>(),					"input file. Use '-' for STDIN.")
			("output,o",			po::value<string>(),					"output file for training instances.")
			("weights,w",			po::value<string>(),					"input weights file for decoder (e.g. from previous iteration)")
			("decoder_config,c",	po::value<string>(),					"decoder config for cdec. SHOULD NOT CONTAIN WEIGHTS SETTING!")
			("k,k",					po::value<int>()->default_value(10),	"number of derivations per TrainingInstance")
			("sample_from",			po::value<string>()->default_value("kbest"), "where to sample from. 'kbest' or 'forest'")
			("unique_k_best,r",		po::value<bool>()->zero_tokens(),		"kbest: use unique kbest list.")
			("chunksize",			po::value<int>()->default_value(1),		"chunk size for an nbestttable")
			("ignore_derivation_scores", po::value<bool>()->zero_tokens(),	"ignore derivaton scores for nbestttable estimations")
			("LOWER,L",				po::value<double>()->default_value(0.005,"0.005"),"lower bound on nbestttable weights")
			("CUMULATIVE,C",		po::value<double>()->default_value(0.95,"0.95"),"cumulative upper bound on nbestttable weights")
			("target-stopwords,s",	po::value<string>(),					"stopword file for nbestttable stopword filtering (target langauge)")
			("rels",				po::value<string>(),					"trec relevance file to fill instances with relevance judgements")
			("check",				po::value<bool>()->zero_tokens(),		"check if weighted permutation calculation is ok!")
			("verbose,v",			po::value<bool>()->zero_tokens(),		"verbose output to STDERR.")
			("help,h",				po::value<bool>()->zero_tokens(),		"print this help message.");

	po::store(parse_command_line(argc, argv, cl), *cfg);
	po::notify(*cfg);

	if (cfg->count("help")) {
		cerr << cl << endl;
		return false;
	}

	if (!cfg->count("input") || !cfg->count("output") || !cfg->count("weights") || !cfg->count("decoder_config")) {
		cerr << cl << endl;
		return false;
	}

	return true;
}

/*
 * creates and sets up a cdec instance with the given
 * cdec config. Returns a pointer to the cdec instance
 */
Decoder* SetupDecoder(const po::variables_map& cfg) {
	register_feature_functions();
	ReadFile ini_rf(cfg["decoder_config"].as<string>());
	stringstream buffer;
	buffer << ini_rf.stream()->rdbuf();
	buffer << "weights=" << cfg["weights"].as<string>() << endl;
	cerr << " setting up decoder ... ";
	cerr << "cdec.ini=" << "'" << cfg["decoder_config"].as<string>() << "' weights='" << cfg["weights"].as<string>() << " ";
	cerr << buffer.str() << endl;

	if (cfg.count("verbose"))
		SetSilent(false);
	else
		SetSilent(true);

	// remember that this line writes lm terms into the TD dict
	Decoder* d = new Decoder(&buffer);
	cerr << "ok.\n";
	return d;
}

#endif /* INSTANCE_TEST_H_ */
