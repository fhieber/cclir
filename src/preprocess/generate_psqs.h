#ifndef GET_OPTIONTABLES_H_
#define GET_OPTIONTABLES_H_

#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>

// cdec imports
#include "decoder.h"
#include "ff_register.h"
#include "verbose.h"
#include "filelib.h"

#include "src/core/observer.h"
#include "src/core/document.h"
#include "src/core/query.h"
#include "src/core/stopwords.h"
#include "src/core/dftable.h"
#include "src/core/nbest-ttable.h"
#include "src/core/lexical-ttable.h"

using namespace std;
using namespace CLIR;
namespace po = boost::program_options;

bool init_params(int argc, char** argv, po::variables_map* cfg);


/*
 * creates and sets up a cdec instance with the given
 * cdec config. Returns a pointer to the cdec instance
 */
Decoder* SetupDecoder(const po::variables_map& cfg) {
	register_feature_functions();
	ReadFile ini_rf(cfg["decoder_config"].as<string>());
	stringstream buffer;
	buffer << ini_rf.stream()->rdbuf();
	if (cfg.count("weights"))
		buffer << "weights=" << cfg["weights"].as<string>() << endl;
	cerr << "cdec cfg: " << "'" << cfg["decoder_config"].as<string>() << "'" << endl;
	if (cfg.count("verbose"))
		SetSilent(false);
	else
		SetSilent(true);

	// remember that this line writes lm terms into the TD dict
	Decoder* d = new Decoder(&buffer);
	return d;
}

#endif /* GET_OPTIONTABLES_H_ */
