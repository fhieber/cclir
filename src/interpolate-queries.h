/*
 * interpolate-queries.h
 *
 *  Created on: May 28, 2013
 */

#ifndef INTERPOLATE_QUERIES_H_
#define INTERPOLATE_QUERIES_H_

#include <iostream>

#include <boost/program_options.hpp>
#include "filelib.h"
#include "query.h"

using namespace std;
using namespace CLIR;
namespace po = boost::program_options;

/*
 * loads and parses commandline parameters
 */
bool init_params(int argc, char** argv, po::variables_map* cfg) {
	po::options_description cl("\nInterpolates two queries: lambda *#1 + (1-lambda) * #2. Writes to STDOUT\nOptions");
	cl.add_options()
			("1,1", po::value<string>(), "* Query #1")
			("2,2", po::value<string>(), "* Query #2")
			("psq,p", po::value<bool>()->zero_tokens(), "indicate if queries are Probabilistic Structured Queries (PSQs)")			
			("lambda,l", po::value<double>(), "* interpolation parameter lambda")
			("LOWER,L",	po::value<double>()->default_value(0.005,"0.005"), "lower bound on interpolated weights")
			("CUMULATIVE,C", po::value<double>()->default_value(0.95,"0.95"), "cumulative upper bound on interpolated weights")
			("help,h", po::value<bool>()->zero_tokens(), "print this help message.");

	po::store(parse_command_line(argc, argv, cl), *cfg);
	po::notify(*cfg);

	if (cfg->count("help")) {
		cerr << cl << endl;
		return false;
	}

	if (!cfg->count("1") || !cfg->count("2") || !cfg->count("lambda")) {
		cerr << cl << endl;
		return false;
	}

	return true;
}

#endif /* INTERPOLATE_QUERIES_H_ */
