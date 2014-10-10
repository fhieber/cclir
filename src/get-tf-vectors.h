#ifndef PROJECT_VECTORS_H_
#define PROJECT_VECTORS_H_

#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>

#include "filelib.h"
#include "dftable.h"
#include "document.h"
#include "stopwords.h"

using namespace std;
using namespace CLIR;
namespace po = boost::program_options;

bool init_params(int argc, char** argv, po::variables_map* cfg);

#endif
