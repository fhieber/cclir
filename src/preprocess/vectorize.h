#ifndef PROJECT_VECTORS_H_
#define PROJECT_VECTORS_H_

#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>

// cdec imports
#include "filelib.h"
// core imports
#include "src/core/dftable.h"
#include "src/core/document.h"
#include "src/core/stopwords.h"

using namespace std;
using namespace CLIR;
namespace po = boost::program_options;

bool init_params(int argc, char** argv, po::variables_map* cfg);

#endif
