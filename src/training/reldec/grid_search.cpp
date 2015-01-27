#include <vector>
#include <numeric>
#include <cfloat>
#include <sys/time.h>

#include <stdlib.h>
#include <stdio.h>

#ifdef _OPENMP
	#include <omp.h>
#else
	#define omp_get_thread_num() 0
#endif

#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

#include "hg.h"
#include "hg_io.h"
#include "viterbi.h"
#include "feature_vector.h" // typedefs FeatureVector,WeightVector, DenseWeightVector
#include "weights.h"
#include "filelib.h"

using namespace std;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

typedef unsigned long long timestamp_t;
int J = 0; // global number of features

static timestamp_t get_timestamp () {
	struct timeval now;
	gettimeofday (&now, NULL);
	return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

string Escape(const string& x) {
  string y = x;
  for (int i = 0; i < y.size(); ++i) {
    if (y[i] == '=') y[i]='_';
    if (y[i] == ';') y[i]='_';
    if (y[i] == ' ') y[i]='_';
  }
  return y;
}

bool init_params(int argc, char** argv, po::variables_map* cfg) {
	stringstream ss;
	ss << "\ngrid search for optimal oracles";
	po::options_description cl(ss.str());
	cl.add_options()
		("input,i", po::value<string>(), "* input directory w/ hypergraphs")
		("output,o", po::value<string>(), "* output file for final weights")
		("features,f", po::value<vector<string> >()->multitoken(), "* set of features to run grid search on (max 4)")
		#ifdef _OPENMP
		("jobs,j", po::value<int>(), "Number of threads")
		#else
		#endif
		("help,h", po::value<bool>()->zero_tokens(), "prints this help message");
	po::store(parse_command_line(argc, argv, cl), *cfg);
	po::notify(*cfg);

	if(cfg->count("help")) {
		cerr << cl << endl;
		return false;
	}
	if(!cfg->count("input") || !cfg->count("output") || !cfg->count("features")) {
		cerr << cl << endl;
		return false;
	}
	return true;
}


void loadHypergraphs(const string& dir, vector<Hypergraph>& hgs) {
	timestamp_t t0 = get_timestamp();
	vector<fs::path> paths;
	fs::directory_iterator end_itr;
    for (fs::directory_iterator itr(dir); itr != end_itr; ++itr)
    {
        if (fs::is_regular_file(itr->path())) {
            paths.push_back(itr->path().string());
        }
    }
    std::sort(paths.begin(), paths.end());
	int edges=0,nodes=0;
	hgs.resize(paths.size());
	int x = paths.size() < 10 ? 1 : paths.size() / 10;
	cerr << "Loading " << paths.size() << " hypergraphs ";
	for (int i=0;i<paths.size();++i) {
		ReadFile rf(paths[i].string());
		string raw;
		rf.ReadAll(raw);
		Hypergraph hg;
		HypergraphIO::ReadFromPLF(raw, &hg);
		edges += hg.NumberOfEdges();
		nodes += hg.NumberOfNodes();
		hgs[i] = hg;
		if (paths.size()%x==0) {
			cerr << '.';
			cerr.flush();
		}
	}
	J = FD::NumFeats();
	timestamp_t t1 = get_timestamp();
	cerr << " ok " 
		 << J << " features, ~" 
		 << edges / (double) hgs.size() 
		 << " edges/hg, ~" 
		 << nodes / (double) hgs.size() 
		 << " nodes/hg (" << (t1 - t0) / 1000000.0L << "s)\n";
}

int main(int argc, char** argv) {
	po::variables_map cfg;
	if (!init_params(argc,argv,&cfg)) exit(1);
	
	#ifdef _OPENMP
	omp_set_num_threads(cfg["jobs"].as<int>());
	#else
	#endif

	// check features
	vector<string> features = cfg["features"].as<vector<string> >();
	assert(features.size() <=4 && features.size() > 0);

	// load hypergraphs
	vector<Hypergraph> hgs;
	loadHypergraphs(cfg["input"].as<string>(), hgs);
	

	vector<double> range = {0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
	vector<double>::const_iterator w0,w1,w2,w3;
	vector<WeightVector> ws; // all possible configurations
	for (w0=range.begin(); w0!=range.end(); ++w0) {
		if (features.size() > 1) {
			for (w1=range.begin(); w1!=range.end(); ++w1) {
				if (features.size() > 2) {
					for (w2=range.begin(); w2!=range.end(); ++w2) {
						if (features.size() > 3) {
							for (w3=range.begin(); w3!=range.end(); ++w3) {
								if (features.size() == 4 && fabs(*w0 + *w1 + *w2 + *w3 - 1.0) < 0.01) {
									WeightVector w;
									w[FD::Convert(features[0])] = *w0;
									w[FD::Convert(features[1])] = *w1;
									w[FD::Convert(features[2])] = *w2;
									w[FD::Convert(features[3])] = *w3;
									ws.push_back(w);
								}
							}
						}
						if (features.size() == 3 && fabs(*w0 + *w1 + *w2 - 1.0) < 0.01) {
							WeightVector w;
							w[FD::Convert(features[0])] = *w0;
							w[FD::Convert(features[1])] = *w1;
							w[FD::Convert(features[2])] = *w2;
							ws.push_back(w);
						}
					}
				}
				if (features.size() == 2 && fabs(*w0 + *w1 - 1.0) < 0.01) {
					WeightVector w;
					w[FD::Convert(features[0])] = *w0;
					w[FD::Convert(features[1])] = *w1;
					ws.push_back(w);
				}
			}
		}
		if (features.size() == 1) {
			WeightVector w;
			w[FD::Convert(features[0])] = *w0;
			ws.push_back(w);
		}
	}

	for (vector<WeightVector>::iterator it=ws.begin(); it!=ws.end(); ++it) {
		WeightVector& w = *it;
		cerr << w << endl;
		vector<vector<WordID> > translations(hgs.size());
		#pragma omp parallel for
		for ( int h=0; h<hgs.size(); ++h) {
			Hypergraph& hg = hgs[h];
			hg.Reweight(w);
			vector<WordID> trans;
			ViterbiESentence(hg, &trans);
			translations[h] = trans;
		}
		stringstream fname;
		fname << cfg["output"].as<string>() << "/" << w;
		cout << fname.str() << endl;
		WriteFile out(Escape(fname.str()));
		for ( int h=0; h<translations.size(); ++h) {
			*out << TD::GetString(translations[h]) << endl;
		}
	}
}

