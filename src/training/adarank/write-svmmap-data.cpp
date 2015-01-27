/*
 * write-svmmap-data.cpp
 *
 *  Created on: Apr 29, 2013
 */

#include "write-svmmap-data.h"

int main(int argc, char** argv) {

	// handle parameters
	po::variables_map cfg;
	if (!init_params(argc,argv, &cfg)) exit(1); // something is wrong

	// init weights
	DenseWeightVector dense_weights;
	WeightVector weights;
	if (cfg.count("weights")) Weights::InitFromFile(cfg["weights"].as<string>(), &dense_weights);
	Weights::InitSparseVector(dense_weights, &weights);
	cerr << "Current Weight Vector:\n";
	for (WeightVector::iterator i=weights.begin(); i!=weights.end(); ++i)
		cerr << i->first << " " << FD::Convert(i->first) << "=" << i->second << endl;
	/*cerr << "\nDense Weights:\n";
	for (int i = 0 ; i < dense_weights.size(); ++i)
		cerr << i << " " << dense_weights[i] << endl;*/
	cerr << "# of features: " << FD::NumFeats() << " (-1 dummy feature @ idx 0)\n\n";


	// load instances
	vector<TrainingInstance> instances;
	loadInstances(cfg["instances"].as<string>(), instances);

	for (int i=0;i<instances.size();++i) {
		cout << instances[i].SVMMapString() << endl;
	}

}


