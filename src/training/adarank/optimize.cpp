/*
 * optimize.cpp
 *
 *  Created on: Apr 29, 2013
 */



#include "optimize.h"

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
	loadInstances(cfg["input"].as<string>(), instances);

 	// setup output directory
	//MkDirP(cfg["output"].as<string>());
	//stringstream outss;
	//outss << cfg["output"].as<string>() << "/";
	//const string out_path = outss.str();

	// setup loss function
	ListwiseLossFunction* lossfunc = set_loss(&cfg);
	cerr << "listwise loss function: " << cfg["loss"].as<string>() << "\n";

	// run AdaRank optimizer
	AdaRank adarank(
			instances,
			instances.size(),
			cfg["iterations"].as<int>(),
			cfg["epsilon"].as<double>(),
			dense_weights,
			lossfunc,
			cfg.count("verbose")
			);

	adarank.run();

	// write output weight vector
	DenseWeightVector new_dense_weights = adarank.GetWeightVector();
	WeightVector new_weights;
	Weights::InitSparseVector(new_dense_weights, &new_weights);

	cerr << "Final Weight Vector:\n";
	for (WeightVector::iterator i=new_weights.begin(); i!=new_weights.end(); ++i)
		cerr << i->first << " " << FD::Convert(i->first) << "=" << i->second << endl;
	Weights::WriteToFile(cfg["output"].as<string>(), new_dense_weights, true, NULL);

}




