/*
 * sgd.cpp
 *
 *  Created on: May 24, 2013
 */



#include "sgd.h"

int main(int argc, char** argv) {
	// handle parameters
	po::variables_map cfg;
	if (!init_params(argc,argv, &cfg)) exit(1); // something is wrong

	// init weights
	DenseWeightVector dense_weights;
	WeightVector weights;
	if (cfg.count("weights")) Weights::InitFromFile(cfg["weights"].as<string>(), &dense_weights);
	Weights::InitSparseVector(dense_weights, &weights);
	cerr << "W ||| " << weights << "\n\n";

	// load instances
	vector<Instance> instances;
	loadInstances(cfg["input"].as<string>(), instances);

	// setup output filenames
	string weights_out = "-";
	if (cfg.count("output"))
		weights_out = cfg["output"].as<string>();

	SGD::Learner sgd_learner(
			instances,
			weights,
			cfg["iterations"].as<int>(),
			cfg["epsilon"].as<double>(),
			cfg["learning-rate"].as<double>(),
			cfg.count("verbose"),
			cfg["batch-size"].as<int>(),
			&std::cout
			);

	sgd_learner.learn();

	WeightVector new_weights = sgd_learner.getWeights();

	cerr << "W ||| " << new_weights << "\n\n";

	writeWeightsToFile(weights_out, new_weights);
	cerr << "weights written to '" << weights_out << "'.\n";

	exit(0);
	// write output weight vector
	/*DenseWeightVector new_dense_weights = adarank.GetWeightVector();
	WeightVector new_weights;
	Weights::InitSparseVector(new_dense_weights, &new_weights);

	cerr << "Final Weight Vector:\n";
	for (WeightVector::iterator i=new_weights.begin(); i!=new_weights.end(); ++i)
		cerr << i->first << " " << FD::Convert(i->first) << "=" << i->second << endl;
	Weights::WriteToFile(cfg["output"].as<string>(), new_dense_weights, true, NULL);*/

}




