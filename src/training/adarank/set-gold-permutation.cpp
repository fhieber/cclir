/*
 * set-gold-permutation.cpp
 *
 *  Created on: Apr 27, 2013
 */


#include "set-gold-permutation.h"


int main(int argc, char** argv) {
	// handle parameters
	po::variables_map cfg;
	if (!init_params(argc,argv, &cfg)) exit(1); // something is wrong

	// load instances
	vector<TrainingInstance> instances;
	loadInstances(cfg["queries"].as<string>(), instances);

	// check input dir
	stringstream dirss;
	dirss << cfg["label-dir"].as<string>();
	if (!util::endsWith(cfg["label-dir"].as<string>(), "/"))
		dirss << "/";

	// load labels
	boost::unordered_map<string,vector<double> > labels;
	loadLabels(dirss.str(), labels);

	int found = 0;
	for (vector<TrainingInstance>::iterator it=instances.begin(); it!=instances.end(); ++it) {
		if (labels.find(it->GetID()) != labels.end()) {
			it->SetGoldPermutation(labels[it->GetID()]);
			++found;
		}
		if (cfg.count("verbose"))
			cerr << it->print() << endl;
		// TODO write SVM MAP training data
	}

	cerr << "found " << found << " labels for " << instances.size() << " instances.\n";

	// save labeled instances
	stringstream outss;
	outss << cfg["queries"].as<string>() << ".labels";
	saveInstances(outss.str(), instances);

}
