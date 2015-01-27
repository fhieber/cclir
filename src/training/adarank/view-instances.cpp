/*
 * view-instances.cpp
 *
 *  Created on: Apr 29, 2013
 */

#include "view-instances.h"

int main(int argc, char** argv) {

	if (argc != 2) {
		cerr << "USAGE: view-instances <binary instance file>\n";
		abort();
	}

	// load instances
	vector<TrainingInstance> instances;
	loadInstances(string(argv[1]), instances);

	for (int i=0;i<instances.size();++i) {
		cout << instances[i].print() << endl;
	}

}


