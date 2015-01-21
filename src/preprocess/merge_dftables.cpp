#include "merge_dftables.h"

int main(int argc, char** argv) {
	if (argc < 2) {
		cerr << "merge-dftables <table1> <table2> [<table3> ...]" << endl << "writes to STDOUT" << endl;
		return 1;
	}

	vector<DfTable> dftables;
	int i = 0;
	for (i=1; i<argc; i++)
		dftables.push_back( DfTable( string(argv[i]) ) );

	if (i < 1) {
		cerr << endl << "Please specify at least 2 DfTable to merge!" << endl;
		return 1;
	}

	DfTable* first = &dftables[0];
	for (int d = 1 ; d < dftables.size() ; ++d) {
		cerr << "- adding " << dftables[d].size() << " entries to " << first->size() << " entries..." << endl;
		first->add_weights(dftables[d]);
	}
	cerr << first->size() << " entries in merged DfTable." << endl;
	first->writeToFile("-");
	return 0;

}



