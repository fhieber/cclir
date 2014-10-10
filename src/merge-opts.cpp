/*
 * interpolate-nbest-ttables.cpp
 *
 *  Created on: May 28, 2013
 */


#include "merge-opts.h"

int main(int argc, char** argv) {
	if (argc < 3) {
		cerr << "merge-opts <dftable> <option table1> <option table2> [<option table3> ...]" << endl << "writes to STDOUT" << endl;
		return 1;
	}

	string dft_fname = string(argv[1]);
	DfTable dft( dft_fname );
	cerr << "DfTable loaded (" << dft.size() << " entries)." << endl;

	NbestTTable global_opts;
	for (int i=2; i<argc; i++) {
		string fname = string(argv[i]);
		ReadFile of( fname );
		string raw;
		// we assume global option table files consist of only one line
		getline(*of, raw);
		stringstream ss_opt(raw);
		global_opts.addPairs(NbestTTable(ss_opt));
		//cerr << global_opts.as_string() << endl;
	}

	cerr << "Global OptionTable now consists of " << global_opts.size() << " entries. " << endl;

	global_opts.normalize();

	DfTable new_dft = cdec_df_project(dft, global_opts);

	cerr << "Projected DF table contains " << new_dft.size() << " entries." << endl;

	new_dft.writeToFile("-");

}

