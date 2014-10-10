/*
 * pairwise-cosine.cpp
 *
 *  Created on: Jan 10, 2013
 *      Author: hieber
 */

#include "pairwise-cosine.h"

int main() {

	string docid1;
	string docid2;
	string v1_raw;
	string v2_raw;
	int len1 = -1;
	int len2 = -1;

	TermVector v1;
	TermVector v2;

	while(cin >> docid1) {
		cin.ignore(1, '\t');
		cin >> len1;
		cin.ignore(1, '\t');
		getline(cin, v1_raw, '\t');

		if (docid1.size() == 0 || len1 <= 0)
			continue;

		cin >> docid2;
		cin.ignore(1, '\t');
		cin >> len2;
		cin.ignore(1, '\t');
		getline(cin, v2_raw);

		if (docid2.size() == 0 || len2 <= 0)
			continue;

		if (v1_raw.size() == 0)
			cerr << "WARNING: input vector 1 (id=" << docid1 << ",len=" << len1 << ") empty!" << endl;

		if (v2_raw.size() == 0)
			cerr << "WARNING: input vector 2 (id=" << docid2 << ",len=" << len2 << ") empty!" << endl;

		v1 = vutils::read_vector(v1_raw);
		v2 = vutils::read_vector(v2_raw);

		// output cosine
		cout << docid1 << "\t" << docid2 << "\t" << v1.dot(v2).as_float() << endl;


	}

}
