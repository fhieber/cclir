/*
 * lexical-ttable.h
 *
 *  Created on: Sep 11, 2012
 *      Author: hieber
 */

#ifndef LEXICALTTABLE_H_
#define LEXICALTTABLE_H_

#include <iostream>

#include "ttable.h"
#include "stringlib.h"
#include "filelib.h"

/*
 * LexicalTTable inherits from TTable and can be loaded from disk.
 * The constructor also allows specifying a lower threshold on the translation weight.
 * Disk format is, as usual: <target> <source> <probability>
 */
class LexicalTTable : public TTable {

	public:

		LexicalTTable() {};

		LexicalTTable(const std::string& fname, const double& L = 0.0) {
			this->clear();
			ReadFile in(fname);
			std::string t;
			std::string s;
			double p;
			while (*in) {
				getline(*in, t, ' ');
				t = Trim(t, " \t\n");
				if (t.size() == 0)
					continue;
				getline(*in, s, ' ');
				*in >> p;

				// L is a lower bound on probability
				if (t == "NULL" || s == "NULL" || p <= L)
					continue;
				table_[ TD::Convert(s) ] [ TD::Convert(t) ] = prob_t(p);
			}
		}

		~LexicalTTable() { this->clear(); };

};

#endif /* LEXICALTTABLE_H_ */
