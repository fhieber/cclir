/*
 * dfTable.h
 *
 *  Created on: Aug 10, 2012
 *      Author: hieber
 */

#ifndef DFTABLE_H_
#define DFTABLE_H_


#include <fstream>
#include <iostream>

#include "tdict.h"
#include "prob.h"
#include "sparse_vector.h"
#include "stringlib.h"
#include "filelib.h"

class DfTable {

	typedef SparseVector<double> df_table;

	public:

		DfTable() : mMaxDf(0) {};
		DfTable(const std::string& fname);
		~DfTable() { mTable.clear(); mMaxDf = 0; };

		void add_weight(const WordID& k, const double& w);
		void add_weights(const DfTable& other);
		size_t size() { return mTable.size(); };
		void clear();
		bool writeToFile(const std::string& fname) const;
		void update(const SparseVector<prob_t>& v);

		double operator [](const WordID& i) const {
			return this->get(i);
		}

		double get(const WordID& i) const {
			if (mTable.find(i) != mTable.end())
				return mTable.find(i)->second;
			return 0.0;
		}

		bool hasKey(const WordID& i) const {
			if (mTable.find(i) == mTable.end())
				return false;
			return true;
		}

		df_table mTable;
		double mMaxDf;

};

#endif /* DFTABLE_H_ */
