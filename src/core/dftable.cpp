/*
 * dfTable.cpp
 *
 *  Created on: Aug 10, 2012
 *      Author: hieber
 */

#include "dftable.h"

/*
 * loads a dftable from disk.
 * The file on disk should have been written
 * with vutils::write_vector using \n as separator
 */
DfTable::DfTable(const std::string& fname) : mMaxDf(0){
	this->clear();
	std::ifstream in(fname.c_str(), std::ifstream::in);
	if (!in.is_open()) {
		std::cerr << "could not load dftable from '" << fname << "'!" << std::endl;
		return;
	}
	std::string w;
	double v;
	while (in.good()) {
		in >> w;
		if (w.size() == 0) continue;
		in >> v;
		this->add_weight(TD::Convert(w), v);
	}
}

void DfTable::add_weight(const WordID& k, const double& w) {
	mTable[k] += w;
	if (w > mMaxDf)
		mMaxDf = w;
}

/*
 * merges another df table into this instance
 */
void DfTable::add_weights(const DfTable& other) {
	for(df_table::const_iterator it = other.mTable.begin() ; it != other.mTable.end() ; ++it)
		this->add_weight(it->first, it->second);
}

void DfTable::clear() {
	mTable.clear();
	mMaxDf = 0;
}

bool DfTable::writeToFile(const std::string& fname) const {
	WriteFile df_out(fname);
	if (df_out) {
		for(df_table::const_iterator it = mTable.begin(); it != mTable.end(); ++it)
				*df_out << TD::Convert(it->first) << " " << it->second << "\n";
		return true;
	}
	return false;
}

/*
 * iterate over given vector keys and add 1 once for each type
 */
void DfTable::update(const SparseVector<prob_t>& v) {
	for(SparseVector<prob_t>::const_iterator it = v.begin(); it != v.end(); ++ it)
		this->add_weight(it->first, 1);
}
