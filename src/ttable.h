/*
 * ttable.h
 *
 *  Created on: Nov 17, 2012
 *      Author: hieber
 */

#ifndef TTABLE_H_
#define TTABLE_H_

#include "tdict.h"
#include "prob.h"
#include <sstream>
#include <boost/unordered_map.hpp>
// serialization includes
#include "serialization.h"

typedef std::pair<WordID,WordID> translation; // helper type
typedef boost::unordered_map<WordID, prob_t> TranslationOptions; // inner map
typedef boost::unordered_map<WordID, TranslationOptions > TranslationTable; // outer map

/*
 * TTable models a classic translation table structure.
 * Each source term in the table contains another table with probability-weighted target options.
 */
class TTable {
public:

	virtual ~TTable() {};
	size_t size() const { return table_.size(); };

	/*
	 * returns the options for key f if it exists.
	 * otherwise return NULL
	 */
	TranslationOptions* getOptions(const WordID& s) {
		if (this->contains(s))
			return &table_.at(s);
		return NULL;
	}

	TranslationTable::const_iterator begin() const {
		return table_.begin();
	}

	TranslationTable::const_iterator end() const {
		return table_.end();
	}

	TranslationTable::iterator begin() {
		return table_.begin();
	}

	TranslationTable::iterator end() {
		return table_.end();
	}

	TranslationTable::const_iterator find(const WordID& s) const {
		return table_.find(s);
	}

	TranslationOptions::const_iterator options_begin(const WordID& s) const {
		return table_.at(s).begin();
	}

	TranslationOptions::const_iterator options_end(const WordID& s) const {
		return table_.at(s).end();
	}

	bool contains(const WordID& s) const {
		return !(find(s) == end());
	}

	bool contains(const WordID& s, const WordID& t) const {
		if (contains(s))
			return !(table_.at(s).find(t) == table_.at(s).end());
		return false;
	}

	/*
	 * adds a new translation to the table
	 */
	void addPair(const WordID s, const WordID t, const prob_t prob) {
		table_[s][t] += prob;
	}

	void clear() {
		TranslationTable::iterator it;
		for ( it = begin() ; it != end() ; ++it )
			it->second.clear();
		table_.clear();
	}

	bool empty() const {
		return table_.empty();
	}

	/*
	 * returns printable representation of the TTable
	 */
	std::string as_string() const {
		std::stringstream ss;
		TranslationTable::const_iterator s;
		TranslationOptions::const_iterator t;
		for ( s = begin() ; s != end() ; ++s ) {
			ss << TD::Convert(s->first) << " (" << s->first << ")" << std::endl;
			for (  t = s->second.begin() ; t != s->second.end() ; ++t ) {
				ss << " " << TD::Convert(t->first)
				   << " (" << t->first << ")"
				   << " "
				   << t->second.as_float()
				   << std::endl;
			}
		}

		return ss.str();
	}

	TranslationTable table_;

private:
	// serialization
	/*friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & table_;
	}*/

};

#endif /* TTABLE_H_ */
