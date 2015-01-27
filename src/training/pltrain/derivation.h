/*
 * derivation.h
 *
 *  Created on: May 22, 2013
 */

#ifndef DERIVATION_H_
#define DERIVATION_H_

#include "vectors.h"
#include "src/core/query.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

namespace Derivation {

struct Derivation {
	Derivation() : mtscore(0.0) , irscore(0.0) {};

	FeatureVector fvec;
	std::string query_str; // query string
	CLIR::Query query;
	double mtscore;
	double irscore;

	std::string AsString() const {
		std::stringstream ss;
		ss << "MT=" << mtscore << " ||| IR=" << irscore
		   << " ||| " << fvec
		   << " ||| Q: " << query_str << "\n";
		return ss.str();
	}
	
	std::string AsString(const WeightVector& w) const {
		std::stringstream ss;
		ss << "MT=" << mtscore << " ||| IR=" << irscore
		   << " ||| " << fvec.dot(w) << " ";
		for (FeatureVector::const_iterator i=fvec.begin();i!=fvec.end();++i)
			ss << FD::Convert(i->first) << "=" << i->second << " ";
		ss << "||| Q: " << query_str << "\n";
		return ss.str();
	}

	void loadQuery() {
		query = CLIR::Query(query_str);
	}

	void loadQuery(const string& id) {
		loadQuery();
		query.set_id(id);
	}

	// serialization
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int /*version*/) {
		ar & fvec;
		ar & query_str;
		ar & mtscore;
		ar & irscore;
	}

};

bool ircomp(const Derivation& a, const Derivation& b) {
	return a.irscore > b.irscore;
}

bool mtcomp(const Derivation& a, const Derivation& b) {
	return a.mtscore > b.mtscore;
}

}

#endif /* DERIVATION_H_ */
