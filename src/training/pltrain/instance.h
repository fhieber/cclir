/*
 * instance.h
 *
 *  Created on: Apr 22, 2013
 */

#ifndef INSTANCE_H_
#define INSTANCE_H_

#include <iostream>
#include <vector>
#include "vectors.h"
#include "derivation.h"

#include "src/core/query.h"
#include "src/core/nbest-ttable.h"

#include "src/core/relevance.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

using namespace std;

/*
 * an instance represents an ordered nbest list. We can only use single sentence queries!
 */
struct Instance {
	Instance() : size(0) , mt_sorted(true) , ir_sorted(false) , has_irscores(false) {};

	void SetID(string& qid) { id = qid; }

	void AddDerivation(const string& q_str, const SparseVector<double>& feature_vector, const double chunk_score, const WeightVector& /*weights*/) {
		Derivation::Derivation d;
		d.query_str = q_str;
		d.fvec = feature_vector;
		d.mtscore = chunk_score;
		derivations.push_back(d);
		++size;
	}

	// assumes scores has same order as derivations
	void SetRetrievalScores(const vector<double>& scores) {
		assert(scores.size() == size);
		assert(!has_irscores);
		for(unsigned short i=0;i<size;++i)
			derivations[i].irscore = scores[i];
		has_irscores = true;
		SortByRetrievalScores();
	}

	void SortByRetrievalScores() {
		if (ir_sorted) return;
		if (!has_irscores) abort();
		sort(derivations.begin(), derivations.end(), Derivation::ircomp);
		ir_sorted = true;
		mt_sorted = false;
	}

	void SortByMTScores() {
		if (mt_sorted) return;
		sort(derivations.begin(), derivations.end(), Derivation::mtcomp);
		mt_sorted = true;
		ir_sorted = false;
	}

	/*
	 * this is ugly but necessary due to non persistency of cdecs term dictionary.
	 */
	void GetQueries() {
		int i=0;
		for (vector<Derivation::Derivation>::iterator it=derivations.begin(); it!=derivations.end();++it) {
			stringstream ss;
			ss << i;
			it->loadQuery(ss.str());
			++i;
		}
	}

	string AsString() {
		stringstream ss;
		ss << "Instance '" << id <<"' (" << size << ") [";
		if (ir_sorted) ss << "ir sorted";
		else ss << "mt sorted";
		ss << "]\n";
		for (unsigned short i=0;i<size;++i)
			ss << i << " ||| " << derivations[i].AsString();
		ss << "\n";
		ss << relevance.as_string() << "\n";
		return ss.str();
	}
	
	string AsString(const WeightVector& w) {
		stringstream ss;
		ss << "Instance '" << id <<"' (" << size << ") [";
		if (ir_sorted) ss << "ir sorted";
		else ss << "mt sorted";
		ss << "]\n";
		for (unsigned short i=0;i<size;++i)
			ss << i << " ||| " << derivations[i].AsString(w);
		ss << "\n";
		ss << relevance.as_string() << "\n";
		return ss.str();
	}

	bool isRelevant(const string& docid, const RLevel& rl) const { return relevance.is_relevant(docid, rl); }
	void SetRelevance(Relevance& r) { relevance = r; }

	// serialization
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int /*version*/) {
		ar & id;
		ar & size;
		ar & derivations;
		ar & mt_sorted;
		ar & ir_sorted;
		ar & has_irscores;
		ar & relevance;
	}

	string id;
	size_t size;
	vector<Derivation::Derivation> derivations;
	bool mt_sorted;
	bool ir_sorted;
	bool has_irscores;
	Relevance relevance; // relevant documents with relevance levels
};


void static loadInstances(const string& input_fname, vector<Instance>& instances) {
	instances.clear();
	std::ifstream in(input_fname.c_str());
	boost::archive::binary_iarchive ia(in, std::ios::binary | std::ios::in);
	ia >> instances;
	in.close();
	cerr << instances.size() << " instances loaded from '" << input_fname << "'.\n";
}

void static saveInstances(const string& output_fname, vector<Instance>& instances) {
	std::ofstream out(output_fname.c_str(), std::ios::binary | std::ios::out);
	boost::archive::binary_oarchive oa(out);
	oa << instances;
	out.close();
	cerr << instances.size() << " serialized and written to '" << output_fname << "'.\n";
}

void static loadInstancesText(const string& input_fname, vector<Instance>& instances) {
	instances.clear();
	std::ifstream in(input_fname.c_str());
	boost::archive::text_iarchive ia(in);
	ia >> instances;
	in.close();
	cerr << instances.size() << " instances loaded from '" << input_fname << "'.\n";
}

void static saveInstancesText(const string& output_fname, vector<Instance>& instances) {
	std::ofstream out(output_fname.c_str());
	boost::archive::text_oarchive oa(out);
	oa << instances;
	out.close();
	cerr << instances.size() << " serialized and written to '" << output_fname << "'.\n";
}

#endif /* INSTANCE_H_ */
