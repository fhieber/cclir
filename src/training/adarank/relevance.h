/*
 * relevance.h
 *
 *  Created on: Apr 25, 2013
 */

#ifndef RELEVANCE_H_
#define RELEVANCE_H_

#include <iostream>
#include <sstream>
#include <assert.h>
#include <boost/unordered_map.hpp>
#include "src/core/serialization.h"
#include <boost/serialization/set.hpp>
#include <set>
#include "filelib.h"
#include "stringlib.h"

using namespace std;

typedef unsigned short RLevel;

class Relevance {
public:
	Relevance() {};

	void add(const pair<string,RLevel>& relevant_doc) {
		rel_docs_.insert(relevant_doc);
	}

	bool is_relevant(const string& docid, const RLevel& rl) const {
		return (rel_docs_.find(make_pair(docid,rl)) != rel_docs_.end());
	}

	string as_string() const {
		stringstream ss;
		ss << "[ ";
		set<pair<string,RLevel> >::const_iterator it;
		for (it=rel_docs_.begin(); it!= rel_docs_.end(); ++it)
			ss << it->first << ":" << it->second << " ";
		ss << "]";
		return ss.str();
	}

	string trec_relevance_string(const int& qid) const {
		stringstream ss;
		set<pair<string,RLevel> >::const_iterator it;
		for (it=rel_docs_.begin(); it!= rel_docs_.end(); ++it)
			ss << qid << "\t1\t" << it->first << "\t" << it->second << "\n";
		return ss.str();
	}

	string trec_relevance_string(const string& qid) const {
		stringstream ss;
		set<pair<string,RLevel> >::const_iterator it;
		for (it=rel_docs_.begin(); it!= rel_docs_.end(); ++it)
			ss << qid << "\t1\t" << it->first << "\t" << it->second << "\n";
		return ss.str();
	}

	size_t size() const { return rel_docs_.size(); }

private:
	set<pair<string,RLevel> > rel_docs_;

	// serialization
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int /*version*/) {
		ar & rel_docs_;
	}

};


static void loadTrecRels(const string& fname, boost::unordered_map<string,Relevance>& trec_rels) {
	ReadFile in(fname);
	string line;
	string prev_qid = "--";
	vector<string> cols;
	while (getline(*in, line)) {
		cols = SplitOnWhitespace(line); assert(cols.size() == 4);
		if (prev_qid != cols[0]) { // new qid
			Relevance r;
			r.add( make_pair( cols[2], atoi(cols[3].c_str() ) ) );
			prev_qid = cols[0];
			trec_rels[cols[0]] = r;
		} else {
			trec_rels[cols[0]].add( make_pair( cols[2], atoi(cols[3].c_str() ) ) );
		}
	}
	/*boost::unordered_map<string,Relevance>::const_iterator it;
	for (it=trec_rels.begin(); it!=trec_rels.end(); ++it) {
		cout << it->first << " " << it->second.as_string() << endl;
	}*/
	cerr << "loaded relevance judgements for " << trec_rels.size() << " queries\n";

}

#endif /* RELEVANCE_H_ */
