/*
 * query.h
 *
 *  Created on: May 28, 2013
 */

#ifndef QUERY_H_
#define QUERY_H_

#include <vector>
#include <sstream>
#include "nbest-ttable.h"
#include "util.h"

using namespace std;

namespace CLIR {

/*
 * Query represents either a bag-of-words query (vector) or a
 * probabilistic structured query (psq).
 */
class Query {
public:
	Query(const bool ispsq) : isPSQ(ispsq), size_(0) {}
	Query(const int id, const TermVector& tf) : isPSQ(false), size_(1), id_(id), v_(tf) {} 
	Query(const bool ispsq, const string& s) : isPSQ(ispsq), size_(0) { load(s); }
	Query(const bool ispsq, const size_t size, const string& id) : isPSQ(ispsq), size_(size), id_(TD::Convert(id)) {
		nbest_ttables_ = vector<NbestTTable>(size);
	}
	// constructor for a single sentence query
	Query(const bool ispsq, NbestTTable& nbt) : isPSQ(ispsq), size_(1), id_(TD::Convert("0")) {
		nbest_ttables_.push_back(nbt);
	}

	const NbestTTable& nbt(const size_t i) const { assert(isPSQ); return nbest_ttables_[i]; }
	const TermVector& tf() const { return v_; }

	void set(const size_t i, const NbestTTable& t) {
		assert(isPSQ); 	
		assert(i < size_);
		nbest_ttables_[i] = t;
	}

	void set(const TermVector& tfvec) {
		v_ = tfvec;
	}

	double tf(const WordID id) const {
		TermVector::const_iterator it= v_.find(id);
		if (it != v_.end())
			return it->second.as_float();
		else
			return .0;
	}

	void add(const NbestTTable& t) {
		assert(isPSQ);
		nbest_ttables_.push_back(t);
		++size_;
	}

	/*
	 * construct query instance from input string (as written by save())
	 */
	void load(const string& s) {
		assert(s.size() > 0);
		nbest_ttables_.clear();
		stringstream ss(s);
		string raw, id;
		ss >> id;
		id_ = TD::Convert(id);
		ss.ignore(1, '\t');
		ss >> size_;
		ss.ignore(1, '\t');
		if (isPSQ) {			
			for (size_t i=0;i<size_;++i) {
				getline(ss, raw, '\t');
				assert(raw.size() > 0);
				nbest_ttables_.emplace_back( raw ); // construct NbestTTable in place
			}
		}
		getline(ss, raw);
		v_ = vutils::read_vector(raw);
	}

	void save(ostream& out, const string& entry_sep=" ") const {
		out << TD::Convert(id_) << "\t" << size_ << "\t";
		if (isPSQ) {
			for (size_t i = 0; i<size_; ++i) {
				nbest_ttables_[i].save(out, entry_sep);
				out << "\t";
			}
		}
		vutils::write_vector(v_, out);
	}

	string asString() const {
		stringstream out;
		out << TD::Convert(id_) << "\t" << size_ << "\t";
		if (isPSQ) {
			for (size_t i = 0; i<size_; ++i) {
				nbest_ttables_[i].save(out, " ");
				out << "\t";
			}
		}
		vutils::write_vector(v_, out);
		return out.str();
	}

	void interpolate(const TTable& ttable, const double lambda) {
		assert(isPSQ);	
		for (size_t i=0;i<size_;++i)
			nbest_ttables_[i] = NbestTTable(nbest_ttables_[i], ttable, lambda);
	}

	Query interpolate(const Query& other, const double lambda) {
		assert(isPSQ);	
		assert(size() == other.size());
		Query interp(isPSQ);
		interp.set_id(TD::Convert(id_));
		for (size_t i=0;i<size_;++i) {
			NbestTTable n = NbestTTable(nbest_ttables_[i], other.nbt(i), lambda);
			interp.add(NbestTTable(nbest_ttables_[i], other.nbt(i), lambda));
		}
		interp.v_ = v_;
		interp.v_ += other.v_;
		return interp;
	}

	void constrain(const prob_t& L = prob_t(0), const prob_t& C = prob_t(1), const bool& swf=false) {
		assert(isPSQ);
		for (size_t i=0;i<size_;++i)
			nbest_ttables_[i].constrain(L, C, swf);
	}

	size_t size() const { return size_; }

	bool empty() {
		if (isPSQ) {
			for (size_t i=0;i<size_;++i) {
				if (!nbest_ttables_[i].empty())
					return false;
			}
			return true;
		} else {
			return v_.empty();
		}
	}

	string idstr() const { return TD::Convert(id_); }

	WordID id() const { return id_; }

	void set_id(const string& id) { id_ = TD::Convert(id); }

	void set_id(const WordID& id) { id_ = id; }
	
	bool isPSQ;
	

private:
	size_t size_;
	WordID id_;
	vector<NbestTTable> nbest_ttables_;
	TermVector v_;
	
};

}



#endif /* QUERY_H_ */
