/*
 * nbest-ttable.h
 *
 *  Created on: Aug 7, 2012
 *      Author: hieber
 */

#ifndef NBESTTTABLE_H_
#define NBESTTTABLE_H_

#include "stopwords.h"
#include "ttable.h"
#include <sstream>
#include <ostream>
#include <istream>
#include "util.h"

using namespace std;

/*
 * NbestTTable inherits from TTable storing translation weights estimated by the cdec decoding step.
 */
class NbestTTable : public TTable {

public:
	NbestTTable() : is_normalized(false) {};
	NbestTTable(istream& in) : is_normalized(false) { load(in); }
	NbestTTable(string& in) : is_normalized(false) { stringstream ss(in); load(ss); }

	NbestTTable(const TTable& a, const TTable& b, double lambda) : is_normalized(false) {
		prob_t l = prob_t(lambda);
		prob_t l_inv = prob_t(1-lambda);
		if ( lambda != 0 ) {
			for (TranslationTable::const_iterator it = a.begin() ; it != a.end() ; ++it)
				this->addWeightedPairs(it->first, it->second, l);
		}
		if ( lambda != 1 ) {
			for (TranslationTable::const_iterator it = b.begin() ; it != b.end() ; ++it)
				this->addWeightedPairs(it->first, it->second, l_inv);
		}
		this->normalize();
	}

	NbestTTable(const TTable& t, const TermVector& stf) : is_normalized(false) {
		TranslationOptions::const_iterator e;
		for (TermVector::const_iterator it=stf.begin(); it!=stf.end(); ++it) {
			if (t.contains(it->first)) {
				for (e = t.options_begin(it->first); e!= t.options_end(it->first); ++e) {
					this->addPair(it->first, e->first, e->second);
				}
			}
		}
	}

	~NbestTTable() { this->clear(); };


	/*
	 * adds a new translation to the table
	 */
	void addPair(const WordID s, const WordID t, const prob_t prob) {
		assert(prob > prob_t(0));
		table_[s][t] += prob;
	}

	/*
	 * adds weights from another TTable instance to this one.
	 */
	void addPairs(const TTable& other) {
		TranslationTable::const_iterator f;
		TranslationOptions::const_iterator e;
		for ( f = other.begin() ; f != other.end() ; ++f ) {
			for (  e = f->second.begin() ; e != f->second.end() ; ++e ) {
				this->addPair(f->first, e->first, e->second);
			}
		}
	}

	/*
	 * multiplies every entry in the table with the given constant.
	 * used for interpolation...
	 */
	void weightWithConstant(const prob_t& c) {
		TranslationTable::const_iterator f;
		TranslationOptions::const_iterator e;
		for ( f = begin() ; f != end() ; ++f ) {
			for (  e = f->second.begin() ; e != f->second.end() ; ++e ) {
				table_[f->first][e->first] *= c;
			}
		}
	}

	/*
	 * adds weights weighted with constant c from given translation options for given source word s
	 */
	void addWeightedPairs(const WordID& s, const TranslationOptions& other, const prob_t& c) {
		TranslationOptions::const_iterator e;
		for (  e = other.begin() ; e != other.end() ; ++e )
			addPair(s, e->first, e->second * c);
	}

	void clear() {
		TTable::clear();
	}


	string as_string() const {
		stringstream ss;
		ss << "NbestTTable [normalized=" << is_normalized << ", size="<< table_.size() << "]\n";
		ss << TTable::as_string();
		return ss.str();
	}

	/*
	 * writes an option table to the given output stream
	 */
	void save(ostream& out, const string& entry_sep=" ") const {
		out << is_normalized << entry_sep << size() << entry_sep;
		TranslationTable::const_iterator f;
		TranslationOptions::const_iterator e;
		for ( f = begin() ; f != end() ; ++f ) {
			for (  e = f->second.begin() ; e != f->second.end() ; ++e ) {
				out << TD::Convert(f->first) << " " << TD::Convert(e->first) << " " << e->second.as_float() << entry_sep;
			}
		}
	}

	/*
	 * saves an option table to the given output stream imposing the specified constraints.
	 * L is a lower bound on the translation weight. (TURE'12: L=0.005)
	 * C is the upper cumulative probability bound (TURE'12: C=0.95)
	 */
	void save(ostream& out, const prob_t& L, const prob_t& C, const string& entry_sep=" ", const bool& sw_filtering=false) const {
		out << is_normalized << entry_sep << size() << entry_sep;
		TranslationTable::const_iterator f;
		TranslationOptions::const_iterator e;
		SortedTransOptions::const_iterator sorted_e;
		// for all source words
		for ( f = table_.begin() ; f != table_.end() ; ++f ) {
			SortedTransOptions sorted_opts;
			// for all translation options for current source word f
			for (e = f->second.begin(); e!= f->second.end() ; ++e) {
				if ( e->second < L || (sw_filtering && sw::isStopword(e->first)) )
					continue;
				sorted_opts.emplace_back( e->first, e->second ); // SortedTransOption in place
			}
			std::sort(sorted_opts.begin(), sorted_opts.end(), compare_sorted_opts);
			prob_t cur_C(0);
			for ( sorted_e = sorted_opts.begin(); sorted_e != sorted_opts.end(); ++ sorted_e) {
				out << TD::Convert(f->first) << " " << TD::Convert(sorted_e->first) << " " << sorted_e->second.as_float() << entry_sep;
				cur_C += sorted_e->second;
				if (cur_C >= C)
					break;
			}
		}
	}

	/*
	 * imposes the specified constraints on the NbestTTable and renormalizes.
	 * L is a lower bound on the translation weight. (TURE'12: L=0.005)
	 * C is the upper cumulative probability bound (TURE'12: C=0.95)
	 */
	void constrain(const prob_t& L = prob_t(0), const prob_t& C = prob_t(1), const bool& swf=false) {
		TranslationTable::iterator f;
		TranslationOptions::const_iterator e;
		SortedTransOptions::const_iterator sorted_e;
		for ( f = table_.begin() ; f != table_.end() ; ++f ) {
			SortedTransOptions sorted_opts;
			// for all translation options for current source word f
			for (e = f->second.begin(); e!= f->second.end() ; ++e) {
				if ( e->second < L || (swf && sw::isStopword(e->first)) )
					continue;
				sorted_opts.emplace_back( e->first, e->second ); // in place sorted trans option
			}
			std::sort(sorted_opts.begin(), sorted_opts.end(), compare_sorted_opts);
			prob_t cur_C(0);

			TranslationOptions constrained_opts;
			for ( sorted_e = sorted_opts.begin(); sorted_e != sorted_opts.end(); ++ sorted_e) {
				constrained_opts[sorted_e->first] = sorted_e->second;
				cur_C += sorted_e->second;
				if (cur_C >= C) break;
			}
			if (constrained_opts.empty())
				table_.erase(f->first);
			else
				f->second = constrained_opts;
		}
		normalize(); // renormalize
	}

	/*
	 * reads translation options from an input stream
	 */
	void load(istream& in) {
		this->clear();
		string s,t;
		int _;
		double p;
		if (in.good()) {
			in >> this->is_normalized;
			in >> _;
			while (in >> s) {
				in >> t;
				in >> p;
				this->addPair(TD::Convert(s), TD::Convert(t), prob_t(p));
			}
		}
	}

	/*
	 * normalization is performed as in a standard lexical translation table:
	 * Given f, the sum of all options e_i sum to 1.
	 */
	void normalize() {
		TranslationTable::iterator f;
		TranslationOptions::iterator e;
		for ( f = table_.begin() ; f != table_.end() ; ++f ) {
			prob_t z(0);
			for (  e = f->second.begin() ; e != f->second.end() ; ++e )
				z += e->second;
			for (  e = f->second.begin() ; e != f->second.end() ; ++e )
				table_[f->first][e->first] = e->second / z;
		}
		is_normalized = true;
	}

private:
	bool is_normalized;

	typedef std::pair<WordID, prob_t> SortedTransOption;
	typedef std::vector<SortedTransOption> SortedTransOptions;
	bool static compare_sorted_opts(const SortedTransOption& left, const SortedTransOption& right) { return left.second > right.second; }

};

#endif /* NBESTTTABLE_H_ */
