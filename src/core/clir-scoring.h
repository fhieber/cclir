#ifndef CLIR_SCORING_H_
#define CLIR_SCORING_H_

#include <math.h>
#include <assert.h>
#include <sstream>
#include <vector>
#include <unordered_map>

#include "src/core/clir.h"

using namespace std;

namespace CLIR {

class IRScorer {
public:
	IRScorer(const string& qrels) : n_(0) {
		reset();
		CLIR::loadQrels(qrels, rels_);
	}
	/*
	 * constructor for the case when n_ is known in
	 * advance (number of queries to be evaluated)
	 */
	IRScorer(const int n, const string& qrels) {
		n_ = n;
		num_rel_ret_ = 0;
		aps_ = vector<double>(n_,.0);
		ndcgs_ = vector<double>(n_,.0);
		CLIR::loadQrels(qrels, rels_);
	}

	void reset() {
		n_ = 0;
		aps_.clear();
		ndcgs_.clear();
		num_rel_ret_ = 0;
	}

	void clear() {
		reset();
		rels_.clear();
	}

	/*
	 * computes metrics for this segment/query if number of queries is known in advance.
	 * Input is the vector of Score objects
	 * this is NOT sorted again by ranks as below so it might give slightly different
	 * results than trec_eval or the below evaluator function. But its faster
	 * because we do not need to build a string first.
	 */
	void evaluateIthSegment(const unsigned i, const int qid, const vector<Score>& result) {
		assert(i<n_);
		if (result.empty()) {
			cerr << "Warning: result empty!\n";
			return;
		}
		unordered_map<WordID,unordered_map<WordID,unsigned> >::const_iterator rel_it = rels_.find(qid);
		if (rel_it == rels_.end()) {
			cerr << " Warning: no relevance judgements found for query '" << TD::Convert(qid) << "'\n";
			return;
		}
		vector<unsigned> retrieved;
		const unordered_map<WordID,unsigned>& qrels = rel_it->second;
		unsigned num_rel_ret = 0;
		unordered_map<WordID,unsigned>::const_iterator qit;
		// iterate in reversed order
		for (vector<Score>::const_reverse_iterator rit=result.rbegin();rit!=result.rend();++rit) {
			qit = qrels.find(rit->mD); // check if doc is relevant
			if (qit==qrels.end()) {
				retrieved.push_back(0);
			} else {
				retrieved.push_back(qit->second);
				num_rel_ret++;
			}
		}
		// get gold standard from qrels
		vector<unsigned> gold;
		for (qit=qrels.begin();qit!=qrels.end();++qit) gold.push_back(qit->second);
		sort(gold.begin(), gold.end(), std::greater<unsigned>());
		assert(gold.size() == qrels.size());
		// compute metrics
		aps_[i] = compute_ap(gold.size(), retrieved);
		ndcgs_[i] = compute_ndcg(retrieved, gold);
		#if defined(_OPENMP)
		#pragma omp atomic
		#endif
		num_rel_ret_ += num_rel_ret;
	}

	/*
	 * computes metrics for this segment/query. Input is a trec-formatted result string
	 * as returned by the clir-server / CLIR::writeResult().
	 * Issue to replicate TREC_EVAL scores:
	 * trec_eval resorts the result list by scores (not by given ranks).
	 * If score output precision is sufficiently small, the result list
	 * may be resorted. This (somewhat strange) behaviour is replicated here.
	 */
	void evaluateSegment(const string& trec_result) {
		// 1	Q0	US-6188605-B1	4	7.02855	1
		istringstream iss(trec_result);
		unordered_map<WordID,unordered_map<WordID,unsigned> >::const_iterator rit;
		const unordered_map<WordID,unsigned>* qrels = 0;
		unordered_map<WordID,unsigned>::const_iterator qit;
		vector<unsigned> retrieved;
		string qstr, dstr, dummy, runid;
		WordID qid, did;
		unsigned rank;
		double score;
		vector<pair<unsigned,double> > scores;
		bool first_line = true;
		unsigned lines_read = 0;
		unsigned num_rel_ret = 0;
		while(iss >> qstr) {
			iss.ignore(1, '\t');
			iss >> dummy;
			iss.ignore(1, '\t');
			iss >> dstr;
			iss.ignore(1, '\t');
			iss >> rank;
			iss.ignore(1, '\t');
			iss >> score;
			iss.ignore(1, '\t');
			iss >> runid;
			qid = TD::Convert(qstr);
			did = TD::Convert(dstr);
			lines_read++;
			if (first_line) { // get pointer to qrels
				first_line = false;
				rit = rels_.find(qid);
				if (rit==rels_.end()) {
					cerr << " Warning: no relevance judgements found for query '" << qstr << "'\n";
					return;
				}
				qrels = &(rit->second);
			}
			// check if document is relevant
			qit = qrels->find(did);
			if (qit==qrels->end()) {
				retrieved.push_back(0);
			} else {
				retrieved.push_back(qit->second);
				num_rel_ret++;
			}
			scores.push_back(make_pair(lines_read-1,score));
		}
		// sort to simulate trec_eval behaviour
		sort(scores.begin(),scores.end(),paircomp);
		vector<unsigned> retrieved2(retrieved.size());
		for (int i=0;i<scores.size();++i) retrieved2[i] = retrieved[scores[i].first];
		retrieved.swap(retrieved2);
		assert(retrieved.size() == lines_read);
		// get gold standard from qrels
		vector<unsigned> gold;
		for (qit=qrels->begin();qit!=qrels->end();++qit) gold.push_back(qit->second);
		sort(gold.begin(), gold.end(), std::greater<unsigned>());
		assert(gold.size() == qrels->size());
		qrels = 0;
		// compute metrics
		double ap = compute_ap(gold.size(), retrieved);
		double ndcg = compute_ndcg(retrieved, gold);
		// update scorer (critical block)
#if defined(_OPENMP)
		#pragma omp critical
#endif
		{
		aps_.push_back(ap);
		ndcgs_.push_back(ndcg);
		num_rel_ret_ += num_rel_ret;
		n_++;
		}
	}

	double MAP() {
		return accumulate( aps_.begin(), aps_.end(), 0.0 ) / n_;
	}

	double NDCG() {
		return accumulate( ndcgs_.begin(), ndcgs_.end(), 0.0 ) / n_;
	}

	unsigned NUMRELRET() {
		return num_rel_ret_;
	}

	unsigned N() {
		return n_;
	}

  /*
   * Author: Laura Jehl
   * calculates average precision given
   * - a vector of relevance levels of ranked retrieval results (e.g. <0,1,1,0,0,...>)
   * - the number of relevant documents
   * (- a cutoff value n)
   * NOTE: all relevance levels are treated equally!
   * implementation follows the irbook (http://nlp.stanford.edu/IR-book) and Wikipedia (http://en.wikipedia.org/wiki/Information_retrieval#Mean_average_precision)
   */
  double compute_ap( const unsigned num_rels, const vector<unsigned>& retrieved, unsigned prec_at_i=0  ) const {
    // if no value is given for prec_at_i, set it to the number of retrieved docs
    if (prec_at_i == 0) { prec_at_i = retrieved.size(); }
    assert(prec_at_i <= retrieved.size() );
    unsigned counter = 0;
    double sum = 0.0;
    //sum over precision at i for each relevant document
    for ( unsigned i = 0; i < prec_at_i; i++ ){
      if ( retrieved.at(i) > 0 ){
        counter++;
        sum += (double) counter / (double)( i+1 );
      }
    }
    // normalize by total number of RELEVANT docs
    return sum / (double) num_rels;
  }

  /*
   * Author: Laura Jehl
   * calculates normalized discounted cumulative gain, given
   * - a vector of relevance levels of ranked retrieval results (e.g. <0,3,1,0,2,...>)
   * - a vector of relevance levels of the gold standard ranking (e.g. <3,3,2,2,2,1>)
   * NOTE: relevance levels matter!
   *  implementation follows the irbook (http://nlp.stanford.edu/IR-book)
   */
  double compute_ndcg( const vector<unsigned>& retrieved, const vector<unsigned>& gold ) const {
    // calculate normalizing factor Z, so that perfect ranking gets score of 1
    // 1/(sum_j=1_num-rel (2^Rel(j) - 1) / log_2 (j+1 ) )
    double Z = 0.0;
    for (unsigned i = 0; i< gold.size(); i++ ) {
      Z += ( (double) gold[i] ) / log2 (i+2); // +2 because i starts from 0
    }
    Z = 1/Z;
    //calculate dcg for retrieved docs
    //sum_j=1_num-retrieved (2^Rel(j) - 1) / log_2 (j+1 )
    double dcg = 0.0;
    for (unsigned i = 0; i< retrieved.size(); i++ ) {
      if ( retrieved[i] > 0 )
        dcg += ( retrieved[i] ) / log2 (i+2); // +2 because i starts from 0
    }
    return Z*dcg;
  }

private:
	unsigned n_; // counts number of queries seen
	unsigned num_rel_ret_; // number of retrieved relevant documents
	unordered_map<WordID,unordered_map<WordID,unsigned> > rels_; // qrel file
	vector<double> aps_; // average precisions seen
	vector<double> ndcgs_; // ndcgs seen

	static bool paircomp(const pair<unsigned,double>& p1, const pair<unsigned,double>& p2 ) { return p1.second > p2.second; } 

};
}
#endif /* CLIR_SCORING_H_ */
