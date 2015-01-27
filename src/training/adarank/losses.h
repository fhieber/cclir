/*
 * losses.h
 *
 *  Created on: Apr 23, 2013
 */

#ifndef LOSSES_H_
#define LOSSES_H_

#include <iostream>
#include <math.h>
#include <vector>
#include <boost/unordered_map.hpp>
#include <assert.h>

using namespace std;

typedef unsigned short ChunkId;
typedef std::vector<ChunkId> Permutation;

/*
 * generic class for a listwise lossfunction
 * (such as MAP, NDCG, Hamming Distance, or a PlackettLuce model)
 * All implementations are somewhat special since they take the Permutation type
 * as input. Permutation is just a vector of ids ranging from 0 to k, where
 * k is also the length of the vector (there is always a total ordering).
 */
class ListwiseLossFunction {
public:
	ListwiseLossFunction(const bool scale) : k_(1000), scale_(scale) {};
	virtual ~ListwiseLossFunction() {};
	// this needs to be implemented by inheriting classes
	virtual double operator()(const Permutation& candidate, const Permutation& truth) = 0;
	double score(const Permutation& candidate, const Permutation& truth) { return operator()(candidate, truth); };
	// scale result in range [0,1] to [-1,+1]
	double scale(const double& s) { return (s * 2.0) - 1; };
	const unsigned short k_; // top k evaluation
	const bool scale_; //scale to [-1,+1]
};

class NDCG : public ListwiseLossFunction {
public:
	NDCG(const bool scale) : ListwiseLossFunction(scale) {};
	double operator()(const Permutation& candidate, const Permutation& truth) {
		assert(candidate.size() == truth.size());
		// calculate normalizer and relevances
		double normalizer_ = 0;
		boost::unordered_map<ID,REL> rels_;
		unsigned short highestRank = truth.size();
		for (unsigned short i=0;i<k_ && i<highestRank;++i) {
			// chunkid truth[i] at position i has relevance highestRank - i
			rels_[truth[i]] = highestRank - i;
			normalizer_ += ( pow(2, double(highestRank-i)) - 1 ) / log(1 + (i+1));
		}
		// calculate DCG
		double dcg = 0.0;
		for(unsigned short i=0;i<k_ && i<candidate.size() ;++i)
			dcg += ( pow(2, double( rels_[candidate[i]] )) - 1 ) / log(1 + (i+1));

		return scale_ ? scale(dcg/normalizer_) : dcg/normalizer_;
	}

private:
	typedef unsigned short ID;
	typedef unsigned short REL;

};

/*
 * deprecated! doesn't make sense!
 */
class MAP : public ListwiseLossFunction {
public:
	MAP(const bool scale) : ListwiseLossFunction(scale) {};
	double operator()(const Permutation& candidate, const Permutation& truth) {
		assert(candidate.size() == truth.size());
		double avp = 0.0; // average precision
		double sum = 0.0; // sum of relevant
		for (unsigned short i=0;i<candidate.size();++i) {
			double P_i = 0.0; // Precision at i
			bool is_rel = false;
			for (unsigned short k=0;k<=i;++k) { // for all positions k until i
				P_i += (candidate[k] == truth[k]) ? 1.0 : 0;
				if (candidate[i] == truth[k]) is_rel = true;
			}
			P_i /= i+1;
			if (is_rel) {
				avp += P_i;
				sum += 1.0;
			}
		}
		return scale_ ? scale(avp/sum) : avp/sum;
	}
};

class HammingDistance : public ListwiseLossFunction {
public:
	HammingDistance(const bool scale) : ListwiseLossFunction(scale) {};
	double operator()(const Permutation& candidate, const Permutation& truth) {
		assert(candidate.size() == truth.size());
		double hd = 0.0;
		for (unsigned short i=0;i<candidate.size();++i)
			hd += (candidate[i] != truth[i]) ? 1 : 0 ;
		return hd;
	}

};

class PlackettLuce : public ListwiseLossFunction {
public:
	PlackettLuce(const bool scale) : ListwiseLossFunction(scale) {};
	double operator()(const Permutation& candidate, const Permutation& truth) {
		unsigned short truth_size = truth.size();
		assert(candidate.size() == truth_size);

		// get mapping from candidate chunkId to score / position
		boost::unordered_map<ChunkId,unsigned short> map;
		unsigned short s = truth_size;
		for (unsigned short i=0; i<truth_size; ++i) {
			map[candidate[i]] = s / truth_size;
			--s;
		}
		assert(s==0);

		// calculate logloss under plackett luce model: E v_i - E log(E exp(v_k))
		double v = 0.0;
		for (unsigned short i=0; i<truth_size; ++i) {
			v += map[truth[i]]; // first summand (top of fraction)
			
		}
		
		// get (candidate) score for each truth chunkID
		vector<double> scores(truth_size);
		double sum_scores = 0.0;
		for (unsigned short j=0; j<truth_size; ++j) {
			scores[j] = exp( map[truth[j]] ) ; // exp( candidate score)
			sum_scores += scores[j];
		}
		//cerr << "sum_scores=" << sum_scores << "\n";
		double P_truth_given_model = 1.0;
		double prev_score = 0.0; // score from previous chunk
		for (unsigned short j=0; j<truth_size; ++j) {
			P_truth_given_model *= exp(scores[j]) / (sum_scores - prev_score);
			prev_score += scores[j];
		}
		assert(prev_score == sum_scores);
		return scale_ ? scale(P_truth_given_model) : P_truth_given_model;
	}
private:
	// computes pairwise logsumexps by doing the famous trick
	double logsumexp(double x, double y, bool flg) {
		if (flg) {
			return y; // init mode
		} if (x == y)  {
			return x + 0.69314718055; // log(2)
		}
		double vmin = x > y ? y: x;
		double vmax = x > y ? x: y;
		return vmax + log(exp(vmin-vmax)+1.0);
	}
	
};

#endif /* LOSSES_H_ */
