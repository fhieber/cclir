
#ifndef RUN_RETRIEVAL_H_
#define RUN_RETRIEVAL_H_


#include <iostream>
#include <assert.h>

#include <boost/program_options.hpp>
#include <boost/unordered_map.hpp>

#include "filelib.h"
#include "util.h"

#include "document.h"
#include "scoring.h"

using namespace std;
namespace po = boost::program_options;

bool init_params(int argc, char** argv, po::variables_map* cfg);
//void loadQueryIDPACtMap(const string& fname, boost::unordered_map<string, string>& map);


/*
 * loads queries from file. Format is
 * DOCID\tDOCLEN\tVECTOR\n
 */
void loadQueries(const string& fname, vector<Document>& queries) {
	ReadFile f(fname);
	string docid;
	int len;
	string raw;
	while(*f >> docid) {
		f->ignore(1,'\t');
		*f >> len;
		f->ignore(1,'\t');
		getline(*f, raw);
		if (docid.size() == 0 || len <= 0)
			continue;
		if (raw.size() == 0) {
			cerr << "WARNING: query vector (id=" << docid << ",len=" << len << ") empty!" << endl;
			cerr << "reporter:counter:RETRIEVAL,empty query vectors,1" << endl;
		}
		queries.push_back( Document(vutils::read_vector(raw), docid, len) );
	}
}

// SOME RETRIEVAL FEATURE/SCORING FUNCTIONS

/*
 * returns cosine of two normalized vectors.
 * this reduces to the dot product between these two vectors.
 */
prob_t const cosine(const TermVector& v1, const TermVector& v2) {
	return v1.dot(v2);
}

/*
 * returns bm25 without query term frequency scaling between query and document.
 * the document vector should be a BM25 vector.
 */
prob_t const bm25_from_precalculated_weights(const TermVector& q, const TermVector& d) {
	prob_t score = prob_t(0);
	for (TermVector::const_iterator it = q.begin(); it != q.end(); ++it) {
		if (d.find(it->first) != d.end()) // if q is in d
			score += d.get(it->first) * it->second;
	}
	return score;
}

prob_t const bm25(const TermVector& Q, const TermVector& D, const int Dlen, BM25* scorer, const DfTable& dft) {
	prob_t s = prob_t(0);
	TermVector::const_iterator dit, qit;
	for (qit = Q.begin(); qit != Q.end(); ++qit) {
		dit = D.find(qit->first);
		if (dit != D.end()) // if q is in d
			s += scorer->computeWeight(dit->second, dft.get(qit->first), Dlen) * scorer->computeQueryWeight(qit->second);
	}
	return s
}

/*
 * returns relevance score according to
 * http://www.cl.uni-heidelberg.de/~riezler/publications/papers/WMT2012.pdf
 * given that the document vector is already "expanded/projected" with
 * Document::calc_target_wvec().
 * Collection is basically a DfTable with relative frequencies across a large
 * collection in the query language.
 */
typedef DfTable Collection;
prob_t const xue_croft_score(const TermVector& q, const TermVector& d,
		const int& doc_len, const Collection& C, const double& lambda,
		const double& beta) {

	prob_t score = prob_t(1.0);
	for (TermVector::const_iterator it = q.begin() ; it != q.end() ; ++it ) {
		prob_t q_score = prob_t(0);

		// back off model P_q_C
		prob_t P_q_C = prob_t(DBL_MIN); // P_ML(q|C)
		if (C.mTable.get(it->first) > 0)
			P_q_C = prob_t(C.mTable.get(it->first));
		P_q_C *= (1-lambda);
		q_score = P_q_C;
		// if query term does not appear in expanded doc anyway, continue
		if (d.find(it->first) == d.end()) {
			score *= q_score;
			continue;	
		}

		if (d.get(it->first).as_float() >= 1.0) { // its self-translation (unweighted tf)
			prob_t P_self = d.get(it->first);
			P_self /= doc_len;
			P_self *= (1-beta);
			P_self *= lambda;
			q_score += P_self;
		} else { // its a weighted translation
			prob_t P_q_D = d.get(it->first);
			P_q_D *= beta;
			P_q_D *= lambda;
			q_score += P_q_D;
		}

		// update finale score with score for current q
		score *= q_score;

	}

	assert(score.as_float() < 1.0);
	return score;

}

double brevityPenalty(const int& q_len, const int& doc_len, const double& factor=1.0515822308) {
	double q = q_len * factor;
	if (q > doc_len) {
		return exp( 1 - q/doc_len );
	} else if (q < doc_len) {
		return exp( 1 - doc_len/q );
	} else {
		return 1.0;
	}

}

#endif /* RUN_RETRIEVAL_H_ */
