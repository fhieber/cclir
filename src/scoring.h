/*
 * scoring.h
 *
 *  Created on: Aug 15, 2012
 *
 */

#ifndef SCORING_H_
#define SCORING_H_

#include <sstream>
#include <vector>
#include <queue>
#include <utility>
#include <cfloat>

#include "prob.h"
#include "util.h"

#include "index.h"
#include "query.h"
#include "document.h"
#include "nbest-ttable.h"
#include "dftable.h"

namespace CLIR {

/*
 * Okapi BM25 functions
 * BM25(q,d) = rsj(q) * tfd(q,d) [* tfq(q,q)]
 */
namespace BM25 {
	// BM25 parameters
	static const double k_1 = 1.2;
	static const double k_3 = 2;
	static const double b = 0.75;
	// compute the Robertson-Spark-Jones weight (idf term).
	// Negative rsj weights are smoothed to DBL_MIN
	inline double RSJ(const double df, const unsigned int N) {
		return (df > N/2) ? DBL_MIN : log( (N - df + .5) / (df + .5) );
	}
	// negative rsj weights are allowed
	inline double NEGRSJ(const double df, const unsigned int N) {
		return log( (N - df + .5) / (df + .5) );
	}
	// common variant of the bm25 term frequency weight: k_1+1 at the enum; additional tf in the denom.
	// as in ivory.pwsim.score
	inline double TFD(const double tf, const unsigned int l, const double avg) {
		return ( (k_1+1)*tf ) / (k_1*((1-b) + b*(l/avg)) + tf + tf );
	}
	// classic variant: asymptotic to 1
	inline double TFD_classic(const double tf, const unsigned int l, const double avg) {
		return ( tf ) / (k_1*((1-b) + b*(l/avg)) + tf );
	}
	// query term frequency weight
	inline double TFQ(const double qtf) {
		return (qtf*(k_3+1)) / (k_3+qtf);
	}
	// classic variant: asymptotic to 1
	inline double TFQ_classic(const double qtf) {
		return (qtf*(k_3+1)) / (k_3+qtf);
	}

	const double TF1 = TFD_classic(1.0,1,1);

}

// abstract class for different scoring models between query and an inverted index (index.h)
class IndexScorer {
public:
	IndexScorer(const Index::Index* i) : idx_(i) {};
	virtual ~IndexScorer() { idx_ = NULL; }
	virtual void score(const Query& /*Q*/,vector<double>& /*scores*/, const bool /*no_qtf*/) const = 0; // score full query against idx
	virtual inline void score_w(const WordID /*q*/, const double /*qw*/, vector<double>& /*scores*/, const TranslationOptions* /*opts*/ = NULL) const = 0; // score a single query term against idx
protected:
	const Index::Index* idx_; // pointer to index
};

// BM25 scorer on inverted index
class IndexBM25Scorer : public IndexScorer {
public:
	IndexBM25Scorer(const Index::Index* i) : IndexScorer(i) {}
	virtual void score(const Query& Q, vector<double>& scores, const bool no_qtf) const {
		const TermVector& tf = Q.tf();

		if (Q.isPSQ) { // psq alternatives

			for (size_t s=0;s<Q.size();++s) { // for each sentence
				const NbestTTable& nbt = Q.nbt(s);
				for (TranslationTable::const_iterator qit=nbt.begin();qit!=nbt.end();++qit) { // for each query term
					score_w(qit->first, no_qtf ? 0 : BM25::TFQ(tf.get(qit->first).as_float()), scores, &(qit->second));
				} // end of query term loop
			} // end of sentence loop

		} else { // 1-best query

			for (TermVector::const_iterator qit=tf.begin();qit!=tf.end();++qit) {
				score_w(qit->first, no_qtf ? 0 : BM25::TFQ(qit->second.as_float()), scores);
			}

		}
	}
	virtual void inline score_w(const WordID q, double qw, vector<double>& scores, const TranslationOptions* opts = NULL) const {
		qw = (qw==0) ? 1.0 : qw;
		Index::Index::const_iterator plp; // pointer to current postingsList

		if (opts) { // its a source language query term from a PSQ and we are given some translation options

			vector<double> wtfs(idx_->NumberOfDocuments(), .0); // weighted tf scores for all documents from the postingsLists
			double wdf = 0; // weighted df
			for(TranslationOptions::const_iterator oit=opts->begin();oit!=opts->end();++oit) { // for each alternative
				const WordID tw = oit->first; // target term
				const double p = oit->second.as_float(); // translation probability
				if (idx_->GetPostingsList(tw, plp)) { // if term in index
					const Index::PostingsList& pl = plp->second;
					wdf += pl.df() * p; // df summand
					for(Index::PostingsList::const_iterator pli=pl.begin();pli!=pl.end();++pli) {
						wtfs[pli->docno] += pli->tf * p; // tf summand for current doc
					}
				}
			} // end of alternatives loop

			// update document scores array with bm25 score of current source term
			const double rsj = BM25::RSJ(wdf, idx_->NumberOfDocuments());
			for(unsigned int i=0;i<wtfs.size();++i) {
				scores[i] += BM25::TFD(wtfs[i], idx_->DocumentLength(i), idx_->AverageDocumentLength()) * rsj * qw;
			}

		} else { // its a regular 1-best translated query

			if (idx_->GetPostingsList(q, plp)) { // if term in index
				const Index::PostingsList& pl = plp->second;
				const double rsj = BM25::RSJ(pl.df(), idx_->NumberOfDocuments());
				for(Index::PostingsList::const_iterator pli=pl.begin();pli!=pl.end();++pli) {
					scores[pli->docno] += BM25::TFD(pli->tf, idx_->DocumentLength(pli->docno), idx_->AverageDocumentLength()) *
										  rsj * qw;
				}
			}
		}
	}
};
// IndexBM25 that uses the classic way of term saturation function (asymptotic to 1)
// !!!! This is actually half a MAP point better than normal BM25, fuck the empirical effectiveness of an additional tf!
class IndexClassicBM25Scorer : public IndexBM25Scorer {
public:
	IndexClassicBM25Scorer(const Index::Index* i) : IndexBM25Scorer(i) {}
	void score(const Query& Q, vector<double>& scores, const bool no_qtf) const {
		const TermVector& tf = Q.tf();

		if (Q.isPSQ) { // psq alternatives

			for (size_t s=0;s<Q.size();++s) { // for each sentence
				const NbestTTable& nbt = Q.nbt(s);
				for (TranslationTable::const_iterator qit=nbt.begin();qit!=nbt.end();++qit) { // for each query term
					score_w(qit->first, no_qtf ? 0 : BM25::TFQ_classic(tf.get(qit->first).as_float()), scores, &(qit->second));
				} // end of query term loop
			} // end of sentence loop

		} else { // 1-best query

			for (TermVector::const_iterator qit=tf.begin();qit!=tf.end();++qit) {
				score_w(qit->first, no_qtf ? 0 : BM25::TFQ_classic(qit->second.as_float()), scores);
			}

		}
	}
	void inline score_w(const WordID q, double qw, vector<double>& scores, const TranslationOptions* opts = NULL) const {
		qw = (qw==0) ? 1.0 : qw;
		Index::Index::const_iterator plp; // pointer to current postingsList

		if (opts) { // its a source language query term from a PSQ and we are given some translation options

			vector<double> wtfs(idx_->NumberOfDocuments(), .0); // weighted tf scores for all documents from the postingsLists
			double wdf = 0; // weighted df
			for(TranslationOptions::const_iterator oit=opts->begin();oit!=opts->end();++oit) { // for each alternative
				const WordID tw = oit->first; // target term
				const double p = oit->second.as_float(); // translation probability
				if (idx_->GetPostingsList(tw, plp)) { // if term in index
					const Index::PostingsList& pl = plp->second;
					wdf += pl.df() * p; // df summand
					for(Index::PostingsList::const_iterator pli=pl.begin();pli!=pl.end();++pli) {
						wtfs[pli->docno] += pli->tf * p; // tf summand for current doc
					}
				}
			} // end of alternatives loop

			// update document scores array with bm25 score of current source term
			const double rsj = BM25::RSJ(wdf, idx_->NumberOfDocuments());
			for(unsigned int i=0;i<wtfs.size();++i) {
				scores[i] += BM25::TFD_classic(wtfs[i], idx_->DocumentLength(i), idx_->AverageDocumentLength()) * rsj * qw;
			}

		} else { // its a regular 1-best translated query

			if (idx_->GetPostingsList(q, plp)) { // if term in index
				const Index::PostingsList& pl = plp->second;
				const double rsj = BM25::RSJ(pl.df(), idx_->NumberOfDocuments());
				for(Index::PostingsList::const_iterator pli=pl.begin();pli!=pl.end();++pli) {
					scores[pli->docno] += BM25::TFD_classic(pli->tf, idx_->DocumentLength(pli->docno), idx_->AverageDocumentLength()) *
										  rsj * qw;
				}
			}
		}
	}
};

// TFIDF scorer on inverted index
class IndexTFIDFScorer : public IndexScorer {
public:

	IndexTFIDFScorer(const Index::Index* i) : IndexScorer(i) {}

	void inline score_w(const WordID q, double qw, vector<double>& scores, const TranslationOptions* /*opts*/ = NULL) const {
		Index::Index::const_iterator plp; // pointer to current postingsList
		if (idx_->GetPostingsList(q, plp)) { // if term in index
			const Index::PostingsList& pl = plp->second;
			const double idf = computeIDF(pl.df());
			for(Index::PostingsList::const_iterator pli=pl.begin();pli!=pl.end();++pli) {
				const Index::Posting& p = *pli;
				scores[p.docno] += p.tf * idf * qw;
			}
		}
	}

	void score(const Query& Q, vector<double>& scores, const bool /*no_qtf*/) const {
		const TermVector& tf = Q.tf();
		for (TermVector::const_iterator qit = tf.begin(); qit != tf.end(); ++qit) {
			score_w(qit->first, qit->second.as_float(), scores);
		}
		// length normalization
		for(int d=0;d<scores.size();++d) { scores[d] /= (double) idx_->DocumentLength(d); }
	}

	virtual inline double computeIDF(const double df) const {
		double idf = log((idx_->NumberOfDocuments() - df + .5) / (df + .5));
		return (idf < 0.0) ? DBL_MIN : idf;
	}

};

// STFIDF scorer on inverted index
class IndexSTFIDFScorer : public IndexTFIDFScorer {
public:
	IndexSTFIDFScorer(const Index::Index* i ) : IndexTFIDFScorer(i) {}
	inline double computeIDF(const double df) const { return log(idx_->NumberOfDocuments() / df); }
};


// abstract class for different scoring models between query and documents
class Scorer {
public:
	Scorer(const int N, const double AvgLen, const DfTable* dft): N_(N), AvgLen_(AvgLen) {
		dft_ = dft;
	}
	virtual ~Scorer() { dft_ = NULL; }
	virtual prob_t score(const Query& /*Q*/, const Document& /*D*/, const bool /*no_qtf*/) const = 0;
	virtual void score(Document& D, bool normalize) const = 0;
	
	unsigned int N_; // document count
	double AvgLen_; // average document length
	const DfTable* dft_; // document frequency table pointer
};

// BM25 scoring implementation as in ivory.pwsim.score
class BM25Scorer : public Scorer {
public:
	BM25Scorer(const int N, const double AvgLen, const DfTable* dft) : Scorer(N,AvgLen,dft) {}
	prob_t score(const Query& Q, const Document& D, const bool no_qtf) const {
		
		if (Q.isPSQ) { // bm25 for PSQs
		
			// (1) get bm25 values
			TermVector bm25s; // weighted bm25 values without query term frequency scaling
			for (size_t s=0; s<Q.size(); ++s) {
				const NbestTTable& nbt = Q.nbt(s);
				for (TranslationTable::const_iterator qit=nbt.begin();qit!=nbt.end();++qit) {
					const WordID sw = qit->first; // source word
					const TranslationOptions& opts = qit->second;
					double wtf = .0, wdf = .0;
					for (TranslationOptions::const_iterator oit=opts.begin();oit!=opts.end();++oit) {
						const WordID tw = oit->first; // target word
						const double p = oit->second.as_float(); // prob
						wdf += dft_->get(tw) * p; // df summand
						TermVector::const_iterator dit = D.v_.find(tw);
						if (dit != D.v_.end()) // if tw in d
							wtf += dit->second.as_float() * p; // tf summand
					}
					if (wtf > 0)
						bm25s[sw] += BM25::TFD(wtf, D.len_, AvgLen_) * BM25::RSJ(wdf, N_);
				}
			}
			// (2) factor in query term frequency
			double score = .0;
			for (TermVector::iterator qit=bm25s.begin(); qit!=bm25s.end(); ++qit) {
				const WordID sw = qit->first;
				const double bm25 = qit->second.as_float();
				if (no_qtf)
					score += bm25;
				else
					score += bm25 * BM25::TFQ(Q.tf(sw));
			}
			return prob_t(score);
			
		} else { // bm25 for translated queries
		
			double score = .0;
			const TermVector& tf = Q.tf();
			for (TermVector::const_iterator qit = tf.begin(); qit != tf.end(); ++qit) {
				TermVector::const_iterator dit = D.v_.find(qit->first);
				if (dit != D.v_.end()) { // if q is in d
					if (no_qtf)
						score += BM25::TFD(dit->second.as_float(), D.len_, AvgLen_) * BM25::RSJ(dft_->get(qit->first), N_);
					else
						score += BM25::TFD(dit->second.as_float(), D.len_, AvgLen_) * BM25::RSJ(dft_->get(qit->first), N_) * BM25::TFQ(qit->second.as_float());
				}
			}
			return prob_t(score);
		}
		
	}
	// scores for Document (query independent)
	void score(Document& D, bool /*normalize*/) const {
		for (TermVector::iterator it=D.v_.begin();it!=D.v_.end();++it) {
			if (dft_->hasKey(it->first)) {
				it->second = prob_t( BM25::TFD(it->second.as_float(), D.len_, AvgLen_) *
						             BM25::RSJ(dft_->get(it->first), N_) );
			} else {
				cerr << TD::Convert(D.id_) << ": no df for '" << TD::Convert(it->first) << "'.\n";
				it->second = prob_t(DBL_MIN);
			}
		}
	}
};

// BM25 scoring implementation as in ivory.pwsim.score
class ClassicBM25Scorer : public Scorer {
public:
	ClassicBM25Scorer(const int N, const double AvgLen, const DfTable* dft) : Scorer(N,AvgLen,dft) {}
	prob_t score(const Query& Q, const Document& D, const bool no_qtf) const {

		if (Q.isPSQ) { // bm25 for PSQs

			// (1) get bm25 values
			TermVector bm25s; // weighted bm25 values without query term frequency scaling
			for (size_t s=0; s<Q.size(); ++s) {
				const NbestTTable& nbt = Q.nbt(s);
				for (TranslationTable::const_iterator qit=nbt.begin();qit!=nbt.end();++qit) {
					const WordID sw = qit->first; // source word
					const TranslationOptions& opts = qit->second;
					double wtf = .0, wdf = .0;
					for (TranslationOptions::const_iterator oit=opts.begin();oit!=opts.end();++oit) {
						const WordID tw = oit->first; // target word
						const double p = oit->second.as_float(); // prob
						wdf += dft_->get(tw) * p; // df summand
						TermVector::const_iterator dit = D.v_.find(tw);
						if (dit != D.v_.end()) // if tw in d
							wtf += dit->second.as_float() * p; // tf summand
					}
					if (wtf > 0)
						bm25s[sw] += BM25::TFD_classic(wtf, D.len_, AvgLen_) * BM25::RSJ(wdf, N_);
				}
			}
			// (2) factor in query term frequency
			double score = .0;
			for (TermVector::iterator qit=bm25s.begin(); qit!=bm25s.end(); ++qit) {
				const WordID sw = qit->first;
				const double bm25 = qit->second.as_float();
				if (no_qtf)
					score += bm25;
				else
					score += bm25 * BM25::TFQ_classic(Q.tf(sw));
			}
			return prob_t(score);

		} else { // bm25 for translated queries

			double score = .0;
			const TermVector& tf = Q.tf();
			for (TermVector::const_iterator qit = tf.begin(); qit != tf.end(); ++qit) {
				TermVector::const_iterator dit = D.v_.find(qit->first);
				if (dit != D.v_.end()) { // if q is in d
					if (no_qtf)
						score += BM25::TFD_classic(dit->second.as_float(), D.len_, AvgLen_) * BM25::RSJ(dft_->get(qit->first), N_);
					else
						score += BM25::TFD_classic(dit->second.as_float(), D.len_, AvgLen_) * BM25::RSJ(dft_->get(qit->first), N_) * BM25::TFQ_classic(qit->second.as_float());
				}
			}
			return prob_t(score);
		}

	}
	// scores for Document (query independent)
	void score(Document& D, bool /*normalize*/) const {
		for (TermVector::iterator it=D.v_.begin();it!=D.v_.end();++it) {
			if (dft_->hasKey(it->first)) {
				it->second = prob_t( BM25::TFD_classic(it->second.as_float(), D.len_, AvgLen_) *
						             BM25::RSJ(dft_->get(it->first), N_) );
			} else {
				cerr << TD::Convert(D.id_) << ": no df for '" << TD::Convert(it->first) << "'.\n";
				it->second = prob_t(DBL_MIN);
			}
		}
	}
};

// BM25 scoring implementation that gives uses negative rsj weights
class NEGBM25Scorer : public BM25Scorer {
public:
	NEGBM25Scorer(const int N, const double AvgLen, const DfTable* dft) : BM25Scorer(N,AvgLen,dft) {}
	prob_t score(const Query& Q, const Document& D, const bool no_qtf) const {

		if (Q.isPSQ) { // bm25 for PSQs

			// (1) get bm25 values
			TermVector bm25s; // weighted bm25 values without query term frequency scaling
			for (size_t s=0; s<Q.size(); ++s) {
				const NbestTTable& nbt = Q.nbt(s);
				for (TranslationTable::const_iterator qit=nbt.begin();qit!=nbt.end();++qit) {
					const WordID sw = qit->first; // source word
					const TranslationOptions& opts = qit->second;
					double wtf = .0, wdf = .0;
					for (TranslationOptions::const_iterator oit=opts.begin();oit!=opts.end();++oit) {
						const WordID tw = oit->first; // target word
						const double p = oit->second.as_float(); // prob
						wdf += dft_->get(tw) * p; // df summand
						TermVector::const_iterator dit = D.v_.find(tw);
						if (dit != D.v_.end()) // if tw in d
							wtf += dit->second.as_float() * p; // tf summand
					}
					if (wtf > 0)
						bm25s[sw] += BM25::TFD(wtf, D.len_, AvgLen_) * BM25::NEGRSJ(wdf, N_);
				}
			}
			// (2) factor in query term frequency
			double score = .0;
			for (TermVector::iterator qit=bm25s.begin(); qit!=bm25s.end(); ++qit) {
				const WordID sw = qit->first;
				const double bm25 = qit->second.as_float();
				if (no_qtf)
					score += bm25;
				else
					score += bm25 * BM25::TFQ(Q.tf(sw));
			}
			return prob_t(score);

		} else { // bm25 for translated queries

			double score = .0;
			const TermVector& tf = Q.tf();
			for (TermVector::const_iterator qit = tf.begin(); qit != tf.end(); ++qit) {
				TermVector::const_iterator dit = D.v_.find(qit->first);
				if (dit != D.v_.end()) { // if q is in d
					if (no_qtf)
						score += BM25::TFD(dit->second.as_float(), D.len_, AvgLen_) * BM25::NEGRSJ(dft_->get(qit->first), N_);
					else
						score += BM25::TFD(dit->second.as_float(), D.len_, AvgLen_) * BM25::NEGRSJ(dft_->get(qit->first), N_) * BM25::TFQ(qit->second.as_float());
				}
			}
			return prob_t(score);
		}

	}
	// scores for Document (query independent)
	void score(Document& D, bool /*normalize*/) const {
		for (TermVector::iterator it=D.v_.begin();it!=D.v_.end();++it) {
			if (dft_->hasKey(it->first)) {
				it->second = prob_t( BM25::TFD(it->second.as_float(), D.len_, AvgLen_) *
						             BM25::NEGRSJ(dft_->get(it->first), N_) );
			} else {
				cerr << TD::Convert(D.id_) << ": no df for '" << TD::Convert(it->first) << "'.\n";
				it->second = prob_t(DBL_MIN);
			}
		}
	}
};

// simple TFIDF
class STFIDFScorer : public Scorer {
public:
	STFIDFScorer(const int N, const double AvgLen, const DfTable* dft) : Scorer(N,AvgLen,dft) {}
	prob_t score(const Query& Q, const Document& D, const bool /*no_qtf*/) const {
		if (Q.isPSQ) {
			assert("not implemented!");
			return prob_t(DBL_MIN);
		} else {
			return Q.tf().dot(D.v_);
		}
	}
	
	// scores for Document (query independent)
	void score(Document& D, bool normalize) const {
		for (TermVector::iterator it=D.v_.begin();it!=D.v_.end();++it) {
			if (dft_->hasKey(it->first))
				it->second = computeWeight(it->second, dft_->get(it->first), D.len_);
			else {
				cerr << TD::Convert(D.id_) << ": no df for '" << TD::Convert(it->first) << "'.\n";
				it->second = prob_t(DBL_MIN);
			}
		}
		if (normalize)
			vutils::normalize_vector(D.v_);
	}
	/*
	 * idf = log( N / df )
	 * tfidf = tf * idf
	 */
	prob_t computeWeight(const prob_t& tf, const double df, const int /*len*/) const {
		double idf = log(N_ / df);
		prob_t w = tf;
		w *= idf;
		return w;
	}

};

// TDIDF
class TFIDFScorer : public Scorer {
public:
	TFIDFScorer(const int N, const double AvgLen, const DfTable* dft) : Scorer(N,AvgLen,dft) {}
	prob_t score(const Query& Q, const Document& D, const bool /*no_qtf*/) const {
		if (Q.isPSQ) {
			assert("not implemented!");
			return prob_t(DBL_MIN);
		} else {
			return Q.tf().dot(D.v_);
		}
	}
	// scores for Document (query independent)
	void score(Document& D, bool normalize) const {
		for (TermVector::iterator it=D.v_.begin();it!=D.v_.end();++it) {
			if (dft_->hasKey(it->first))
				it->second = computeWeight(it->second, dft_->get(it->first), D.len_);
			else {
				cerr << TD::Convert(D.id_) << ": no df for '" << TD::Convert(it->first) << "'.\n";
				it->second = prob_t(DBL_MIN);
			}
		}
		if (normalize)
			vutils::normalize_vector(D.v_);
	}
	/*
	 * idf = log( (N - df + 0.5) / df + 0.5 )
	 * tfidf = tf * idf
	 */
	prob_t computeWeight(const prob_t tf, const double df, const int& /*len*/) const {
		double idf = log((N_ - df + 0.5) / (df + 0.5));
		if (idf < 0.0) idf = DBL_MIN;
		prob_t w = tf;
		w *= idf;
		return w;
	}

};

// scores tf vectors using the BM25 TF metric (document normalization, asymptotic to 1) only. NO general RSJ/IDF weight
class ClassicBM25TFScorer : public Scorer {
public:
	ClassicBM25TFScorer(const int N, const double AvgLen, const DfTable* dft) : Scorer(N,AvgLen,dft) {}
	prob_t score(const Query& Q, const Document& D, const bool /*no_qtf*/) const {
		if (Q.isPSQ) {
			assert("not implemented!");
			return prob_t(DBL_MIN);
		} else {
			return Q.tf().dot(D.v_);
		}
	}
	// scores for Document (query independent)
	void score(Document& D, bool normalize) const {
		for (TermVector::iterator it=D.v_.begin();it!=D.v_.end();++it)
			it->second = prob_t( BM25::TFD_classic(it->second.as_float(), D.len_, AvgLen_) );
		if (normalize)
			vutils::normalize_vector(D.v_);
	}
};

/*
 * represents a pair of docid and corresponding score
 */
struct Score {
	Score(const WordID docid) : mD(docid), mS(0) {}
	Score(const WordID docid, const prob_t& s) : mD(docid) { mS=s; }
	bool operator > (const Score& o) const { return (mS > o.mS); }
	bool operator < (const Score& o) const { return (mS < o.mS); }
	bool operator == (const Score& o) const { return (mS == o.mS); }
	bool operator >= (const Score& o) const { return (mS >= o.mS); }
	bool operator <= (const Score& o) const { return (mS <= o.mS); }
	bool is_0() const { return mS.is_0(); }
	WordID mD;
	prob_t mS;
};

struct compare_score {
	bool operator()(const Score& s1, const Score& s2) const {
		return s1.mS > s2.mS;
	}
};

inline std::ostream& operator<<(std::ostream& out, const Score& s) {
	out << TD::Convert(s.mD)<<":"<<s.mS.as_float();
	return out;
}

/*
 * represents the scores during query evaluation as a min heap.
 */
class Scores {

	typedef std::priority_queue<Score,vector<Score>,compare_score> MinHeap;
	//typedef boost::heap::fibonacci_heap<Score, boost::heap::compare<compare_score> > MinHeap;
public:
	Scores(const unsigned short& K) : mS(K) { };

	short size() { return mH.size(); }
	bool empty() { return mH.empty(); }
	short maxSize() { return mS; }
	void reset() { mH = MinHeap(); }
	bool atCapacity() { return (mH.size() == mS); }
	const Score& top() const { return mH.top(); }

	/*
	 * add new score s to heap if heap size < maxSize/K
	 * and s is larger than the smallest item (heap top).
	 * Otherwise do nothing.
	 */
	void update(const Score& s) {
		if (s.is_0()) // check for negative infinity score
			return;
		if (mH.size() < mS)
			mH.push(s);
		else if (mH.size() == mS) {
			if (s > mH.top()) {
				mH.pop() ; mH.push(s);
			}
		}
	}
	void update(const WordID id, const prob_t& s) { update(CLIR::Score(id,s)); }

	/*
	 * incorporate/merge other Scores into this one
	 */
	void update(Scores& other) {
		 while(!other.empty()) {
		 	update( other.mH.top() );
		 	other.mH.pop();
		 }
	}
	 
	/*
	 * return the elements of the heap in a vector.
	 * IMPORTANT: vector is sorted by ascending score!
	 */
	std::vector<Score> k_largest() {
		std::vector<Score> r;
		while (!mH.empty()) {
			r.push_back(mH.top()); mH.pop();
		}
		return r;
	}

	MinHeap mH;
	const unsigned short mS;
};

}

#endif
