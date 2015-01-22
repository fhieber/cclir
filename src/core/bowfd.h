#ifndef _BOWFD_H_
#define _BOWFD_H_

#include <vector>
#include <algorithm>    // std::reverse
#include <boost/shared_ptr.hpp>
#include <queue>

#ifdef _OPENMP
	#include <omp.h>
#else
	#include <time.h>
	#define omp_get_wtime() clock()
	#define omp_get_thread_num() 0
#endif

// cdec imports
#include "config.h"
#include "sentence_metadata.h"
#include "verbose.h"
#include "prob.h"
#include "hg.h"
#include "tdict.h"
#include "filelib.h"
#include "stringlib.h"
#include "fdict.h"
#include "viterbi.h"
#include "ff_register.h"
#include "decoder.h"
#include "inside_outside.h"
#include "weights.h"
#include "feature_vector.h" // typedefs FeatureVector,WeightVector, DenseWeightVector

#include "src/core/clir.h"
#include "src/core/bowfd_features.h"
#include "src/core/bowfd_inference.h"

struct QuerySearchSpace : public DecoderObserver {
	QuerySearchSpace() {}
	virtual void NotifyTranslationForest(const SentenceMetadata& /*smeta*/, Hypergraph* forest) {
		hg = *forest;
		hg_nodes_size = hg.nodes_.size();
		hg_edges_size = hg.edges_.size();
		int lowlen=-1;
		smt_score = Viterbi<PathLengthTraversal>(hg, &lowlen);
	}
	void clear() {
		ir_features.clear();
		ir_edges.clear();
		ir_features.clear();
		ir_active.clear();
		io.outside.clear();
		io.inside.clear();
		smt_score = prob_t::Zero();
		top_score = prob_t::One();
		hg_nodes_size = 0;
		hg_edges_size = 0;
	}
	void InsideOutside(const WeightVector& best_bm25) {
		top_score = io.compute(hg, IRViterbiWeightFunction(&best_bm25,&ir_features));
		cerr << "Seach Space Score Range: " << top_score / smt_score << " [ "<<smt_score<<" , "<<top_score<<" ]\n";
	}
	// sorts incoming edges at each node by their viterbi+bestIR probs.
	void ViterbiSortInEdges() {
		assert (io.inside.size() == hg_nodes_size);
		hg.SortInEdgesByNodeViterbi(io.inside);
		// TODO think of pruning away edges that exceed the beam. would destroy order for bounds again
	}
	Hypergraph hg;
	size_t hg_nodes_size;
	size_t hg_edges_size;
	vector<FeatureVector> ir_features;
	vector<bool> ir_edges;
	WeightVector ir_active;
	prob_t smt_score;
	InsideOutsides<prob_t> io; // inside outside scores of search space
	prob_t top_score;
};

prob_t IRViterbi(const QuerySearchSpace& qspace, CLIR::FDocument& doc, const bool score_dense, const double w_dense, const unsigned beam=100000) {
	IRWeightFunction weightfunc( &doc, &(qspace.ir_features), score_dense, w_dense );
	return ViterbiRanker( qspace.hg, weightfunc, beam );
}

prob_t IRViterbiFeatures(const QuerySearchSpace& qspace, CLIR::FDocument& doc, FeatureVector& features, const bool score_dense, const double w_dense, const int dense_fid, const bool extract_smt, const bool extract_sparse, const bool extract_dense) {
	IRWeightFunction weightfunc( &doc, &(qspace.ir_features), score_dense, w_dense );
	IRFeatureVectorTraversal traversal( &doc, &(qspace.ir_features), dense_fid, extract_smt, extract_sparse, extract_dense );
	return ViterbiRanker( qspace.hg, &features, traversal, weightfunc );
}

inline bool close_enough(double a,double b,double epsilon=1e-5) {
	using std::fabs;
	double diff=fabs(a-b);
	return diff<=epsilon*fabs(a) || diff<=epsilon*fabs(b);
}

/*
 * adds tail node scores to <score>. If any of the tail nodes has been marked
 * as non-optimal, returns false (meaning we can skip this edge now)
 */
inline bool ScoreTailNodes(prob_t& score, const HG::Edge& edge, const vector<prob_t>& vit_scores, const vector<bool>& non_optimal_nodes) {
	for (unsigned k = 0; k < edge.tail_nodes_.size(); ++k) {
		if ( non_optimal_nodes[edge.tail_nodes_[k]] ) return false;
		score *= vit_scores[edge.tail_nodes_[k]];
	}
	return true;
}
/*
 * <hg>: SMT weighted hypergraph
 * <w_doc>: document model containing doc-specific IR weights
 * <lb>: current lower bound (Viterbi or worst topk document)
 * <ir_edges>: true for edges that can possibly match
 * <ub>: vector of upper bounds for each node
 * Performs bound checks to skip computation of IR scores as often as possible.
 * Regular Viterbi otherwise.
 */
prob_t ConstrainedIRViterbi2(
		const QuerySearchSpace qspace,
		const CLIR::FDocument& doc,
		const bool score_dense,
		const double w_dense,
		const prob_t& lb,
		const vector<prob_t>& ub,
		unsigned& cbn, // sum nodes marked bad
		unsigned& cse, // sum edges skipped
		const unsigned beam=100000
		) {
  
  const Hypergraph& hg = qspace.hg;
  const vector<FeatureVector>& ir_features = qspace.ir_features;
  vector<prob_t> vit_scores(qspace.hg_nodes_size, prob_t());
  vector<bool> non_optimal_nodes(qspace.hg_nodes_size, false);

  for (int i = 0; i < qspace.hg_nodes_size; ++i) {

	  const Hypergraph::Node& cur_node = hg.nodes_[i];
	  prob_t* const cur_node_best_weight = &vit_scores[i];

	  const unsigned num_in_edges = min(static_cast<unsigned>(cur_node.in_edges_.size()), beam);
	  if (num_in_edges == 0) {
		  *cur_node_best_weight = prob_t::One();
		  continue;
	  } // IDEA: can save leaf nodes once to skip them in the future. Does not save computation though..

	  // find best edge score
	  HG::Edge const* edge_best = 0;
	  for (unsigned j = 0; j < num_in_edges; ++j) {
		  const unsigned edge_idx = cur_node.in_edges_[j];
		  const HG::Edge& edge = hg.edges_[edge_idx];
		  prob_t score = edge.edge_prob_; // SMT score
		  if (!ScoreTailNodes(score, edge, vit_scores, non_optimal_nodes)) {
			  cse++;
			  continue; // can safely skip this edge due to non_optimal tail nodes
		  }
		  if (score_dense && w_dense!=0)
		  	score *= prob_t::exp( (ir_features[edge_idx].dot(doc.bm25_q())) * w_dense); // if ir_edge: IR score
		  else
		  	score *= prob_t::exp( ir_features[edge_idx].dot(doc.wbm25_q()) ); // if ir_edge: IR score
		  if (*cur_node_best_weight < score) {
			  *cur_node_best_weight = score;
			  edge_best=&edge;
		  }
	  }
	  // IF (skipped all in_edges) OR (vit_path*ub[i] fails to beat lower bound) THEN mark i non_optimal
	  if (!edge_best || *cur_node_best_weight * ub[i] <= lb) {
		  non_optimal_nodes[i] = true;
		  cbn++;
	  }
  }
  return vit_scores.back();
}

class Ranker {
public:
	Ranker(Decoder* d, CLIR::IRFeatureIndicator* i, const WeightVector* w_smt, const WeightVector* w_ir, const bool score_dense, const bool quiet=true) : decoder_(d), indicator_(i), w_smt_(w_smt), w_ir_(w_ir), score_dense_(score_dense), quiet_(quiet) {
		assert(d);
		assert(indicator_);
		assert(w_smt_);
		assert(w_ir_);
	}
	void translate(const string& id, const string& query) {
		assert(decoder_);
		qid_ = id;
		double start = omp_get_wtime();
		//TIMER::timestamp_t t0 = TIMER::get_timestamp();
		qspace_.clear();
		decoder_->Decode(query, &qspace_); // decode with decoder_weights
		double stop = omp_get_wtime();
		double time_taken = stop - start;
		//TIMER::timestamp_t t1 = TIMER::get_timestamp();
		if (!quiet_) cerr << "SPEED::Translation: " << time_taken << "s\n";
	}
	virtual void applyIndicator() {
		assert(indicator_);
		if (!quiet_) cerr << "HG("<<qspace_.hg_nodes_size<<" nodes, "<<qspace_.hg_edges_size<<" edges)\n";
		double start = omp_get_wtime();
		//TIMER::timestamp_t t0 = TIMER::get_timestamp();
		indicator_->apply(qspace_.hg, qspace_.ir_features, qspace_.ir_edges, qspace_.ir_active, score_dense_);
		double stop = omp_get_wtime();
		double time_taken = stop - start;
		//TIMER::timestamp_t t1 = TIMER::get_timestamp();
		//double time = (t1-t0) / 1000000.0L;
		if (!quiet_) cerr << "SPEED::IR features: " << time_taken << "s\n";
	}

protected:
	Decoder* decoder_;
	QuerySearchSpace qspace_;
	CLIR::IRFeatureIndicator* indicator_;
	const WeightVector* w_smt_;
	const WeightVector* w_ir_;
	const bool score_dense_; // if true, scoring is based on dense bm25 feature
	const bool quiet_;
	string qid_;
};

class TrainingRanker : public Ranker {
public:
	TrainingRanker(Decoder* d, CLIR::IRFeatureIndicator* i, const WeightVector* w_smt, const WeightVector* w_ir, const bool score_dense, const bool quiet=true) : Ranker(d, i, w_smt, w_ir, score_dense, quiet) {}

	prob_t score(CLIR::FDocument& doc, FeatureVector& f) {
		f.clear();
		doc.compute_bm25_q( qspace_.ir_active, true ); // we need the unweighted bm25 vector for feature extraction
		return IRViterbiFeatures(qspace_, doc, f, score_dense_, qspace_.ir_active.get(indicator_->bm25_fid),
								 indicator_->bm25_fid, extract_smt, extract_sparse, extract_dense);
	}
	WeightVector& ir_active() { return qspace_.ir_active; }
	void pushSMTWeightsToHypergraph() { qspace_.hg.Reweight(*w_smt_); }
	prob_t smt_score() { return qspace_.smt_score; }
	bool extract_smt;
	bool extract_sparse;
	bool extract_dense;
};

class TestRanker : public Ranker {
public:
	TestRanker(Decoder* d, CLIR::IRFeatureIndicator* i, const WeightVector* w_smt, const WeightVector* w_ir, const bool score_dense, const bool quiet=true) : Ranker(d, i, w_smt, w_ir, score_dense, quiet) {
		beam_size = 10000000;
	}

	void applyIndicator() {
		assert(indicator_);
		if (!quiet_) cerr << "HG("<<qspace_.hg_nodes_size<<" nodes, "<<qspace_.hg_edges_size<<" edges)\n";
		double start = omp_get_wtime();
		//TIMER::timestamp_t t0 = TIMER::get_timestamp();
		indicator_->apply(qspace_.hg, qspace_.ir_features, qspace_.ir_edges, qspace_.ir_active, score_dense_, (bound || sort_edges) ? &best_bm25 : NULL, N);
		double stop = omp_get_wtime();
		double time_taken = stop - start;
		//TIMER::timestamp_t t1 = TIMER::get_timestamp();
		//double time = (t1-t0) / 1000000.0L;
		if (!quiet_) cerr << "SPEED::IR features: " << time_taken << "s\n";
	}
	void maybe_bound_prune() {
		if (bound || sort_edges)
			qspace_.InsideOutside(best_bm25);
		if (sort_edges)
			qspace_.ViterbiSortInEdges();
	}
	/*
	 * Main algorithm: score the currently set documents with the current search space
	 * terminates early if possible
	 */
	vector<CLIR::Score> score() {
		assert(documents);
		// initialize J heaps and J lower_bounds for J threads
		vector<CLIR::Scores> scores(J, CLIR::Scores(K));
		vector<prob_t> lower_bounds(J, qspace_.smt_score);

		double start = omp_get_wtime();
		//TIMER::timestamp_t t0 = TIMER::get_timestamp();

		unsigned cbn=0;
		unsigned cse=0; // count of bad nodes, skipped edges
		unsigned cvr=0; // score attempts
		unsigned sum_active=0,min_active=INT_MAX,max_active=0; // # sum and min and max active features from w_doc_q

		const double bm25_weight = qspace_.ir_active.get(indicator_->bm25_fid);

		#pragma omp parallel for reduction(+:cbn,cse,cvr,sum_active)
		for ( int d = 0; d < N; ++d ) {

			const unsigned short j = omp_get_thread_num();
			CLIR::Scores& heap = scores[j];
			prob_t& lb = lower_bounds[j];
			CLIR::FDocument& doc = documents->at(d);

			if ( doc.compute_bm25_q( qspace_.ir_active, score_dense_ ) ) { // we need to score this

				if (bound) { // use bounding algorithm

					if (ideal_bounds) qspace_.InsideOutside(doc.wbm25_q()); // TODO currently broken

					heap.update( doc.id(), ConstrainedIRViterbi2( qspace_, doc, score_dense_, bm25_weight, lb, qspace_.io.outside, cbn, cse, beam_size ) );

					if ( heap.atCapacity() && heap.top().mS > lb ) lb = heap.top().mS;

				} else { // exhaustive scoring

						heap.update( doc.id(), IRViterbi(qspace_, doc, score_dense_, bm25_weight, beam_size) );

				}
				cvr++;
			}

			sum_active += doc.wbm25_q().size();
			if (doc.wbm25_q().size() < min_active) min_active = doc.wbm25_q().size();
			if (doc.wbm25_q().size() > max_active) max_active = doc.wbm25_q().size();

		}

		double stop = omp_get_wtime();
		//TIMER::timestamp_t t1 = TIMER::get_timestamp();
		double time_taken = stop - start;
		//double time = (t1-t0) / 1000000.0L;
		if (!quiet_) cerr << "SPEED::Document Scoring: " << time_taken << "s\n";
		if (!quiet_) {
			cerr << "Scoring Stats:\n"
			 << N << " documents\n"
			 << cvr << " score attempts (viterbi) (" << (cvr/(double)N)*100 << "%)\n"
			 << (cbn/(double)(cvr*qspace_.hg_nodes_size))*100 << "% of nodes marked bad on average\n"
			 << (cse/(double)(cvr*qspace_.hg_edges_size))*100 << "% of edges skipped on average\n"
			 << sum_active/(double)N << " ir features active per document. (min="<< min_active<<",max="<<max_active<<")\n";
		}

		// merge the heaps & write results
		CLIR::Scores& result = scores[0]; // master thread scores
		for (vector<CLIR::Scores>::iterator sit=scores.begin()+1;sit!=scores.end();++sit) result.update(*sit);
		return result.k_largest(); // THIS IS REVERSED!

	}

	//void prune() {
		// Pruning (this is not finalized yet) TODO
		// THIS IS BROKEN DUE TO HYERPGRAHS THAT DO NOT CONTAIN IR FEATURES!
		// NEED MY OWN IMPLEMENTATION!
		/*if (use_beam_prune || use_density_prune) {
			assert (0==1);
			double presize=hg.edges_.size();
			// preserve_edge mask: never prune matching edges! (not a good idea because we will keep all the low scoring ir edges too that do not discriminate well)
			hg.PruneInsideOutside(beam_prune,density_prune,NULL,false,1);
			cerr << "Pruning: "<<(hg.edges_.size()/presize)*100<<"% of edges kept.\n"
				 << "HG("<<hg.nodes_.size()<<" nodes, "<<hg.edges_.size()<<" edges)\n";
		}*/
	//}

	void rank(const string& id, const string& query, vector<CLIR::Score>& results) {
		if (!quiet_) cerr << "\nQ="<<id<<endl;
		translate(id, query);
		applyIndicator();
		maybe_bound_prune();
		// prune
		// TODO check how many documents actually contain matching terms (reduce set of candidates)
		//      already done by doc.compute_w_q(w_ir_q), but maybe use additional index to make it even faster ?
		// document scoring
		results = score();
	}

	void setDocuments(vector<CLIR::FDocument>* ds) { documents = ds; N = ds->size(); }

	vector<CLIR::FDocument>* documents;
	unsigned N;
	unsigned K;
	unsigned J;
	bool bound;
	bool ideal_bounds;
	double avg_len;
	WeightVector best_bm25;
	unsigned beam_size;
	bool sort_edges;

};


#endif