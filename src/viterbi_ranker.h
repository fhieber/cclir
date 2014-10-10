#ifndef _VITERBI_RANKER_H_
#define _VITERBI_RANKER_H_

#include <vector>
#include <limits>
#include "prob.h"
#include "hg.h"
#include "tdict.h"
#include "filelib.h"
#include <boost/make_shared.hpp>

template<class Traversal,class WeightFunction>
typename WeightFunction::Weight ViterbiRanker(
				   const Hypergraph& hg,
                   typename Traversal::Result* result,
                   const Traversal& traverse,
                   const WeightFunction& weight) {

  typedef typename Traversal::Result T;
  typedef typename WeightFunction::Weight WeightType;

  const int num_nodes = hg.nodes_.size();
  std::vector<T> vit_result(num_nodes);
  std::vector<WeightType> vit_weight(num_nodes, WeightType());

  for (int i = 0; i < num_nodes; ++i) {

    const Hypergraph::Node& cur_node = hg.nodes_[i];
    WeightType* const cur_node_best_weight = &vit_weight[i];
    T*          const cur_node_best_result = &vit_result[i];

    const unsigned num_in_edges = cur_node.in_edges_.size();
    if (num_in_edges == 0) {
      *cur_node_best_weight = WeightType(1);
      continue;
    }
    HG::Edge const* edge_best=0;
    for (unsigned j = 0; j < num_in_edges; ++j) {
      const HG::Edge& edge = hg.edges_[cur_node.in_edges_[j]];
      WeightType score = weight(edge);
      for (unsigned k = 0; k < edge.tail_nodes_.size(); ++k)
        score *= vit_weight[edge.tail_nodes_[k]];
      if (!edge_best || *cur_node_best_weight < score) {
        *cur_node_best_weight = score;
        edge_best=&edge;
      }
    }
    assert(edge_best);
    HG::Edge const& edgeb=*edge_best;
    std::vector<const T*> antsb(edgeb.tail_nodes_.size());
    for (unsigned k = 0; k < edgeb.tail_nodes_.size(); ++k)
      antsb[k] = &vit_result[edgeb.tail_nodes_[k]];
    traverse(edgeb, antsb, cur_node_best_result);

  }

  if (vit_result.empty())
    return WeightType(0);
  std::swap(*result, vit_result.back());
  return vit_weight.back();
}


/*
 * computes only the score. no viterbi yield
 */
template<class WeightFunction>
typename WeightFunction::Weight ViterbiRanker(const Hypergraph& hg, const WeightFunction& weight, const unsigned beam=100000) {
  typedef typename WeightFunction::Weight WeightType;

  const int num_nodes = hg.nodes_.size();
  std::vector<WeightType> vit_weight(num_nodes, WeightType());

  for (int i = 0; i < num_nodes; ++i) {

    const Hypergraph::Node& cur_node = hg.nodes_[i];
    WeightType* const cur_node_best_weight = &vit_weight[i];

    const unsigned num_in_edges = min(static_cast<unsigned>(cur_node.in_edges_.size()), beam);
    if (num_in_edges == 0) {
      *cur_node_best_weight = WeightType(1);
      continue;
    }
    HG::Edge const* edge_best=0;
    for (unsigned j = 0; j < num_in_edges; ++j) {
      const HG::Edge& edge = hg.edges_[cur_node.in_edges_[j]];
      WeightType score = weight(edge);
      for (unsigned k = 0; k < edge.tail_nodes_.size(); ++k)
        score *= vit_weight[edge.tail_nodes_[k]];
      if (!edge_best || *cur_node_best_weight < score) {
        *cur_node_best_weight = score;
        edge_best=&edge;
      }
    }
  }
  return vit_weight.back();
}


/*
 * SOME TRAVERSAL FUNCTIONS 
 */
struct IRFeatureVectorTraversal {
  typedef SparseVector<double> Result;
  IRFeatureVectorTraversal(const CLIR::FDocument* doc,
                           const vector<FeatureVector>* ir_features,
                           const int bm25_fid,
                           const bool extract_smt, const bool extract_sparse, const bool extract_dense
                           ) : doc_(doc), ir_features_(ir_features), bm25_fid_(bm25_fid), extract_smt_(extract_smt), extract_sparse_(extract_sparse), extract_dense_(extract_dense) {}
  const CLIR::FDocument* doc_;
  const vector<FeatureVector>* ir_features_;
  const int bm25_fid_;
  const bool extract_smt_;
  const bool extract_sparse_;
  const bool extract_dense_;

  void operator()(HG::Edge const& edge, std::vector<Result const*> const& ants, Result* result) const {
    if (extract_smt_) // SMT feature values
      *result+=edge.feature_values_;
    const FeatureVector& fe = ir_features_->at(edge.id_); // IR features active at this edge
    //cerr << "Edge IR : " << fe << endl;
    if (!fe.empty()) {
      const WeightVector& doc_bm25 = doc_->bm25_q(); // unweighted bm25 values
      double edge_bm25 = 0;
      for (FeatureVector::const_iterator i=fe.begin();i!=fe.end();++i) {
        WeightVector::const_iterator wit = doc_bm25.find(i->first);
        if (wit != doc_bm25.end()) {
          if (extract_sparse_) (*result)[i->first] += wit->second;
          if (extract_dense_) edge_bm25 += wit->second;
        }
      }
      if (extract_dense_) (*result)[bm25_fid_] += edge_bm25;
    }
    // add in antecedents
    for (unsigned i = 0; i < ants.size(); ++i) *result += *ants[i];
  }
};

/*
 * IR Weight Function
 * - always includes the SMT score edge_prob_
 * - scores either dense or sparse IR features
 */
 struct IRWeightFunction {
  typedef prob_t Weight;
  IRWeightFunction(const CLIR::FDocument* doc,
                   const vector<FeatureVector>* ir_features,
                   const bool use_dense,
                   const double w_bm25) : doc_(doc), ir_features_(ir_features), use_dense_(use_dense), w_bm25_(w_bm25) {}
  const CLIR::FDocument* doc_;
  const vector<FeatureVector>* ir_features_;
  const bool use_dense_;
  const double w_bm25_;

  inline prob_t operator()(const Hypergraph::Edge& e) const {
    if (use_dense_) // dense IR feature (uses unweighted bm25_q is unweighted) * bm25 weight
      return e.edge_prob_ * prob_t::exp( (ir_features_->at(e.id_).dot(doc_->bm25_q())) * w_bm25_ );
    else // sparse IR features (assumes that bm25_q is already weighted)
      return e.edge_prob_ * prob_t::exp( ir_features_->at(e.id_).dot(doc_->wbm25_q()) );
  }
 };




/*
 * THIS IS FOR BOUNDING/PRUNING/UPPERBOUNDS
 */
struct TropicalValue {
  TropicalValue() : v_() {}
  TropicalValue(int v) {
    if (v == 0) v_ = prob_t::Zero();
    else if (v == 1) v_ = prob_t::One();
    else { cerr << "Bad value in TropicalValue(int).\n"; abort(); }
  }
  TropicalValue(unsigned v) : v_(v) {}
  TropicalValue(const prob_t& v) : v_(v) {}
//  operator prob_t() const { return v_; }
  inline TropicalValue& operator+=(const TropicalValue& o) {
    if (v_ < o.v_) v_ = o.v_;
    return *this;
  }
  inline TropicalValue& operator*=(const TropicalValue& o) {
    v_ *= o.v_;
    return *this;
  }
  inline bool operator==(const TropicalValue& o) const { return v_ == o.v_; }
  prob_t v_;
};
struct IRViterbiWeightFunction {
  typedef TropicalValue Weight;
  IRViterbiWeightFunction(const WeightVector* w,
                          const vector<FeatureVector>* ir_features) : w_(w), ir_features_(ir_features) {}
  const WeightVector* w_;
  const vector<FeatureVector>* ir_features_;

  inline TropicalValue operator()(const Hypergraph::Edge& e) const {
      return TropicalValue( e.edge_prob_ * prob_t::exp( ir_features_->at(e.id_).dot(*w_) ) );
  }
};

#endif
