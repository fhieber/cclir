/*
 * observer.h
 *
 *  Created on: Apr 22, 2013
 */

#ifndef OBSERVER_H_
#define OBSERVER_H_

#include <vector>
#include <string>
#include <queue>

#include "kbest.h" // cdec
#include "viterbi.h"
#include "inside_outside.h"
#include "sparse_vector.h"
#include "sampler.h"
#include "verbose.h"
#include "viterbi.h"
#include "ff_register.h"
#include "decoder.h"
#include "weights.h"
#include "prob.h"

#include "TrainingInstance.h"


/*
 * implements a traversal struct that collects translations for all rules
 * used in the derivation.
 * ( compare with viterbi.h's ViterbiPathTraversal )
 */

struct TranslationPairTraversal {
	typedef std::vector<translation> Result;
	void operator()(const Hypergraph::Edge& edge,
					const std::vector<const Result*>& ants,
					Result* result) const {
		for (unsigned i=0; i<ants.size(); ++i) {
			for (unsigned j=0;j<ants[i]->size(); ++j)
				result->push_back((*ants[i])[j]);
		}
		for (unsigned j = 0; j < edge.rule_->a_.size(); ++j)
			result->push_back(std::make_pair( edge.rule_->f_[edge.rule_->a_[j].s_] , edge.rule_->e_[edge.rule_->a_[j].t_]) );
	}
};

/*
 * abstract class that defines interface for either getting derivations
 * for a training instance by using the kbest list, or by sampling from
 * forest
 */

struct InstanceGetter : public DecoderObserver {

	const unsigned k_; // no of chunks
	const bool ignore_derivation_scores_;
	const bool swf_; // stopword filter?
	const int chunksize_;
	const prob_t L_;
	const prob_t C_;
	const WeightVector weights_;
	TrainingInstance instance_;

	InstanceGetter(
				const unsigned& k,
				const bool ignore_derivation_scores,
				const bool swf,
				const int& chunksize,
				const prob_t& L,
				const prob_t& C,
				const WeightVector& weights
			) :
				k_(k),
				ignore_derivation_scores_(ignore_derivation_scores),
				swf_(swf),
				chunksize_(chunksize),
				L_(L),
				C_(C),
				weights_(weights)
			{}

	virtual ~InstanceGetter() {};

	TrainingInstance& GetTrainingInstance() { return instance_; }

	virtual void NotifyTranslationForest(const SentenceMetadata& /*smeta*/, Hypergraph* hg) {
		CreateInstanceFromKDerivations(*hg);
	}

	virtual void CreateInstanceFromKDerivations(const Hypergraph& forest) = 0;

};

/*
 * DERIVATIONS FROM KBEST LIST
 */

struct KbestInstanceGetter : public InstanceGetter {

	const bool unique_kbest_; // unique kbest list?
	/*
	 * unique filter for kbest derivations
	 */
	struct FilterUnique {
		boost::unordered_set<std::vector<translation> > unique;
		bool operator()(const std::vector<translation>& yield) {
			return !unique.insert(yield).second;
		}
	};

	KbestInstanceGetter(
				const unsigned& k,
				const bool unique,
				const bool ignore_derivation_scores,
				const bool swf,
				const int& chunksize,
				const prob_t& L,
				const prob_t& C,
				const WeightVector& weights
			) :
				InstanceGetter(k, ignore_derivation_scores, swf, chunksize, L, C, weights),
				unique_kbest_(unique)
			{}

	void CreateInstanceFromKDerivations(const Hypergraph& forest) {
		TrainingInstance instance;
		if (!unique_kbest_)
			KbestGet<KBest::NoFilter<std::vector<translation> > >(forest, instance);
		else
			KbestGet<FilterUnique>(forest, instance);

		instance.SetFeaturePermutations();
		instance_ = instance;
	}

	template<class Filter>
	void KbestGet(const Hypergraph& forest, TrainingInstance& instance) {

		typedef KBest::KBestDerivations<std::vector<translation>,
			TranslationPairTraversal, Filter> K;

		K kbest(forest, k_);

		NbestTTable nbest_ttable;
		SparseVector<double> feature_vector; // accumulated feature_vector
		double chunk_score = 0.0; // accumulated chunk_score
		int cur_chunksize = 0; // to trigger start of new chunk
		// for each item in kbest list:
		for (int i=0; i<k_;++i) {

			typename K::Derivation *d = kbest.LazyKthBest(
								forest.nodes_.size() - 1, i);

			if (!d) break;
			if ( d->yield.empty() ) { continue; }

			// create optionTables for each single items (chunks of size 1) // TODO

			for (std::vector<translation>::const_iterator t_it=d->yield.begin(); t_it!=d->yield.end();t_it++)
				nbest_ttable.addPair(t_it->first, t_it->second, ignore_derivation_scores_ ? prob_t(1) : d->score);

			// add current feature vector to accumulated feature vector
			feature_vector += d->feature_values; // feature values are log values!!
			chunk_score += d->score.v_; // derivation score is log value!!!
			++cur_chunksize;

			if (/*i == k_-1 ||*/ cur_chunksize == chunksize_) { // start new chunk

				// write current chunk
				nbest_ttable.normalize();
				stringstream nbest_ttable_ss;
				nbest_ttable.save(nbest_ttable_ss, L_, C_, " ", swf_);
				instance.AddChunk(nbest_ttable_ss.str(), feature_vector, chunk_score, weights_);
				// reset
				cur_chunksize = 0;
				nbest_ttable.clear(); // = NbestTTable();
				feature_vector.clear(); //= SparseVector<double>();
				chunk_score = 0.0;

			}
		}
	}

};

/*
 * DERIVATIONS BY SAMPLING FROM FOREST
 */

 // adapted hypergraph sampler to use own hypothesis format (vector<translation>)
struct HypergraphSampler {

	struct SampledDerivationWeightFunction {
		typedef double Weight;
		explicit SampledDerivationWeightFunction(const vector<bool>& sampled) : sampled_edges(sampled) {}
		double operator()(const Hypergraph::Edge& e) const {
			return static_cast<double>(sampled_edges[e.id_]);
		}
		const vector<bool>& sampled_edges;
	};

	struct Hypothesis {
		std::vector<translation> translations;
		SparseVector<double> fmap;
		prob_t model_score; // log unnormalized probability
		bool operator>(const Hypothesis& o) const { return model_score > o.model_score; }
	};

	static void
	sample_hypotheses(const Hypergraph& hg,
			unsigned n,
			MT19937* rng,
			vector<Hypothesis>* hypos) {
		hypos->clear();
		hypos->resize(n);

		// compute inside probabilities
		vector<prob_t> node_probs;
		Inside<prob_t, EdgeProb>(hg, &node_probs, EdgeProb());

		vector<bool> sampled_edges(hg.edges_.size());
		queue<unsigned> q;
		SampleSet<prob_t> ss;
		for (unsigned i = 0; i < n; ++i) {
			fill(sampled_edges.begin(), sampled_edges.end(), false);
			// sample derivation top down
			assert(q.empty());
			Hypothesis& hyp = (*hypos)[i];
			SparseVector<double>& deriv_features = hyp.fmap;
			q.push(hg.nodes_.size() - 1);
			prob_t& model_score = hyp.model_score;
			model_score = prob_t::One();
			while(!q.empty()) {
				unsigned cur_node_id = q.front();
				q.pop();
				const Hypergraph::Node& node = hg.nodes_[cur_node_id];
				const unsigned num_in_edges = node.in_edges_.size();
				unsigned sampled_edge_idx = 0;
				if (num_in_edges == 1) {
					sampled_edge_idx = node.in_edges_[0];
				} else {
					assert(num_in_edges > 1);
					ss.clear();
					for (unsigned j = 0; j < num_in_edges; ++j) {
						const Hypergraph::Edge& edge = hg.edges_[node.in_edges_[j]];
						prob_t p = edge.edge_prob_;   // edge weight
						for (unsigned k = 0; k < edge.tail_nodes_.size(); ++k)
							p *= node_probs[edge.tail_nodes_[k]];  // tail node inside weight
						ss.add(p);
					}
					sampled_edge_idx = node.in_edges_[rng->SelectSample(ss)];
				}
				sampled_edges[sampled_edge_idx] = true;
				const Hypergraph::Edge& sampled_edge = hg.edges_[sampled_edge_idx];
				deriv_features += sampled_edge.feature_values_;
				model_score *= sampled_edge.edge_prob_;
				//sampled_deriv->push_back(sampled_edge_idx);
				for (unsigned j = 0; j < sampled_edge.tail_nodes_.size(); ++j) {
					q.push(sampled_edge.tail_nodes_[j]);
				}
			}
			Viterbi(hg, &hyp.translations, TranslationPairTraversal(), SampledDerivationWeightFunction(sampled_edges));
		}
	}
};

struct ForestSampleInstanceGetter : public InstanceGetter {

	MT19937* prng_;

	ForestSampleInstanceGetter(
				const unsigned& k,
				MT19937* prng,
				const bool& ignore_derivation_scores,
				const bool& swf,
				const int& chunksize,
				const prob_t& L,
				const prob_t& C,
				const WeightVector& weights
			) :
				InstanceGetter(k, ignore_derivation_scores, swf, chunksize, L, C, weights),
				prng_(prng)
			{}

	void CreateInstanceFromKDerivations(const Hypergraph& forest) {
		TrainingInstance instance;
		KSample(forest, instance);
		instance.SetFeaturePermutations();
		instance_ = instance;
	}

	void KSample(const Hypergraph& forest, TrainingInstance& instance) {
		std::vector<HypergraphSampler::Hypothesis> samples;
		HypergraphSampler::sample_hypotheses(forest, k_, prng_, &samples);
		std::stable_sort(samples.begin(), samples.end(), std::greater<HypergraphSampler::Hypothesis>());
		for (unsigned i = 0; i < k_; ++i) { // for each sample (TODO implement chunksize)
			NbestTTable nbest_ttable;
			for (vector<translation>::const_iterator it=samples[i].translations.begin(); it!= samples[i].translations.end(); ++it)
				nbest_ttable.addPair(it->first, it->second, ignore_derivation_scores_ ? prob_t(1) : samples[i].model_score);
			nbest_ttable.normalize();
			stringstream nbest_ttable_ss;
			nbest_ttable.save(nbest_ttable_ss, L_, C_, " ", swf_);
			instance.AddChunk(nbest_ttable_ss.str(), samples[i].fmap, samples[i].model_score.v_, weights_);
		}
	}

};

#endif /* OBSERVER_H_ */
