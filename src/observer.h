/*
 * observer.h
 *
 *  Created on: May 27, 2013
 */

#ifndef OBSERVER_H_
#define OBSERVER_H_

#include <vector>
#include <string>
#include <queue>

#include <boost/unordered_set.hpp>

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
#include "nbest-ttable.h"


/*
 * implements a traversal struct that collects translations for all rules
 * used in the derivation.
 * ( compare with viterbi.h's ViterbiPathTraversal )
 */
struct TranslationPairTraversal {

	typedef std::vector<translation> Result;
	
	inline void selectAlignments(const Hypergraph::Edge& edge, std::vector<translation>& result, const bool filter=false) const {
	
		if (!filter) { // use all alignments
		
			for (unsigned j = 0; j < edge.rule_->a_.size(); ++j)
				result.push_back(std::make_pair( edge.rule_->f_[edge.rule_->a_[j].s_] , edge.rule_->e_[edge.rule_->a_[j].t_]) );
				
		} else { // only one-to-one
		
			// we assume that alignment points are ordered by source index
			std::vector<AlignmentPoint>& as = edge.rule_->a_;
			std::vector<unsigned short> cf(edge.rule_->FLength(),0); // counts times source tokens are visited by as
			std::vector<unsigned short> ce(edge.rule_->ELength(),0); // counts times target tokens are visited by as
			for (unsigned j=0;j<as.size();++j) {
				++cf[as[j].s_];
				++ce[as[j].t_];
			}	
			for (unsigned j=0;j<as.size();++j) {
				if (cf[as[j].s_] == 1 && ce[as[j].t_] == 1) // if both s_ and t_ only visited once
					result.push_back(std::make_pair( edge.rule_->f_[as[j].s_] , edge.rule_->e_[as[j].t_]) );
			}
		}
	}	
	
	void operator()(const Hypergraph::Edge& edge,
					const std::vector<const Result*>& ants,
					Result* result) const {
		for (unsigned i=0; i<ants.size(); ++i) {
			for (unsigned j=0;j<ants[i]->size(); ++j)
				result->push_back((*ants[i])[j]);
		}	
		if (!edge.rule_->a_.empty())
			selectAlignments(edge, *result, false);	
	}
	
};

struct NbestTTableGetter : public DecoderObserver {

	/*
	 * unique filter for kbest derivations
	 */
	struct FilterUnique {
		boost::unordered_set<std::vector<translation> > unique;
		bool operator()(const std::vector<translation>& yield) {
			return !unique.insert(yield).second;
		}
	};

	NbestTTable nbest_ttable_;
	const unsigned n_; // size of nbest
	const bool ignore_derivation_scores_;
	const prob_t L_;
	const prob_t C_;
	const bool unique_nbest_;
	const bool swf_; // stopword filter?

	NbestTTableGetter(
			const unsigned n,
			const bool ignore_derivation_scores,
			const bool /*add_passthrough_rules*/, // TODO
			const prob_t& L,
			const prob_t& C,
			const bool unique_nbest,
			const bool target_swf)
			: n_(n),
			  ignore_derivation_scores_(ignore_derivation_scores),
			  L_(L), C_(C),
			  unique_nbest_(unique_nbest),
			  swf_(target_swf) {
		nbest_ttable_.clear();
	}

	template<class Filter>
	void KbestGet(const Hypergraph& hg) {

		typedef KBest::KBestDerivations<std::vector<translation>,
			TranslationPairTraversal, Filter> K;

		K kbest(hg, n_);
		// for each item in kbest list:
		for (int i=0; i<n_;++i) {
			typename K::Derivation *d = kbest.LazyKthBest(
								hg.nodes_.size() - 1, i);
			if (!d) break;
			if ( d->yield.empty() ) { continue; }
			for (std::vector<translation>::const_iterator t_it=d->yield.begin(); t_it!=d->yield.end();t_it++)
				nbest_ttable_.addPair(t_it->first, t_it->second, ignore_derivation_scores_ ? prob_t(1) : d->score);
		}
	}

	void GetNbestTTableFromForest(Hypergraph* hg) {
		if (!unique_nbest_)
			KbestGet<KBest::NoFilter<std::vector<translation> > >(*hg);
		else
			KbestGet<FilterUnique>(*hg);

	}

	virtual void NotifyTranslationForest(const SentenceMetadata& /*smeta*/, Hypergraph* hg) {
		ClearNbestTTable();
		GetNbestTTableFromForest(hg);
		NormalizeNbestTTable();
	}

	void ClearNbestTTable() { nbest_ttable_.clear(); }
	void NormalizeNbestTTable() { nbest_ttable_.normalize(); }
	void ConstrainNbestTTable() { nbest_ttable_.constrain(L_, C_, swf_); }
	NbestTTable& GetNbestTTable() { return nbest_ttable_; }

};

#endif /* OBSERVER_H_ */
