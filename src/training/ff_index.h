// I DONT KNOW ANYMORE WHAT THIS IS!!!
#ifndef FF_INDEX_H_
#define FF_INDEX_H_

#include <vector>
#include <map>
#include <sstream>
#include <cassert>

#include "index.h"
#include "scoring.h"

#include "ff.h"
#include "hg.h"
#include "sparse_vector.h"
#include "filelib.h"
#include "stringlib.h"
#include "sentence_metadata.h"

using namespace std;

const string SGML_REL_MARKUP="rel";
const string FEAT_BASENAME="Relevance_";

class SparseRankFeatures : public FeatureFunction {
public:
	SparseRankFeatures(const std::string& param) : pt_(1), lp_(-1.0 / log(10)) {
		if (param.empty())
			cerr << "Relevance: penalty type: LP\n";
		else {
			const vector<string> argv = SplitOnWhitespace(param);
			assert(argv.size() == 1);
			pt_ = atoi(argv[0].c_str());
			cerr << "Relevance: penalty type: ";
			if (pt_ == 1) cerr << "LP\n";
			else if (pt_ == 2) cerr << "Junk\n";
			else {
				cerr << "None\n";
				pt_ = 0;
			}
		}
	}
	static std::string usage(bool p,bool d) {
	    return usage_helper("Relevance","penalty type","IR score of terminals to given documents with same relevance level (local feature)",p,d);
	}
protected:
	virtual void TraversalFeaturesImpl(const SentenceMetadata& smeta,
			const HG::Edge& edge,
			const std::vector<const void*>& ant_contexts,
			SparseVector<double>* features,
			SparseVector<double>* estimated_features,
			void* context) const;

	virtual void PrepareForInput(const SentenceMetadata& smeta);
private:
	/*
	 * loads relevance vectors from the given file
	 */
	void loadRelevanceFile(const std::string& fname) {
		ReadFile input(fname);
		rels_.clear();
		string rl, raw_scores, key;
		float val;
		while(input->good()) {
			*input >> rl; // read relevance level
			getline(*input,raw_scores);
			if (raw_scores.empty())
				continue;
			SparseVector<float> scores;
			stringstream in(Trim(raw_scores, " \t"));
			while(in.good()) {
				in >> key;
				in >> val;
				if (val != 0) scores[ TD::Convert(key) ] = val;
			}
			assert(!scores.empty());
			rels_[ FD::Convert(FEAT_BASENAME + rl) ] = scores;
		}
		cerr << "  Relevance values loaded from '" << fname << "'. " << rels_.size() << " relevance levels\n";
	}

	map<int,SparseVector<float> > rels_;
	unsigned short pt_; // penalty type
	const float lp_; // length penalty (see WordPenalty)

	mutable std::map<const TRule*, SparseVector<double> > rule2feats_;
};

void SparseRankFeatures::TraversalFeaturesImpl(const SentenceMetadata& smeta,
		const HG::Edge& edge,
		const std::vector<const void*>& /*ant_contexts*/,
		SparseVector<double>* features,
		SparseVector<double>* /*estimated_features*/,
		void* /*context*/) const {

	for (int i=0;i<edge.rule_->ELength();++i) {
		WordID w = edge.rule_->e_[i];
		if (w >= 1) { // is terminal
			bool isrel=false;
			for (map<int,SparseVector<float> >::const_iterator rlit=rels_.begin(); rlit!=rels_.end(); ++rlit) {
				SparseVector<float>::const_iterator bm25it = rlit->second.find(w);
				if (bm25it != rlit->second.end() && bm25it->second != 0) { // if w in rels_[rl]
					features->add_value( rlit->first, bm25it->second );
					isrel=true;
				}
			}
			// Junk word penalty
			if (pt_==2 && !isrel) features->add_value(FD::Convert(FEAT_BASENAME + "0"),-1);
		}
	}
	// length penalty
	if (pt_==1) features->set_value(FD::Convert(FEAT_BASENAME + "0"), edge.rule_->EWords() * lp_);

}
void SparseRankFeatures::PrepareForInput(const SentenceMetadata& smeta) {
	string fname = smeta.GetSGMLValue(SGML_REL_MARKUP);
	if (fname.empty())
		cerr << "Warning! Input '" << smeta.GetSGMLValue("id") << "' (length=" << smeta.GetSourceLength() << ") has no '" << SGML_REL_MARKUP << "' markup!\n";
	else
		loadRelevanceFile(fname);
}
#endif
