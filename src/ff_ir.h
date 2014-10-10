#ifndef _FF_IR_H_
#define _FF_IR_H_

#include <vector>
#include <map>
#include "prob.h"
#include "hg.h"
#include "tdict.h"
#include "fdict.h"
#include "irfdict.h"
#include <queue>

#include "clir.h"
#include "index.h"
#include "dftable.h"
#include "weights.h"
#include "feature_vector.h" // typedefs FeatureVector, WeightVector, DenseWeightVector

namespace CLIR {

string escape_token(const string& s) {
	string y = s;
	for (unsigned i = 0; i < y.size(); ++i) {
	  if (y[i] == '=') y[i]='_';
	  if (y[i] == ';') y[i]='_';
	}
	return y;
}

/*
 * converts between term dictionary of document terms and corresponding IR feature match ids
 */
WordID CreateIRFID(const WordID& t) {
  	return FD::Convert("IR:" + escape_token(TD::Convert(t))); // FOR NOW USE A SINGLE FEATURE SPACE!
}

// IR feature extractor:
// extracts feature ids and values for sparse lexical ir features, e.g. IR:word=1 if lexical.
// also extract IR:cluster=1 if cluster_classes
// the extractor should handle the feature VALUES.
// so given a document it should return an FDocument with the right vectors (replcaes the FDocument.convert() method)
// and it should handle giving the indicators for the search space if called from indicator
struct IRFeatureExtractor {
	IRFeatureExtractor(const WeightVector* w_ir, const double w_default, const string& cluster_file="") : w_ir_(w_ir), w_default_(w_default), word_classes_(false), unk_(TD::Convert("<unk>")) {
		cerr << "IRModel(|W|=" << w_ir_->size() << ",default="<<w_default_<<",class_file='"<<cluster_file<<"')\n";
		if (!cluster_file.empty()) {
			word_classes_ = true;
			read_cluster_file(cluster_file);
			cerr << "cluster file loaded.\n";
		}
	}

	const WeightVector* w_ir_;
	const double w_default_;
	bool word_classes_;
	const string fid_prefix = "IR:";
	vector<WordID> cluster_map_;
	const WordID unk_;


	// returns vector containing the feature ids for a given term.
	vector<int> extract(const WordID term) const {
		vector<int> f;
		f.push_back(FD::Convert(fid_prefix + escape_token(TD::Convert(term))));
		if (word_classes_) {
			WordID cluster_id = to_cluster(term);
			if (cluster_id != 0)
				f.push_back(FD::Convert(fid_prefix + TD::Convert(cluster_id)));
		}
		return f;
	}

	// converts a TermVector from a document into the IRFeature space
	WeightVector convert(const TermVector& v) {
		WeightVector f_ir;
		for (TermVector::const_iterator it=v.begin();it!=v.end();++it) {
			const vector<int> fids = extract(it->first);
			for (vector<int>::const_iterator fid_it=fids.begin();fid_it!=fids.end();++fid_it) {
				// we need to use add_value if a multiple words in document map to the same word class
				f_ir.add_value(*fid_it, it->second.as_float());
			}
		}
		return f_ir;
	}

	// queries the global w_ir vector, and if no weight found returns the default weight
	double get_weight(const weight_t k) const {
		WeightVector::const_iterator it = w_ir_->find(k);
		if (it == w_ir_->end()) // new IR feature
			return w_default_;
		return it->second;
	}

	void read_cluster_file(const string& clusters) {
		// FORMAT IS cluster word
		ReadFile rf(clusters);
	    istream& in = *rf.stream();
	    string line;
	    int lc = 0;
	    string cluster;
	    string word;
	    while(getline(in, line)) {
	      ++lc;
	      if (line.size() == 0) continue;
	      if (line[0] == '#') continue;
	      unsigned cend = 1;
	      while((line[cend] != ' ' && line[cend] != '\t') && cend < line.size()) {
	        ++cend;
	      }
	      if (cend == line.size()) {
	        cerr << "Line " << lc << " in " << clusters << " malformed: " << line << endl;
	        abort();
	      }
	      unsigned wbeg = cend + 1;
	      while((line[wbeg] == ' ' || line[wbeg] == '\t') && wbeg < line.size()) {
	        ++wbeg;
	      }
	      if (wbeg == line.size()) {
	        cerr << "Line " << lc << " in " << clusters << " malformed: " << line << endl;
	        abort();
	      }
	      unsigned wend = wbeg + 1;
	      while((line[wend] != ' ' && line[wend] != '\t') && wend < line.size()) {
	        ++wend;
	      }
	      const WordID clusterid = TD::Convert(line.substr(0, cend));
	      const WordID wordid = TD::Convert(line.substr(wbeg, wend - wbeg));
	      if (wordid >= cluster_map_.size())
	        cluster_map_.resize(wordid + 10, unk_);
	      cluster_map_[wordid] = clusterid;
	    }
	 }

	WordID to_cluster(const WordID w) const {
		if (cluster_map_.size() == 0) return 0;
		if (w >= cluster_map_.size()) return 0;
		return cluster_map_[w];
  	}
};

/*
 * new version of IRFeatureFunction: does not compute RSJ part here, only looks at edges where possible matches can occur
 */
// the IRFeatureIndicator goes through the search space and looks for edges where IRFeatures may occur
class IRFeatureIndicator {
public:
	IRFeatureIndicator(const IRFeatureExtractor* extractor, const DfTable* dft) : bm25_fid(FD::Convert("IR:BM25")), extractor_(extractor), dft_(dft)  {}

	void apply(const Hypergraph& hg, // search space
			   vector<FeatureVector>& ir_features,
			   vector<bool>& ir_edges, // marks edges with features true
			   WeightVector& ir_active, // contains IR features ids that are active in this search space
			   const bool include_dense = false,
			   WeightVector* best_bm25 = NULL,
			   const unsigned N=0) {

		clear_cache();
		const unsigned hg_edges_size = hg.edges_.size();
		ir_features.resize(hg_edges_size);
		ir_edges.resize(hg_edges_size); // indicates if an edge can possibly match
		ir_active.clear();
		if (best_bm25) best_bm25->clear();

		unsigned cme = 0; // count matching edges
		for (unsigned i=0; i<hg_edges_size;++i) {
			map<const TRule*, SparseVector<double> >::iterator cache_it = rule2feats_.find(hg.edges_[i].rule_.get());
			if (cache_it == rule2feats_.end()) { // not in rule cache yet
				const TRule& rule = *hg.edges_[i].rule_;
				cache_it = rule2feats_.insert(make_pair(&rule, FeatureVector())).first;
				FeatureVector& f = cache_it->second;
				for (unsigned short j=0;j<rule.ELength();++j) {
					const WordID t = rule.e_[j];
					if (t>=1 && !( sw::isPunct(t) || sw::isStopword(t) ) && dft_->hasKey(t)) { // if useful terminal, in collection, fire IR features
						const vector<int> tf = extractor_->extract(t);
						for (vector<int>::const_iterator tfit=tf.begin();tfit!=tf.end(); ++tfit) {
							f.add_value(*tfit, 1.0); // indicator
							const double w = extractor_->get_weight(*tfit);
							ir_active.set_value(*tfit, w);
							if (best_bm25) // for bounding etc
								best_bm25->set_value(*tfit, CLIR::BM25::RSJ(dft_->get(t), N) * CLIR::BM25::TF1 * w);
						}
					}
				}
			}
			if (cache_it->second.size() != 0) {
				++cme;
				ir_edges[i] = true;
				ir_features[i] += cache_it->second;
			}
		}

		if ( include_dense && !ir_active.empty() ) ir_active[bm25_fid] = extractor_->get_weight(bm25_fid);

		cerr << "IRFF: " << cme << " of " 
		     << hg_edges_size << " IR relevant (" << (cme/(double) hg_edges_size)*100 << "%), " 
		     << ir_active.size() << " IR features.\n";

		/*vector<bool> ir_nodes(hg.nodes_.size(), false); // indicates if a node has incoming IR edges
		// try to get some useful structure information from the hypergraph
		for (unsigned i=0; i<hg.nodes_.size();i++) {
			const Hypergraph::Node& cur_node = hg.nodes_[i];
			const unsigned num_in_edges = cur_node.in_edges_.size();
			const unsigned num_out_edges = cur_node.out_edges_.size();
			//if (num_in_edges!=0) continue; // select only leaf nodes
			int num_ir_in_edges =0;
			for (unsigned j = 0; j < num_in_edges; ++j) {
				const unsigned edge_idx = cur_node.in_edges_[j];
				const HG::Edge& edge = hg.edges_[edge_idx];
				if (ir_edges[edge_idx])  {
					ir_nodes[i] = true;
					num_ir_in_edges++;
				}
			}
			int num_ir_out_edges =0;
			for (unsigned j = 0; j < num_out_edges; ++j) {
				const unsigned edge_idx = cur_node.out_edges_[j];
				const HG::Edge& edge = hg.edges_[edge_idx];
				if (ir_edges[edge_idx])  {
					num_ir_out_edges++;
				}
			}
			cout << "Node " << i << " in_edges=" << num_in_edges << " out_edges=" << num_out_edges << " span=" << hg.NodeSpan(i) << " IR=" <<ir_nodes[i] << " num_ir_in_edges=" << num_ir_in_edges << " num_ir_out_edges=" << num_ir_out_edges << endl;
		}*/

	}
	void clear_cache() { rule2feats_.clear(); }
	int bm25_fid;
private:
	const IRFeatureExtractor* extractor_;
	const DfTable* dft_;
	mutable std::map<const TRule*, FeatureVector > rule2feats_;
};

// represents a document in the IR feature space
struct FDocument {

	FDocument(const string& id, const unsigned len, const WeightVector& bm25) : id_(TD::Convert(id)), len_(len), bm25_(bm25) {}

	const WordID& id() const { return id_; }
	const unsigned& len() const { return len_; }
	const WeightVector& bm25()    const { return bm25_; }
	const WeightVector& bm25_q()  const { return bm25_q_;  }
	const WeightVector& wbm25_q() const { return wbm25_q_;  }
	weight_t bm25(const int k)    const { return bm25_.get(k); }
	weight_t bm25_q(const int k)  const { return bm25_q_.get(k); }
	weight_t wbm25_q(const int k) const { return wbm25_q_.get(k); }

	bool compute_bm25_q(const WeightVector& ir_active, const bool build_unweighted=false) {
		bm25_q_.clear();
		wbm25_q_.clear();
		for (WeightVector::iterator it=bm25_.begin();it!=bm25_.end();++it) {
			WeightVector::const_iterator wit = ir_active.find(it->first);
			if (wit != ir_active.end()) {
				if (build_unweighted) // necessary for dense feature and feature vector extraction
					bm25_q_.set_value( it->first, it->second ); // raw BM25
				wbm25_q_.set_value( it->first, it->second * wit->second ); // BM25 * w
			}
		}
		return !wbm25_q_.empty(); // returns true if matched something
	}

	WordID id_;
	unsigned len_;
	WeightVector bm25_; // classic bm25 vector of document in the IR feature space (created by convert())
	WeightVector bm25_q_; // classic bm25 vector filtered by query search space
	WeightVector wbm25_q_; // same but weighted with ir weights already
};



/*
 * same as loadDocuments() in clir.h, but creates FDocument instances (writing into feature space FD)
 */
double loadDocuments(const string& fname, vector<CLIR::FDocument>& documents, IRFeatureExtractor* extractor) {
	TIMER::timestamp_t t0 = TIMER::get_timestamp();
	boost::interprocess::file_mapping m_file(fname.c_str(), boost::interprocess::read_only);
	boost::interprocess::mapped_region region(m_file, boost::interprocess::read_only);
	void * addr       = region.get_address();
	std::size_t size  = region.get_size();
	char *data = static_cast<char*>(addr);
	std::istringstream in;
	in.rdbuf()->pubsetbuf(data, size);
	string docid, raw;
	int len;
	double sumDocLen = 0.0;
	while (in >> docid) {
		in.ignore(1,'\t');
		in >> len;
		in.ignore(1,'\t');
		getline(in, raw);
		if (docid.size() == 0 || len <= 0 || raw.size() == 0)
			continue;
		documents.emplace_back(docid, len, extractor->convert(vutils::read_vector(raw))); // construct document in place
		sumDocLen += len;
	}
	unsigned n = documents.size();
	TIMER::timestamp_t t1 = TIMER::get_timestamp();
	float time = (t1-t0) / 1000000.0L;
	cerr << "loaded " << n << " fdocuments. (average length=" << sumDocLen / (double) n << ") [" << time << "s]\n";
	return sumDocLen / (double) n;
}


} // end of namespace

#endif
