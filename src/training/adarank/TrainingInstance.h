/*
 * instance.h
 *
 *  Created on: Apr 22, 2013
 */

#ifndef INSTANCE_H_
#define INSTANCE_H_

#include <iostream>
#include <iomanip>

#include <assert.h>
#include <boost/unordered_map.hpp>

#include "losses.h"
#include "relevance.h"

#include "sparse_vector.h"
#include "weights.h"

#include "src/core/nbest-ttable.h"
#include "src/core/serialization.h"


typedef std::vector<double> DenseFeatureVector;
typedef FastSparseVector<double> SparseFeatureVector;
typedef std::vector<weight_t> DenseWeightVector;
typedef FastSparseVector<weight_t> WeightVector;

class TrainingInstance {
	typedef std::string Feature;
public:
	TrainingInstance() : maxRank(0), nbest_ttables_parsed(false), retrieval_perm_set(false) {};
	void SetID(const string& id) { qid = id; }
	const string& GetID() const { return qid; }

	void AddChunk(const string& nbt, const SparseVector<double>& feature_vector, const double chunk_score, const WeightVector& weights) {
		nbest_ttable_strings.push_back(nbt);
		decoder_perm.push_back(maxRank);
		decoder_scores.push_back(chunk_score);
		
		assert( fabs(chunk_score - feature_vector.dot(weights)) < 0.00001 );
		//cerr << " ||| " << chunk_score <<  " " << feature_vector.dot(weights) << " " << dot << endl;
		
		for(WeightVector::const_iterator kf=weights.begin();kf!=weights.end();++kf)
			feature_column_vectors[FD::Convert(kf->first)].push_back(feature_vector.get(kf->first));
			
		++maxRank;
	}

	void SetFeaturePermutations() {
		if (maxRank == 0) return;
		assert(nbest_ttable_strings.size() == maxRank);
		assert(decoder_perm.size() == maxRank);
		assert(decoder_perm[maxRank-1] == maxRank-1);

		// all vectors of values in feature_columns now should have the same size!!!!
		boost::unordered_map<Feature,DenseFeatureVector>::const_iterator fcv_it;
		for(fcv_it=feature_column_vectors.begin(); fcv_it!=feature_column_vectors.end(); ++fcv_it) {
			assert(fcv_it->second.size() == maxRank);
			Permutation p(decoder_perm); // permutation p
			std::stable_sort( p.begin(), p.end(), Comparator( &(fcv_it->second) ) ); // sort permutation by current feature vals
			feature_perms[fcv_it->first] = p;
		}
	}

	Permutation GetWeightedPermutation(WeightVector& weights) const {
		vector<double> weighted_scores(maxRank,0.0);
		boost::unordered_map<Feature,DenseFeatureVector>::const_iterator fcv_it;
		for(fcv_it=feature_column_vectors.begin(); fcv_it!=feature_column_vectors.end(); ++fcv_it) {
			for (int k=0;k<fcv_it->second.size();++k)
				weighted_scores[k] += fcv_it->second[k] * weights.value( FD::Convert(fcv_it->first) );
		}
		Permutation p(decoder_perm);
		std::stable_sort( p.begin(), p.end(), Comparator( &weighted_scores) );
		return p;
	}

	Permutation GetWeightedPermutation(DenseWeightVector& weights) const {
		vector<double> weighted_scores(maxRank,0.0);
		boost::unordered_map<Feature,DenseFeatureVector>::const_iterator fcv_it;
		for(fcv_it=feature_column_vectors.begin(); fcv_it!=feature_column_vectors.end(); ++fcv_it) {
			for (int k=0;k<fcv_it->second.size();++k)
				weighted_scores[k] += fcv_it->second[k] * weights[FD::Convert(fcv_it->first)];
		}
		Permutation p(decoder_perm);
		std::stable_sort( p.begin(), p.end(), Comparator( &weighted_scores) );
		return p;
	}
	
	/*
	 * checks if the scores created with the given weight vector are 
	 * consistent with the saved decoder scores. So you should pass 
	 * the weight vector that was used for decoding/creating this instance
	 */
	bool CheckWeightedPermutation(DenseWeightVector& weights, const bool verbose) const {
		vector<double> weighted_scores(maxRank,0.0);
		boost::unordered_map<Feature,DenseFeatureVector>::const_iterator fcv_it;
		for(fcv_it=feature_column_vectors.begin(); fcv_it!=feature_column_vectors.end(); ++fcv_it) {
			for (int k=0;k<fcv_it->second.size();++k)
				weighted_scores[k] += fcv_it->second[k] * weights[FD::Convert(fcv_it->first)];
		}
		Permutation p(decoder_perm);
		std::stable_sort( p.begin(), p.end(), Comparator( &weighted_scores) );
		
		if (verbose) {
			for (int i=0;i<maxRank;++i)
				cerr << decoder_perm[i] << "-" << p[i] << " ";
			cerr << "\n";
			for (int i=0;i<maxRank;++i)
				cerr << decoder_scores[i] << "==" << weighted_scores[i] << " ";
			cerr << "\n";
		}
		
		return p == decoder_perm;
		
	}

	void SetRelevance(Relevance& r) { relevance = r; }

	const Relevance& GetRelevance() const { return relevance; }

	bool isRelevant(const string& docid, const RLevel& rl) const { return relevance.is_relevant(docid, rl); }

	/*
	 * takes retrieval scores in the order of decoder permutation
	 */
	void SetGoldPermutation(const vector<double>& scores)  {
		retrieval_scores = scores;
		assert(retrieval_scores.size() == maxRank);
		retrieval_perm = Permutation(decoder_perm);
		std::stable_sort ( retrieval_perm.begin(), retrieval_perm.end(), Comparator ( &retrieval_scores ) );
		retrieval_perm_set = true;
	}

	int size() const { return maxRank; } 

	const Permutation* GetGoldPermutation() const { assert(retrieval_perm_set == true); return &retrieval_perm; }

	const Permutation* GetFeaturePermutation(const string& fname) const {
		if ( feature_perms.find(fname) != feature_perms.end() )
			return &(feature_perms.at(fname));
		return NULL;
	}

	/*
	 * this is ugly but necessary due to non persistency of cdecs term dictionary.
	 */
	void LoadNbestTTables() {
		nbest_ttables = std::vector<NbestTTable>(maxRank);
		for (unsigned short q=0; q<maxRank; ++q) {
			istringstream tmp(nbest_ttable_strings[q]);
			nbest_ttables[q] = NbestTTable(tmp);
		}
		nbest_ttables_parsed = true;
	}

	const string& GetNbestTTableString(const int& i) const { assert(i<maxRank); return nbest_ttable_strings[i]; }
	const NbestTTable& GetNbestTTable(const int& i) const { assert(nbest_ttables_parsed == true); return nbest_ttables[i]; }

	const Permutation* GetFeaturePermutation(const int& fid) const { return GetFeaturePermutation(FD::Convert(fid)); }

	const string SVMMapString() {
		stringstream out;
		boost::unordered_map<Feature,DenseFeatureVector>::const_iterator fcv_it;
		for (int k=0;k<maxRank;++k) {
			ChunkId& chunk = retrieval_perm[k];
			out << maxRank - k << " " // label
				<< "qid:" << qid << " "; // qid
			for (fcv_it=feature_column_vectors.begin(); fcv_it!=feature_column_vectors.end(); ++fcv_it)
				out << FD::Convert(fcv_it->first) << ":" << fcv_it->second[chunk] << " ";
			out << "\n";
		}
		return out.str();
	}

	std::string print() {
		ListwiseLossFunction* map = new MAP(false);
		ListwiseLossFunction* ndcg = new NDCG(false);
		ListwiseLossFunction* plackett = new PlackettLuce(false);
		stringstream ss;
		ss << "TRAINING INSTANCE: " << qid << " (" << maxRank << " chunks, "
		   << feature_perms.size() << " features, nbest_ttables_parsed="
		   << nbest_ttables_parsed << ", retrieval_perm_set=" << retrieval_perm_set
		   << ")" << "\n";
		ss << left << setw(20) << "Decoder";
		if (retrieval_perm_set) {
			ss << "MAP=" << map->score(decoder_perm, retrieval_perm)
			   << ", NDCG=" << ndcg->score(decoder_perm, retrieval_perm)
			   << ", PlackettLuce=" << plackett->score(decoder_perm, retrieval_perm)
			   << "\n" << setw(20) << " ";
		}
		Permutation::const_iterator it;
		for (it=decoder_perm.begin();it!=decoder_perm.end();++it)
			ss << setw(6) << *it << " ";
		ss << "\n";
		ss << left << setw(20) << " ";			
		for (it=decoder_perm.begin();it!=decoder_perm.end();++it)
			ss << setw(6) << fixed << setprecision(3) << decoder_scores[*it] << " ";
		
		ss << "\n";
		ss << left << setw(20) << "Retrieval";
		for (it=retrieval_perm.begin();it!=retrieval_perm.end();++it)
			ss << setw(6) << *it << " ";
		ss << "\n";
		ss << left << setw(20) << " ";
		for (it=retrieval_perm.begin();it!=retrieval_perm.end();++it)
			ss << setw(6) << fixed << setprecision(3) << retrieval_scores[*it] << " ";
		ss << "\n";		
		boost::unordered_map<Feature,Permutation>::const_iterator fit;
		for(fit=feature_perms.begin();fit!=feature_perms.end();++fit) {
			ss << left << setw(20) << fit->first;
			if (retrieval_perm_set) {
				ss << "MAP=" << map->score(feature_perms[fit->first], retrieval_perm)
				   << ", NDCG=" << ndcg->score(feature_perms[fit->first], retrieval_perm)
				   << ", Plackett=" << plackett->score(feature_perms[fit->first], retrieval_perm)
				   << "\n" << setw(20) << " ";
			}
			for(it=fit->second.begin();it!=fit->second.end();++it)
				ss << setw(6) << *it << " ";
			ss << "\n";
			ss << left << setw(20) << " ";
			for(it=fit->second.begin();it!=fit->second.end();++it)
				ss << setw(6) << fixed << setprecision(3) << feature_column_vectors[fit->first][*it] << " ";						
			
			ss << "\n";
		}
		ss << "\nNbestTTable strings:\n";
		for(std::vector<std::string>::const_iterator it = nbest_ttable_strings.begin();it!=nbest_ttable_strings.end();++it)
			ss << *it << "\n";
		ss << "\nRelevance Judgements:\n  "
		   << relevance.as_string()
		   << "\n\n";

		if (retrieval_perm_set) {
			delete map;
			delete ndcg;
		}

		return ss.str();
	}

private:
	std::string qid; // query id
	Relevance relevance; // relevant documents with relevance levels
	int maxRank; // number of chunks/items in the instance
	std::vector<std::string> nbest_ttable_strings; // nbest_ttables in string format (ugly but necessary for term dict unconstness)
	std::vector<NbestTTable> nbest_ttables; // this is not serialized!!! needs to be constructed from nbest_ttable_strings
	bool nbest_ttables_parsed;
	Permutation decoder_perm; // permutation given by decoding (original nbest list order)
	std::vector<double> decoder_scores; // (summed) derivation scores for each chunk
	Permutation retrieval_perm; // permutation given by retrieval step (ordering induced by retrieval scores / metrics )
	std::vector<double> retrieval_scores; // retrieval scores
	bool retrieval_perm_set;
	boost::unordered_map<Feature,DenseFeatureVector > feature_column_vectors;
	boost::unordered_map<Feature,Permutation > feature_perms; // permutations given by the features

	struct Comparator {
		Comparator(const std::vector<double>* fvals) : fvals(fvals) {};
		bool operator() (int a, int b) { return fvals->at(a) > fvals->at(b); }
		const std::vector<double>* fvals;
	};

	// serialization
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int /*version*/) {
		nbest_ttables_parsed = false;
		ar & qid;
		ar & maxRank;
		ar & nbest_ttable_strings;
		ar & nbest_ttables_parsed;
		ar & decoder_perm;
		ar & decoder_scores;
		ar & retrieval_perm;
		ar & retrieval_scores;
		ar & retrieval_perm_set;
		ar & BOOST_SERIALIZATION_NVP(feature_column_vectors);
		ar & BOOST_SERIALIZATION_NVP(feature_perms);
		ar & relevance;
	}
};

void loadInstances(const string& input_fname, vector<TrainingInstance>& instances) {
	instances.clear();
	std::ifstream in(input_fname.c_str());
	boost::archive::binary_iarchive ia(in, std::ios::binary | std::ios::in);
	ia >> instances;
	in.close();
	cerr << instances.size() << " instances loaded from '" << input_fname << "'.\n";
}

void saveInstances(const string& output_fname, vector<TrainingInstance>& instances) {
	std::ofstream out(output_fname.c_str(), std::ios::binary | std::ios::out);
	boost::archive::binary_oarchive oa(out);
	oa << instances;
	out.close();
	cerr << instances.size() << " serialized and written to '" << output_fname << "'.\n";
}

#endif /* INSTANCE_H_ */
