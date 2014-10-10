#ifndef _INVERTED_INDEX_H_
#define _INVERTED_INDEX_H_

#include <algorithm>
#include <utility>
#include <numeric>
#include <unordered_map>
#include <vector>
#include <forward_list>
#include <ostream>
#include <istream>
#include "tdict.h"
#include "document.h"
#include "util.h"

#include <iomanip>
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>

using namespace std;

namespace Index {

typedef unsigned int Docno;
typedef unsigned int TF;
typedef unsigned int DF;

struct Posting {
	Posting(Docno d, TF t) : docno(d), tf(t) {};
	Docno docno;
	TF tf;
};
// compare by docno
bool operator>(const Posting& a, const Posting& b) { return a.docno > b.docno; }
bool operator<=(const Posting& a, const Posting& b) { return !(a>b); }
bool operator<(const Posting& a, const Posting& b) { return a.docno < b.docno; }
bool operator>=(const Posting& a, const Posting& b) { return !(a<b); }
bool operator==(const Posting& a, const Posting& b) { return a.docno == b.docno; }
bool operator!=(const Posting& a, const Posting& b) { return !(a==b); }

class PostingsList { // TODO remove size. it is equal to df
public:
	PostingsList() : df_(0), size_(0) {};
	PostingsList(istream& in) { load(in); }
	unsigned int size() const { return size_; }
	void clear() { postings_.clear(); df_ = 0; size_ = 0; }
	void sort() { postings_.sort(); } // should be called after all documents were added

	DF df() const { return df_; }
	void set_df(DF df) { df_ = df; }
	void add(const Docno docno, const TF tf) {
		postings_.emplace_front(docno, tf);
		++size_;
		++df_; // TODO enable way of setting df values by hand!
	}
	void add(Posting&& posting) { // move semantic!
		postings_.push_front(posting);
		++size_;
		++df_;
	}
	void remove(const Docno docno) {
		postings_.remove(Posting(docno,0));
		--size_;
		--df_;
	}
	void remove(const Posting& p) {
		postings_.remove(p);
	}
	typedef forward_list<Posting>::iterator iterator;
	typedef forward_list<Posting>::const_iterator const_iterator;
	const_iterator begin() const { return postings_.cbegin(); }
	const_iterator end() const { return postings_.cend(); }
	void save(ostream& out) const {
		out << df_ << " " << size_;
		for(PostingsList::const_iterator i=begin();i!=end();++i) {
			out << " " << i->docno << " " << i->tf;
		}
	}
	void load(istream& in) {
		postings_.clear();
		PostingsList::const_iterator it = postings_.before_begin();
		in >> df_ >> size_;
		assert(df_ != 0);
		assert(size_ != 0);
		Docno docno;
		TF tf;
		for(unsigned int s=0;s<size_;++s) {
			in >> docno >> tf;
			it = postings_.emplace_after(it, docno, tf);
		}
		if (df_ != size_) {
			cerr << "PostingsList Error: df_="<<df_<<" size_="<<size_<<endl;
			abort();
		}
	}

private:
	DF df_;
	unsigned int size_;
	std::forward_list<Posting> postings_;
};

inline std::ostream& operator<<(std::ostream& out, const PostingsList& pl) {
	out << "df="<<pl.df()<<" size="<<pl.size()<<" ";
	for (PostingsList::const_iterator i=pl.begin();i!=pl.end();++i) out<<"("<<i->docno<<","<<i->tf<<") ";
	return out;
}

class Index {
public:
	Index() : n_(0), t_(0), avg_len_(0) {}
	Index(istream& in) { Load(in); }
	unsigned int NumberOfDocuments() const { return n_; }
	unsigned int NumberOfTerms() const { return t_; }
	double AverageDocumentLength() const { return avg_len_; }
	unsigned short DocumentLength(const Docno docno) const { assert(docno < n_); return lengths_.at(docno); }
	void clear() {
		documents_.clear();
		lengths_.clear();
		doc2docno_.clear();
		postingsLists_.clear();
		n_ = 0;
		t_ = 0;
		avg_len_ = 0;
	}

	typedef unordered_map<WordID,PostingsList>::const_iterator const_iterator;
	const_iterator cbegin() const { return postingsLists_.cbegin(); }
	const_iterator cend() const { return postingsLists_.cend(); }

	void ComputeAverageDocumentLength() {
		avg_len_ = (double) std::accumulate(lengths_.begin(), lengths_.end(), 0) / (double) n_;
	}
	Docno AddDocument(const CLIR::Document& doc) {
		// check if docname is already in index
		unordered_map<WordID,Docno>::const_iterator i = doc2docno_.find(doc.id_);
		if (i != doc2docno_.end()) {
			cerr << "Warning! Document " << TD::Convert(doc.id_) << " already in index!\n";
			return i->second;
		}
		Docno docno = documents_.size();
		doc2docno_[doc.id_] = docno;
		documents_.push_back(doc.id_);
		lengths_.push_back(doc.len_);
		// add in terms from document
		const TermVector& v = doc.v_;
		for (TermVector::const_iterator it=v.begin();it!=v.end();++it) {
			pair<unordered_map<WordID,PostingsList>::iterator,bool> res = postingsLists_.insert(pair<WordID,PostingsList>(it->first,PostingsList()));
			if (res.second) t_++;
			PostingsList& pl = res.first->second;
			pl.add( docno, round(it->second.as_float()) ); // IMPORTANT! we need to round up the floats to ints since the index stores unsigned ints!
		}
		n_++;
		return docno;
	}
	void AddDocuments(const vector<CLIR::Document>& docs) {
		for (vector<CLIR::Document>::const_iterator it=docs.begin();it!=docs.end();++it) AddDocument(*it);
		SortPostingsLists();
		ComputeAverageDocumentLength();
	}
	void SortPostingsLists() { for (iterator it=begin();it!=end();++it) it->second.sort(); }
	bool GetPostingsList(const WordID& wordid, const_iterator& p) const {
		p = postingsLists_.find(wordid);
		if (p==postingsLists_.end()) return false;
		else return true;
	}
	WordID GetDocID(const Docno docno) const {
		if (docno < n_)
			return documents_.at(docno);
		return -1;
	}
	void Save(ostream& out) const {
		// save n,t,avg_len
		out << n_ << " " << t_ << " " << avg_len_;
		// save docids and lengths
		for(unsigned int i=0;i<n_;++i) {
			out << " " << TD::Convert(documents_[i]) << " " << lengths_[i];
		}
		// save postings lists (sorted by size)
		vector<pair<WordID,const PostingsList*> > tmp;
		for(const_iterator i=postingsLists_.begin();i!=postingsLists_.end();++i) {
			tmp.push_back(pair<WordID,const PostingsList*>(i->first, &(i->second)));
		}
		std::sort(tmp.begin(), tmp.end(), PostingsListsDFCompare());
		assert(tmp.size() == t_);
		for(unsigned int i=0;i<t_;++i) {
			out << "\n" << TD::Convert(tmp[i].first) << " ";
			tmp[i].second->save(out);
		}
	}
	void Load(istream& in) {
		clear();
		// load n,t,avg_len
		in >> n_ >> t_ >> avg_len_;
		assert(n_>0);
		assert(t_>0);
		assert(avg_len_>0);
		// load docids and lengths
		documents_.reserve(n_);
		lengths_.reserve(n_);
		doc2docno_.reserve(n_);
		string docid;
		unsigned short length;
		for(unsigned int i=0;i<n_;++i) {
			in >> docid >> length;
			documents_.push_back(TD::Convert(docid));
			doc2docno_[TD::Convert(docid)] = i;
			//TD::Convert(docid); // add docids to term dictionary
			lengths_.push_back(length);
		}
		assert(documents_.size()==n_);
		assert(lengths_.size()==n_);
		assert(doc2docno_.size()==n_);
		// load postingslists
		postingsLists_.reserve(t_);
		string term;
		for(unsigned int i=0;i<t_;++i) {
			in.ignore(1, '\n');
			in >> term;
			postingsLists_[TD::Convert(term)] = PostingsList(in);
		}
		//cerr << postingsLists_.size() << " " << t_ << endl;
		assert(postingsLists_.size()==t_);
	}

private:
	typedef unordered_map<WordID,PostingsList>::iterator iterator;
	iterator begin() { return postingsLists_.begin(); }
	iterator end() { return postingsLists_.end(); }
	vector<WordID> documents_;
	vector<unsigned short> lengths_;
	unordered_map<WordID,Docno> doc2docno_;
	unordered_map<WordID,PostingsList> postingsLists_; // think of making this a vector<PostingsList> where index is WordID after loading
	unsigned int n_; // number of documents
	unsigned int t_; // number of terms
	double avg_len_; // average length of documents

	struct PostingsListsDFCompare {
		inline bool operator() (const pair<WordID,const PostingsList*>& a, const pair<WordID,const PostingsList*>& b) {
			return (a.second->df() > b.second->df());
		}
	};

};
inline std::ostream& operator<<(std::ostream& out, const Index& idx) {
	out <<"Index[|D|="<<idx.NumberOfDocuments()<<" |T|=" << idx.NumberOfTerms() << " avgL="<<idx.AverageDocumentLength()<<"]";
	//for(Index::const_iterator i=idx.begin();i!=idx.end();++i) out <<" "<<TD::Convert(i->first)<<" "<<i->second<<"\n";
	return out;
}

// load from file mapping. not really faster than regular loading
void LoadIndex(const string& fname, Index& idx) {
	TIMER::timestamp_t t0 = TIMER::get_timestamp();
	boost::interprocess::file_mapping m_file(fname.c_str(), boost::interprocess::read_only);
	boost::interprocess::mapped_region region(m_file, boost::interprocess::read_only);
	void * addr       = region.get_address();
	std::size_t size  = region.get_size();
	char *data = static_cast<char*>(addr);
	std::istringstream in;
	in.rdbuf()->pubsetbuf(data, size);
	idx.Load(in);
	TIMER::timestamp_t t1 = TIMER::get_timestamp();
	float time = (t1-t0) / 1000000.0L;
	cerr << "loaded " << idx <<  " [" << time << "s]\n";
}

}

#endif
