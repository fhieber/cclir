#ifndef CLIR_H_
#define CLIR_H_

#include <iomanip>
#include <boost/program_options.hpp>
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include "filelib.h"
#include "util.h"

#include "document.h"
#include "query.h"
#include "dftable.h"
#include "index.h"
#include "scoring.h"


using namespace std;
namespace po = boost::program_options;

namespace CLIR {

// loads Query instances from file
void loadQueries(const string& fname, vector<Query>& queries, bool isPSQ=false, bool isText=false) {
	TIMER::timestamp_t t0 = TIMER::get_timestamp();
	ReadFile O(fname);
	string raw;
	queries.clear();
	while(getline(*O, raw)) {
		if (raw.size() == 0) continue;
		if (isText) { 
			CLIR::Document d(raw);
			if (!d.parsed_) cerr << "Warning: Query is malformed: '" << raw << "'\n";
			d.computeTfVector(false);
			queries.emplace_back(d.id_, d.v_); // construct query in place
		} else {
			queries.emplace_back(isPSQ, raw); // construct query in place	
		}
		if (queries.back().size() == 0) cerr << "WARNING: Query (id=" << queries.back().idstr() << ") empty!\n";
	}
	TIMER::timestamp_t t1 = TIMER::get_timestamp();
	float time = (t1-t0) / 1000000.0L;
	cerr << "loaded " << queries.size() << (isText ? " text" : "") << " queries. [" << time << "s]\n";
}

double loadDocuments(const string& fname, vector<Document>& documents) {
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
		documents.emplace_back(vutils::read_vector(raw), docid, len); // construct document in place
		//documents.push_back( Document(vutils::read_vector(raw), docid, len) );
		sumDocLen += len;
	}
	unsigned n = documents.size();
	TIMER::timestamp_t t1 = TIMER::get_timestamp();
	float time = (t1-t0) / 1000000.0L;
	cerr << "loaded " << n << " documents. (average length=" << sumDocLen / (double) n << ") [" << time << "s]\n";
	return sumDocLen / (double) n;
}

/*
 * loads a qrel file in trec format
 */
typedef unordered_map<WordID,unordered_map<WordID,unsigned> > Qrels;
void loadQrels(const string& fname, Qrels& rels) {
	// JP-2003000276-A 0       US-6605696-B1   1
	rels.clear();
	ReadFile rf(fname);
	istream *in = rf.stream();
	assert(*in);
	string qidstr , docidstr, dummy;
	unsigned short rl;
	WordID qid, docid;
	Qrels::iterator rit;
	while(*in >> qidstr) {
		in->ignore(1, '\t');
		*in >> dummy;
		in->ignore(1, '\t');
		*in >> docidstr;
		in->ignore(1, '\t');
		*in >> rl;
		if(qidstr.empty() || docidstr.empty()) continue;
		qid = TD::Convert(qidstr);
		docid = TD::Convert(docidstr);
		rit = rels.find(qid);
		if (rit==rels.end())
			rit = rels.insert(make_pair(qid, unordered_map<WordID,unsigned>())).first;
		rit->second.insert(make_pair(docid,rl));
	}
}

inline Scorer* setupScorer(const string& metric, const unsigned int N, const double avg_len, const DfTable* dft) {
	Scorer* ranker = NULL;
	if (metric == "bm25")
		ranker =  new BM25Scorer(N, avg_len, dft);
	else if (metric == "classicbm25")
		ranker =  new ClassicBM25Scorer(N, avg_len, dft);
	else if (metric == "negbm25")
		ranker =  new NEGBM25Scorer(N, avg_len, dft);
	else if (metric == "classicbm25tf")
		ranker = new ClassicBM25TFScorer(N, avg_len, dft);
	else if (metric == "tfidf")
		ranker = new TFIDFScorer(N, avg_len, dft);
	else if (metric == "stfidf")
		ranker = new STFIDFScorer(N, avg_len, dft);
	else {
		cerr << "LOG::unknown ranking_function!\n";
		abort();
	}
	return ranker;
}

inline IndexScorer* setupIndexScorer(const string& metric, const Index::Index* idx) {
	IndexScorer* ranker = NULL;
	if (metric == "bm25")
		ranker =  new IndexBM25Scorer(idx);
	else if (metric == "classicbm25")
		ranker = new IndexClassicBM25Scorer(idx);
	else if (metric == "tfidf")
		ranker = new IndexTFIDFScorer(idx);
	else if (metric == "stfidf")
		ranker = new IndexSTFIDFScorer(idx);
	else {
		cerr << "LOG::unknown ranking_function!\n";
		abort();
	}
	return ranker;
}

inline void writeResult(ostream& out, const string& query_id, const vector<Score>& result, const string& run_id, const bool show_empty=true, const string separator="\n") {
	out.precision(10);
	if (show_empty && result.empty()) {
		out << query_id << "\tQ0\t-1\t1\t0.0\t" << run_id << "\n";
	} else {
		int r = 1;
		for (vector<Score>::const_reverse_iterator rit=result.rbegin();rit!=result.rend();++rit) {
			out << query_id // query id
				<< "\tQ0\t"
				<< TD::Convert(rit->mD) << "\t" // docid
				<< r << "\t" // rank
				<< setprecision(10) << rit->mS.as_float() << "\t" // score
				<< run_id << separator; // run id
			++r;
		}
		assert(r == result.size() + 1);
	}
}

inline void writeResult(ostream& out, const Query& query, const vector<Score>& result, const string& run_id, const bool show_empty=true) {
	writeResult(out, query.idstr(), result, run_id, show_empty);
}

}

#endif /* CLIR_H_ */
