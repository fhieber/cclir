#ifndef DOCUMENT_H_
#define DOCUMENT_H_

#include <vector>
#include "pugixml.hpp"
#include "stringlib.h"
#include "stopwords.h"
#include "tdict.h"
#include "util.h"

using namespace std;

namespace CLIR {
class Document {
public:
	// construct document from input text (possibly with sgml markup)
	Document( string& text, const string& id);
	// construct document from input line.
	Document( const string& line, const bool is_vector=false );
	// construct document from TermVector
	Document(const TermVector& vec, const string& id, const int len);
	
	void computeTfVector(const bool sw_filtering);
	void computeTfVector(const int s, const bool sw_filtering);
	void computeBooleanVector(const bool sw_filtering);
	void computeBooleanVector(const int s, const bool sw_filtering);

	bool parsed_;
	WordID id_;
	vector<vector<WordID> > text_; // sentences as WordID vectors
	vector<map<string,string> > sgml_;
	int len_;
	int s_count; // sentence count
	
	TermVector v_;

	void parse( string& text );

	std::string sentence(const int& i, const bool& with_sgml=false) const {
		if (with_sgml) {
			map<string,string> map = sgml_.at(i);
			if (map.empty())
				return TD::GetString( text_.at(i) );
			stringstream ss; //data/c.10k.89.q.grammar.gz" id="14">")
			ss << "<seg grammar=\"" << map["grammar"] << "\" id=\"" << map["id"] << "\">";
			ss << TD::GetString( text_.at(i) ) << "</seg>";
			return ss.str();
		} else
			return TD::GetString( text_.at(i) ) ;
	}

	std::string asText(const bool& with_sgml=false) const;
	std::string asVector() const;

};
}
#endif
