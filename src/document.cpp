#include "document.h"

using namespace CLIR;

Document::Document( string& text, const string& id ) :
	parsed_(false), id_(TD::Convert(id)), len_(0), s_count(0) {
	Document::parse(text);
}

Document::Document( const string& line, const bool is_vector ) :
	parsed_(false), len_(0), s_count(0) {
	stringstream ss(line);
	string id, text;
	ss >> id;
	id_ = TD::Convert(id);
	ss.ignore(1, '\t');
	if (is_vector) { // s_count remains undefined
		ss >> len_;
		ss.ignore(1, '\t');
		getline(ss, text);
		v_ = vutils::read_vector(text);
		if (len_ > 0) parsed_ = true;
	} else { // v_ remains undefined
		getline(ss, text);
		Document::parse(text);
	}
}

Document::Document(const TermVector& v, const string& id, const int len) :
	parsed_(false), id_(TD::Convert(id)), len_(len), s_count(0) {
	v_ = v;
	if (len_ > 0) parsed_ = true;
}

void Document::parse(string& text) {
	if ( strcmp(text.substr(0,3).c_str(),"<d>") != 0 ) {
		// no <d>/<s> markup
		std::vector<WordID> ids;
		map<string, string> map;
		ProcessAndStripSGML(&text, &map);
		TD::ConvertSentence(text, &ids);
		len_ += ids.size();
		text_.push_back(ids);
		sgml_.push_back(map);
		s_count ++;
	} else {
		// pseudo-xml markup
		pugi::xml_document doc;
		pugi::xml_parse_result result = doc.load(text.c_str());
		if (!result) {
			cerr << result.description() << endl;
			cerr << result.status << std::endl;
			cerr << id_ << " parsed with errors!" << endl;
			return;
		}
		pugi::xml_node doc_node = doc.child("d");
		for (pugi::xml_node node = doc_node.first_child(); node; node = node.next_sibling()) {
			if ( strcmp(node.name(), "s") == 0 ) {
				std::vector<WordID> ids;
				map<string, string> map;
				if (pugi::xml_node seg = node.child("seg")) {
					TD::ConvertSentence(seg.text().as_string(), &ids);
					map ["grammar"] = seg.attribute("grammar").as_string();
					map ["id"] = seg.attribute("id").as_string();
				} else {
					TD::ConvertSentence(node.text().as_string(), &ids);
				}
				len_ += ids.size();
				text_.push_back(ids);
				sgml_.push_back(map);
				s_count ++;
			}
		}
	}
	if (s_count > 0 && len_ > 0) parsed_ = true;
}

// computes term frequencies for all sentences
void Document::computeTfVector( const bool sw_filtering ) {
	for (int s = 0 ; s < s_count ; ++s) computeTfVector(s, sw_filtering);
}

// computes term frequencies for given sentence index, stores it in v_
void Document::computeTfVector(const int s, const bool sw_filtering) {
	for(vector<WordID>::const_iterator w = text_[s].begin() ; w != text_[s].end() ; ++w) {
		if (sw::isPunct(*w))
			continue;
		if (sw_filtering && sw::isStopword(*w))
			continue;
		v_[*w] += 1;
	}
}

// computes boolean vectors for all sentences
void Document::computeBooleanVector( const bool sw_filtering ) {
	for (int s = 0 ; s < s_count ; ++s) computeBooleanVector(s, sw_filtering);
}

// computes term frequencies for given sentence index, stores it in v_
void Document::computeBooleanVector(const int s, const bool sw_filtering) {
	for(vector<WordID>::const_iterator w = text_[s].begin() ; w != text_[s].end() ; ++w) {
		if (sw::isPunct(*w))
			continue;
		if (sw_filtering && sw::isStopword(*w))
			continue;
		v_[*w] = 1;
	}
}

// returns a string representation of document as text (can be used with the constructor)
std::string Document::asText(const bool& with_sgml) const {
	stringstream ss;
	ss << TD::Convert(id_) << "\t<d>";
	for (int s = 0 ; s < s_count ; ++s)
		ss << "<s>" << Document::sentence(s, with_sgml) << "</s>";
	ss << "</d>";
	return ss.str();
}
// returns a string representation of document as vector (can be used with the constructor)
std::string Document::asVector() const {
	stringstream ss;
	ss << TD::Convert(id_) << "\t" << len_ << "\t";
	vutils::write_vector(v_,ss);
	return ss.str();
}
