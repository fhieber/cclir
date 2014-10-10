/*
 * patent.cpp
 *
 *  Created on: Aug 9, 2012
 *      Author: hieber
 */

#include "patent.h"

/*
 * creates a new Patent Document
 */
Patent::Patent(const std::string& xml_filename, const bool& loadAbstract,
		const bool& loadDescription) :
	isParsed(false), fname(xml_filename) , s_count(0), source_tokens(0), target_tokens(0) {

	pugi::xml_document doc;
	pugi::xml_parse_result result = doc.load_file(fname.c_str());

	if (!result) {
		std::cerr << fname << " parsed with errors!" << std::endl;
		std::cerr << "Error description: " << result.description() << std::endl;
		std::cerr << "Error offset: " << result.offset << std::endl;
		return;
	}

	this->_init(doc, loadAbstract, loadDescription);

}


/*
 * load Patent from stringstream
 */
Patent::Patent(std::istream& inp_stream, const bool& loadAbstract,
		const bool& loadDescription) :
	isParsed(false), fname("-") , s_count(0), source_tokens(0), target_tokens(0) {

	pugi::xml_document doc;
	pugi::xml_parse_result result = doc.load(inp_stream);

	if (!result) {
		std::cerr << fname << " parsed with errors!" << std::endl;
		std::cerr << "Error description: " << result.description() << std::endl;
		std::cerr << "Error offset: " << result.offset << std::endl;
		return;
	}

	this->_init(doc, loadAbstract, loadDescription);
}

/*
 * constructor for existing tf target vectors. we do not need the xml stuff
 * anymore.
 */
Patent::Patent(TermVector& tf_vec) : isParsed(false), fname("-"), s_count(0), source_tokens(0), target_tokens(0) {
	this->target_tfvec = tf_vec;
}

Patent::~Patent() {
	text.clear();
	source_tfvec.clear();
	target_tfvec.clear();
}

void Patent::_init(const pugi::xml_document& doc, const bool& loadAbstract, const bool& loadDescription) {
	pugi::xml_node root = doc.child("patent");

	text.clear();
	source_tfvec.clear();
	target_tfvec.clear();

	// walk through nodes below the root node to collect abstracts and descriptions
	for (pugi::xml_node node = root.first_child(); node; node = node.next_sibling()) {

		if ( strcmp(node.name(), "ucid") == 0 )
			ucid = node.text().as_string();
		else if ( strcmp(node.name(), "lang") == 0 )
			lang = node.text().as_string();
		else if ( loadAbstract
			 && strcmp(node.name(), "abstract") == 0 )
			this->loadText(node);
		else if ( loadDescription
			 && strcmp(node.name(), "description") == 0 )
			this->loadText(node);

	}

	s_count = text.size();

	isParsed = true;
}

/*
 * returns a string with some status information regarding the patent xml
 */
std::string const Patent::status() {
	std::stringstream ss;
	ss << "'" << fname << "' parsed. [ucid=" << ucid
	   << ", lang=" << lang
	   << ", abstract ("
	   << text.size()
	   << " sentences)"
	   << ", " << source_tokens << " source tokens"
	   << ", "<< target_tfvec.size() << " target types"
	   << "]";
	return ss.str();
}

/*
 * utility function to print out the whole text
 */
std::string const Patent::getText() {
	stringstream ss;
	ss << "PATENT TEXT:" << endl;
	for (int i = 0 ; i < s_count ; ++i)
		ss << "S" << i << ": " << TD::GetString(text[i]) << endl;
	return ss.str();
}

/*
 * reads in sentences within the <s></s> tags and adds each sentence,
 * converted into a WordID vector.
 */
void Patent::loadText(const pugi::xml_node& node) {
	for (pugi::xml_node s = node.first_child(); s; s = s.next_sibling()) {
		std::vector<WordID> ids;
		TD::ConvertSentence(s.text().as_string(), &ids);
		text.push_back(ids);
	}
}

/*
 * calculates the source term frequency vector for the whole text.
 */
void Patent::calc_source_tfvec(const bool& sw_filtering) {
	for (int s = 0 ; s < s_count ; ++s)
		this->calc_source_tfvec(s, sw_filtering);
}

/*
 * calculates the source term frequency vector for sentence index s.
 * Filters punctuation!
 */
void Patent::calc_source_tfvec(const int& s, const bool& sw_filtering) {
	for(std::vector<WordID>::const_iterator w = text[s].begin() ; w != text[s].end() ; ++w) {
		if (sw::isPunct(*w))
			continue;
		if (sw_filtering && sw::isStopword(*w))
			continue;
		source_tfvec[*w] += 1;
		source_tokens++;
	}
}

/*
 * creates the target side tf vector for the patent by calculating:
 * tf_t = sum(tf_s * s2t[s][t])
 * stopword filtering is done at the target side:
 * if a target term e is a stopword it is not added to the target
 * vector.
 */
void Patent::calc_target_tfvec(NbestTTable& s2t) {

	for(TermVector::iterator it = source_tfvec.begin(); it != source_tfvec.end(); ++it) {

		const WordID f = it->first;
		const prob_t tf = it->second;

		// get E_Options for current f
		TransOptions* e_options = s2t.getOptions(f);
		if (!e_options) // check if NULL
			continue;

		TransOptions::const_iterator e_opt;
		for(e_opt = (*e_options).begin(); e_opt != (*e_options).end(); ++e_opt) {
			prob_t w = e_opt->second;
			WordID e = e_opt->first;
			if (sw::isStopword(e) || sw::isPunct(e))
				continue;
			target_tfvec[e] += tf * w;
		}

	}

}

/*
 * creates the target side tf vector for the patent using the baseline algorithm
 * from Ture et al. SIGIR'11 (using a lexical translation table)
 */
void Patent::calc_target_tfvec(LexicalTTable& s2t) {

	for(TermVector::iterator it = source_tfvec.begin(); it != source_tfvec.end(); ++it) {

			const WordID f = it->first;
			const prob_t tf = it->second;

			// get Target_Options for current f
			LexTransOptions* t_options = s2t.getOptions(f);
			if (!t_options) // check if NULL
				continue;

			LexTransOptions::const_iterator e_opt;
			for(e_opt = (*t_options).begin(); e_opt != (*t_options).end(); ++e_opt) {
				prob_t w = e_opt->second;
				WordID e = e_opt->first;
				if (sw::isStopword(e) || sw::isPunct(e))
					continue;
				target_tfvec[e] += tf * w;
			}

		}

}

/*
 * calculates the weighted term vector according to the given scoring method
 */
void Patent::calc_target_wvec(DfTable& dft, Scorer& scorer) {

	target_wvec.clear();
	//const int len = target_tfvec.size();
	const double len = vutils::vector_values_sum(target_tfvec).as_float();
	for(TermVector::iterator it = target_tfvec.begin(); it != target_tfvec.end(); ++it) {
		if (dft[it->first] == 0.0)
			cerr << ucid << ": suspicious df for '" << TD::Convert(it->first)
				 << "': " << dft.mTable[it->first]
				 << ". Skipping this term!" << endl;
		target_wvec[it->first] = scorer.computeWeight(it->second,dft[it->first],len);
	}
	// normalize weighted term vector
	vutils::normalize_vector(target_wvec);

}

/*
 * prints source term frequency vector
 */
std::string const Patent::print_source_tfvec() {
	return vutils::print_vector(source_tfvec);
}

/*
 * prints target term frequency vector
 */
std::string const Patent::print_target_tfvec() {
	return vutils::print_vector(target_tfvec);
}

/*
 * prints target weighted term vector
 */
std::string const Patent::print_target_wvec() {
	return vutils::print_vector(target_wvec);
}

