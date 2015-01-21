/*
 * stopwords.cpp
 *
 *  Created on: Aug 10, 2012
 *      Author: hieber
 */

#include "stopwords.h"

Stopword_Set sw::stopwords;

void sw::loadStopwords(std::string fname) {
	std::ifstream sw_file;
	sw_file.open(fname.c_str());
	if (sw_file.is_open()) {
		std::string w;
		while(sw_file.good()) {
			getline(sw_file, w);
			sw::stopwords.insert(TD::Convert(w));
		}
		sw_file.close();
	} else {
		std::cerr << "could not load stopword file '" << fname << "'" << std::endl;
	}

}

bool sw::isStopword(const WordID& w) {
	Stopword_Set::const_iterator got = sw::stopwords.find(w);
	if(got==sw::stopwords.end())
		return false;
	return true;
}


int sw::stopwordCount() {
	return sw::stopwords.size();
}

bool sw::isPunct(const WordID& w) {
	// also regards terms as punctuation terms that end with ":".
	// This will cause problems in the vector format
	if ( boost::algorithm::ends_with(TD::Convert(w), ":") )
		return true;
	if ( boost::algorithm::all(TD::Convert(w), boost::is_punct()) )
		return true;
	if ( boost::algorithm::starts_with(TD::Convert(w), "&") &&
		 boost::algorithm::ends_with  (TD::Convert(w), ";") )
		return true;
	return false;
}


