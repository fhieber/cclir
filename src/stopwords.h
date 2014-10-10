/*
 * stopwords.h
 *
 *  Created on: Aug 9, 2012
 *      Author: hieber
 */

#ifndef STOPWORDS_H_
#define STOPWORDS_H_

#include <iostream>
#include <fstream>

#include <boost/unordered_set.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "tdict.h"

typedef boost::unordered_set<WordID> Stopword_Set;

class sw {

	private:
		static Stopword_Set stopwords;

	public:
		/*
		 * loads stopword file and registers stopwords with the
		 * term dictionary (TD)
		 */
		static void loadStopwords(std::string fname);
		static bool isStopword(const WordID& w);
		static int stopwordCount();

		static bool isPunct(const WordID& w);

};

#endif /* STOPWORDS_H_ */
