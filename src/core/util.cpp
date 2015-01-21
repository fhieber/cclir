/*
 * util.cpp
 *
 *  Created on: Aug 10, 2012
 */

#include "util.h"

/*
 * utility function to print a sparse vector
 */
std::string vutils::print_vector(const TermVector& v) {
	std::stringstream ss;
	ss << "< ";
	for(TermVector::const_iterator it = v.begin(); it != v.end(); ++it)
		ss << TD::Convert(it->first) << ":" << it->second.as_float() << "(" << it->second << ") ";
	ss << ">";
	return ss.str();
}

/*
 * utility function to write a sparse vector to a file.
 * <sep> defines the separator between the vector entries. use "\n" for easy reading :)
 * Format: <feature> <value> <feature> <value>...
 */
void vutils::write_vector(const TermVector& v, std::ostream& out) {
	for(TermVector::const_iterator it = v.begin(); it != v.end(); ++it)
		out << TD::Convert(it->first) << " " << it->second << " ";
}

/*
 * utility function to read a sparse vector (element) from a string.
 * reverse function for vutils::write_vector().
 * Assumes ws as separator char.
 */
TermVector vutils::read_vector(const std::string& s) {
	TermVector v;
	std::stringstream in(Trim(s, " \t"));
	std::string key;
	double val;
	while(in.good()) {
		in >> key;
		in >> val;
		v[ TD::Convert(key) ] = prob_t(val,false);
	}
	return v;
}

void vutils::normalize_vector(TermVector& v) {
	v /= v.l2norm_sq().root(2.0);
}

prob_t vutils::vector_values_sum(const TermVector& v) {
	prob_t s = 0;
	for(TermVector::const_iterator it = v.begin(); it != v.end(); ++it) {
		s += it->second;
	}
	return s;
}

int common_keys(const TermVector& a, const TermVector& b) {
	int c = 0;
	for (TermVector::const_iterator a_it=a.begin(); a_it!=a.end(); ++a_it) {
		if (b.find(a_it->first) != b.end())
			c++;
	}
	return c;
}

/*
 * reads the contents of a directory and writes pathnames to the given vector
 */
void vutils::read_dir(const std::string& s, std::vector<std::string>& files) {
	DIR *dp;
	struct dirent *dirp;
	if((dp = opendir(s.c_str())) == NULL)
		std::cerr << "Error opening " << s << std::endl;
	while ((dirp = readdir(dp)) != NULL) {
		if (std::string(dirp->d_name) == "."
				|| std::string(dirp->d_name) == "..")
			continue;
		std::stringstream ss;
		ss << s << std::string(dirp->d_name);
		files.push_back(ss.str());
	}
	closedir(dp);
}

/*
 * string utility function. returns true if fullString ends with ending.
 */
bool vutils::endsWith (std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}


