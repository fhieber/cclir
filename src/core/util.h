#ifndef UTIL_H_
#define UTIL_H_

#include <sstream>
#include <fstream>
#include <sys/types.h>
#include <dirent.h>
#include <sys/time.h>

#include <boost/algorithm/string.hpp>

// cdec imports
#include "prob.h"
#include "sparse_vector.h"
#include "stringlib.h"
#include "tdict.h"


typedef SparseVector<prob_t> TermVector;

struct vutils {

	static std::string print_vector(const TermVector& v);
	static void write_vector(const TermVector& v, std::ostream& out);
	static TermVector read_vector(const std::string& in);
	static void normalize_vector(TermVector& v);
	static prob_t vector_values_sum(const TermVector& v);
	static int common_keys(const TermVector& a, const TermVector& b);

	static void read_dir(const std::string& s, std::vector<std::string>& files);
	static bool endsWith (std::string const &fullString, std::string const &ending);

};



namespace TIMER {
typedef unsigned long long timestamp_t;
inline timestamp_t get_timestamp () {
	struct timeval now;
	gettimeofday (&now, NULL);
	return now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

}

#endif /* UTIL_H_ */
