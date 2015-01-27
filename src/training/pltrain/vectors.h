/*
 * vectors.h
 *
 *  Created on: May 22, 2013
 */

#ifndef VECTORS_H_
#define VECTORS_H_

#include <vector>
#include "filelib.h"
#include "sparse_vector.h"
#include "weights.h"

typedef FastSparseVector<double> FeatureVector;

typedef FastSparseVector<weight_t> WeightVector;
typedef std::vector<weight_t> DenseWeightVector;
void static writeWeightsToFile(const std::string fname, const WeightVector w) {
	WriteFile out(fname);
	for (WeightVector::const_iterator i=w.begin(); i!=w.end(); ++i)
		*out << FD::Convert(i->first) << " " << i->second << "\n";
}

typedef FastSparseVector<double> Gradient;

#endif /* VECTORS_H_ */
