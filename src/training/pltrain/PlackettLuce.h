/*
 * PlackettLuce.h
 *
 *  Created on: May 22, 2013
 */

#ifndef PLACKETTLUCE_H_
#define PLACKETTLUCE_H_

#include <iostream>
#include <vector>
#include <math.h>
#include <numeric>
#include <float.h>
#include "instance.h"

namespace PlackettLuce {

typedef FastSparseVector<weight_t> WeightVector;

static inline double myexp(double x) {
#define A0 (1.0)
#define A1 (0.125)
#define A2 (0.0078125)
#define A3 (0.00032552083)
#define A4 (1.017256e-5)
	bool r = false;
	if (x < 0) {
		x = -x;
		r = true;
	}
	double y;
	y = A0+x*(A1+x*(A2+x*(A3+x*A4)));
	y *= y;
	y *= y;
	y *= y;
	if (r) y = 1./y;
	return y;
#undef A0
#undef A1
#undef A2
#undef A3
#undef A4
}

static inline double logsumexp(double x, double y, bool flg) {
	if (flg)
		return y; // init mode
	if (x == y)
		return x + 0.69314718055; // log(2)
	double vmin = x > y ? y: x;
	double vmax = x > y ? x: y;
	return vmax + log(exp(vmin-vmax)+1.0);
}

// L(x;w) = log P(y|x;w) = sum(log(v_i) - log(sum(v_k)))
static double pl_likelihood(const Instance& x, const WeightVector& w) {
	double likelihood = 0.0;
	assert(x.ir_sorted); // derivations are sorted by label / ir scores
	vector<double> e; // enumerator of PL factors
	for (int did=0; did!=x.size;++did) {
		const FeatureVector& fv = x.derivations[did].fvec;
		double v = fv.dot(w); // dot product w*fv
		e.push_back(v);
	}
	// partition
	reverse(e.begin(), e.end());
	double z = 0.0;
	vector<double> zz;
	vector<double>::iterator eit = e.begin();
	for (; eit != e.end(); ++eit) {
		z = logsumexp(z, *eit, (eit == e.begin() ) );
		zz.push_back(z);
	}
	reverse(zz.begin(), zz.end());
	reverse(e.begin(), e.end());
	for (int k=0; k < x.size; ++k)
		likelihood += e[k] - zz[k];

	return likelihood;
}

// L(x;w) = log P(y|x;D) = sum(log(v_i) - log(sum(v_k)))
static double pl_likelihood(const Instance& x) {
	double likelihood = 0.0;
	assert(x.ir_sorted); // derivations are sorted by label / ir scores
	vector<double> e; // enumerator of PL factors
	for (int did=0; did!=x.size;++did) {
		double v = x.derivations[did].mtscore;
		e.push_back(v);
	}
	// partition
	reverse(e.begin(), e.end());
	double z = 0.0;
	vector<double> zz;
	vector<double>::iterator eit = e.begin();
	for (; eit != e.end(); ++eit) {
		z = logsumexp(z, *eit, (eit == e.begin() ) );
		zz.push_back(z);
	}
	reverse(zz.begin(), zz.end());
	reverse(e.begin(), e.end());
	for (int k=0; k < x.size; ++k)
		likelihood += e[k] - zz[k];

	return likelihood;
}

// L(X;w) = sum( L(x;w))
static double likelihood(const std::vector<Instance>& X, const WeightVector& w) {
	double l = 0.0;
	for (vector<Instance>::const_iterator xit=X.begin();xit!=X.end();++xit)
		l += PlackettLuce::pl_likelihood(*xit, w);
	return l;
}

// L(X;D) = sum( L(x;D))
static double likelihood(const std::vector<Instance>& X) {
	double l = 0.0;
	for (vector<Instance>::const_iterator xit=X.begin();xit!=X.end();++xit)
		l += PlackettLuce::pl_likelihood(*xit);
	return l;
}

// O(2n*J)
static Gradient calculate_gradient(const Instance& x, const WeightVector& w) {
	assert(x.size > 1);
	assert(x.ir_sorted); // derivations are sorted by label / ir scores
	
	Gradient gradient;
	
	size_t n = x.size; // number of chunks
	
	vector<double> dots(n); // precalculated dot products
	double dot_max = -DBL_MAX; // maximum dot product
	for ( size_t i = 0 ; i != n ; ++i ) {
		for ( WeightVector::const_iterator wit=w.begin(); wit!=w.end(); ++wit)
			gradient[wit->first] += x.derivations[i].fvec.get(wit->first);
		dots[i] = x.derivations[i].fvec.dot(w);
		if (dots[i] > dot_max)
			dot_max = dots[i];
	}
	
	///////////////////////////////////////
	// do fancy summing
	///////////////////////////////////////	
	FastSparseVector<double> frac_sums; // sum of fractions for each feature (second term)
	
	map<int,vector<double> > tops; // numerators (depend on j)
	vector<double> bot(n, 0.0); // denominator (independent of j)
	
	// set last value for denominator (i=k=n)
	bot[n-1] = exp(dots[n-1] - dot_max);
	
	// set last values for enumerators (i=k=n)
	for ( Gradient::iterator fit = gradient.begin(); fit != gradient.end(); ++fit) {
		vector<double> top(n, 0.0);
		top[n-1] = exp(dots[n-1] - dot_max) * x.derivations[n-1].fvec.value( fit->first );
		tops[ fit->first ] = top;
		frac_sums[ fit->first ] = top[n-1] / bot[n-1];
	}
	
	// start from penultimate position
	for( short i = n-2 ; i >= 0; --i ) {
		double exp_term = exp(dots[i] - dot_max);
		// sum from i to end
		bot[i] = ( exp_term ) + bot[i+1]; // exp(wx-wx_max) + rest
		for ( Gradient::iterator fit = gradient.begin(); fit != gradient.end(); ++fit) {
			// exp(wx-wx_max)*xij + rest
			tops[ fit->first ][i] = ( exp_term * x.derivations[i].fvec.value(fit->first) ) + tops[fit->first][i+1];
			frac_sums[ fit->first ] += tops[ fit->first ][i] / bot[i];
		}
	}
	
	gradient -= frac_sums;
	assert(gradient.size() == w.size());
	return gradient;
}


// O(2n)
static double gradient(const Instance& x, const WeightVector& w, int j) {
	assert(x.size > 1);
	assert(x.ir_sorted); // derivations are sorted by label / ir scores
	
	size_t n = x.size;
	
	double xj_sum = 0.0; // sum of x_ij (first term)
	vector<double> dots(n); // precalculated dot products
	double dot_max = -DBL_MAX; // maximum dot product
	for ( size_t i = 0 ; i != n ; ++i ) {
		xj_sum += x.derivations[i].fvec.value(j);
		dots[i] = x.derivations[i].fvec.dot(w);
		if (dots[i] > dot_max)
			dot_max = dots[i];
	}
	
	///////////////////////////////////////
	// do fancy summing
	///////////////////////////////////////	
	vector<double> top(n, 0.0); // numerator
	vector<double> bot(n, 0.0); // denominator
	// set values for i=k=n
	top[n-1] = exp(dots[n-1] - dot_max) * x.derivations[n-1].fvec.value(j);
	bot[n-1] = exp(dots[n-1] - dot_max);
	double frac_sum = top[n-1] / bot[n-1]; // sum of fractions
	
	// start from penultimate position
	for( short i = n-2 ; i >= 0; --i ) {
		double exp_term = exp(dots[i] - dot_max);
		// sum from i to end
		top[i] = ( exp_term * x.derivations[i].fvec.value(j) ) + top[i+1]; // exp(wx-wx_max)*xij + rest
		bot[i] = ( exp_term ) + bot[i+1];	 						 	   // exp(wx-wx_max) + rest
		frac_sum += top[i] / bot[i];	
	}
	
	double grad = xj_sum - frac_sum;
	return grad;
}
	
// O(n^2)
static double gradient_naive(const Instance& x, const WeightVector& w, int j) {
	assert(x.size > 1);
	assert(x.ir_sorted); // derivations are sorted by label / ir scores
	// sum of x_ij ( first term)
	double sum_x_ij = 0.0;
	for (int i=0; i!=x.size;++i) {
		sum_x_ij += x.derivations[i].fvec.value(j);
	}

	
	// second term
	double sum = 0.0;
	for (int i=0; i!=x.size;++i) {
	
		double dot_max = -DBL_MAX;
		for (int k=i; k!=x.size;++k) { // get max dot product
			double wx_k = x.derivations[k].fvec.dot(w);
			if (wx_k > dot_max) dot_max = wx_k;
		}
		
		double top = 0.0;
		double bot = 0.0;
		double top2 = 0.0;
		double bot2 = 0.0;
		for (int k=i; k!=x.size;++k) {
			double x_kj = x.derivations[k].fvec.value(j);
			double wx_k = x.derivations[k].fvec.dot(w);
			double exp_term = exp(wx_k - dot_max);
			//cerr << "x_kj=" << x_kj << endl;
			//cerr << "expterm="<< exp_term << endl;
			top += exp_term*x_kj;
			bot += exp_term;
			top2 += exp(wx_k)*x_kj;
			bot2 += exp(wx_k);
		}
		sum += top/bot;
		//cerr << i << " top=" << top << ", bot="<< bot << ", summand="<< top/bot << " ||| " << "top2=" << top2 << ", bot2=" << bot2 << ", summand2="<<top2/bot2 <<endl;
		//assert(top/bot == top2/bot2);
	}

	double grad_j =  sum_x_ij - sum;
	return grad_j;
}

static Gradient calculate_gradient_naive(const Instance& x, const WeightVector& w) {
	Gradient g;
	for ( WeightVector::const_iterator wit=w.begin(); wit!=w.end(); ++wit) {
		g[wit->first] = gradient_naive(x, w, wit->first);
	}
	return g;
}

}


#endif /* PLACKETTLUCE_H_ */
