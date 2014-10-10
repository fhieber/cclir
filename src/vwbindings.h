#ifndef _VWBINDINGS_H_
#define _VWBINDINGS_H_

#include <algorithm>
#include "../vowpalwabbit/parser.h"
#include "../vowpalwabbit/vw.h"
#include "feature_vector.h"

namespace VWBindings {

example* build_vwexample(vw& model, const FeatureVector& x) {
	vector < feature > features;
	for (FeatureVector::const_iterator it = x.begin(); it != x.end(); ++it) {
		if (it->second != 0) {
			feature f = { (float) it->second, (uint32_t) it->first };
			features.push_back(f);
		}
	}
	vector< VW::feature_space > fs = {VW::feature_space('f', features)};
	return VW::import_example(model, fs);
}

int num_nonzero(vw& model) {
	int n = 0;
	for (unsigned x = 0; x < VW::num_weights(model); ++x) {
		if (VW::get_weight(model, x, 0) != 0)
			n++;
	}
	return n;
}
string printNonZeroWeights(vw& model) {
	ostringstream ss;
	for (unsigned x = 0; x < VW::num_weights(model); ++x) {
		float val = VW::get_weight(model, x, 0);
		if (val != 0)
			ss << FD::Convert(x) << "(" << x << ")=" << val << " ";
	}
	return ss.str();
}

void set_vw_weights(vw& model, const WeightVector& w) {
	for (WeightVector::const_iterator it = w.begin(); it != w.end(); ++it)
		VW::set_weight(model, (uint32_t) it->first, 0, (float) it->second);
}
void set_vw_weights(vw& model, const DenseWeightVector& w) {
	for (unsigned i = 0; i < w.size(); ++i)
		VW::set_weight(model, (uint32_t) i, 0, (float) w[i]);
}

inline bool close_enough(double a, double b, double epsilon = 1e-5) {
	using std::fabs;
	double diff = fabs(a - b);
	return diff <= epsilon * fabs(a) || diff <= epsilon * fabs(b);
}

/*
 * to get the current VW model back to cdec for decoding
 */
const string IR_PREFIX = "IR:";
inline bool is_ir_feature(const int x) {
	string fname = FD::Convert(x);
	if (fname.size() < IR_PREFIX.size())
		return false;
	return (std::mismatch(IR_PREFIX.begin(), IR_PREFIX.end(), fname.begin()).first
			== IR_PREFIX.end());
	//return res.first == foo.end();
}
void get_vw_weights(vw& model, WeightVector& w_smt, WeightVector& w_ir) {
	for (unsigned x = 0; x < FD::NumFeats(); ++x) {
		float v = VW::get_weight(model, x, 0);
		if (v != 0) { //what about regularization from VW?
			if (is_ir_feature(x))
				w_ir.set_value(x, v);
			else
				w_smt.set_value(x, v);
		}
	}
	//cerr << "|W_SMT|="<<w_smt.size()<< " |W_IR|="<<w_ir.size()<<endl;
}

void get_vw_weights2(vw& model, WeightVector& w_smt, WeightVector& w_ir) {
	for (unsigned x = 0; x < FD::NumFeats(); ++x) {
		const float v = VW::get_weight(model, x, 0);
		WeightVector::iterator smt_it = w_smt.find(x);
		if (smt_it != w_smt.end()) { // known smt feature
			smt_it->second = v;
			continue;
		}
		WeightVector::iterator ir_it = w_ir.find(x);
		if (ir_it != w_ir.end()) { // known ir feature
			ir_it->second = v;
			continue;
		}
		if (v != 0) {
			// classify feature:
			if (is_ir_feature(x))
				w_ir.set_value(x, v);
			else
				w_smt.set_value(x, v);
		}
	}
}
/*
 * slow but safe
 */
void get_vw_weights3(vw& model, WeightVector& w_smt, WeightVector& w_ir) {
	WeightVector w_ir_new;
	WeightVector w_smt_new;
	for (unsigned x = 0; x < FD::NumFeats(); ++x) {
		float v = VW::get_weight(model, x, 0);
		if (is_ir_feature(x))
			w_ir_new.set_value(x, v);
		else
			w_smt_new.set_value(x, v);
	}
	w_ir = w_ir_new.erase_zeros();
	w_smt = w_smt_new.erase_zeros();
	cerr << "|W_SMT|="<<w_smt.size()<< " |W_IR|="<<w_ir.size()<<endl;
}
/*
 * gets only updated feature weights back from VW
 */
void get_vw_weights(vw& model, const FeatureVector& gradient, WeightVector& w_smt, WeightVector& w_ir, const bool debug=false) {
	unsigned u=0;
	if (debug) cerr << " Gradient: " << gradient << endl; 
	for (FeatureVector::const_iterator i=gradient.begin();i!=gradient.end();++i) {
		if (!i->second) continue;
		if (is_ir_feature(i->first)) {
			w_ir.set_value(i->first, VW::get_weight(model, i->first, 0));
			if (debug) cerr << FD::Convert(i->first) << "=" << w_ir.get(i->first) << " ";
		} else {
			w_smt.set_value(i->first, VW::get_weight(model, i->first, 0));
			if (debug) cerr << FD::Convert(i->first) << "=" << w_smt.get(i->first) << " ";
		}
		u++;
	}
	if (debug) cerr << endl << u << " features updated!\n";
}
void get_vw_weights(vw& model, DenseWeightVector& w) {
	assert(w.size() == FD::NumFeats());
	for (unsigned x = 0; x < FD::NumFeats(); ++x)
		w[x] = VW::get_weight(model, x, 0);
}

double get_vw_weight(vw& model, unsigned x) {
	return (double)VW::get_weight(model,x,0);
}

void compare_vw_weights(vw& model, const WeightVector& w_smt, const WeightVector& w_ir) {
	cerr << "VW WEIGHTS:" << endl;
	for (WeightVector::const_iterator it=w_smt.begin();it!=w_smt.end();++it) {
		double vww = get_vw_weight(model, it->first);
		if (vww == 0 && it->second ==0)
			continue;
		cerr << FD::Convert(it->first) << " vw=" << get_vw_weight(model, it->first) << " me=" << it->second << (!close_enough(vww,it->second) ? " *" : "" ) << endl;
	}
	for (WeightVector::const_iterator it=w_ir.begin();it!=w_ir.end();++it) {
		double vww = get_vw_weight(model, it->first);
		if (vww == 0 && it->second ==0)
			continue;
		cerr << FD::Convert(it->first) << " vw=" << get_vw_weight(model, it->first) << " me=" << it->second << (!close_enough(vww,it->second) ? " *" : "" ) << endl;
	}
	cerr << endl;
}

}

/* EXAMPLE
 int main(int argc, char** argv) {

 FeatureVector X;
 X[FD::Convert("LanguageModel")] = -25.3;
 X[FD::Convert("Glue")] = 1;
 X[FD::Convert("IR:action")] = 0.423;
 // VW stuff
 vw* model = VW::initialize("-l 1.0 --sgd --noconstant -a -b 10 --readable_model tmp.vw");
 cerr << endl << "EXAMPLE1" << endl;
 cerr << "W: " << printNonZeroWeights(*model) << endl;
 example* myexample = build_vwexample(*model, X);
 VW::add_label(myexample, 2.0);
 model->learn(myexample);
 VW::finish_example(*model, myexample);
 cerr << "W: " << printNonZeroWeights(*model) << endl;

 WeightVector w_smt,w_ir;
 get_vw_weights(*model, w_smt, w_ir);
 cerr << "W_SMT " << w_smt << endl;
 cerr << "W_IR  " << w_ir << endl;
 w_ir[FD::Convert("IR:action")] = -66;
 set_vw_weights(*model, w_ir);
 cerr << "W: " << printNonZeroWeights(*model) << endl;
 cerr << endl << "EXAMPLE2" << endl;
 FeatureVector X2;
 X2[FD::Convert("LanguageModel")] = 100;
 X2[FD::Convert("Glue")] = 100;
 X2[FD::Convert("IR:action")] = 100;
 example* e2 = build_vwexample(*model, X2);
 VW::add_label(e2, 10);
 model->learn(e2);
 VW::finish_example(*model, e2);
 cerr << "W: " << printNonZeroWeights(*model) << endl;

 VW::finish(*model); // after this *model points nowhere
 }
 */
#endif
