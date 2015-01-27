/*
 * sgd.h
 *
 *  Created on: May 24, 2013
 */

#ifndef SGD_H_
#define SGD_H_

#include <iostream>
#include <fstream>
#include "filelib.h"
#include "stringlib.h"
#include "weights.h"

#include <boost/program_options.hpp>

#include "instance.h"
#include "PlackettLuce.h"

using namespace std;
namespace po = boost::program_options;


/*
 * loads and parses commandline parameters
 */
bool init_params(int argc, char** argv, po::variables_map* cfg) {
	stringstream ss;
	ss << "\nruns Stochastic Gradient Descent for the Plackett-Luce model";
	po::options_description cl(ss.str());
	cl.add_options()
			("input,i",				po::value<string>(),					"Labeled instances as generated from set-gold-permutation")
			("weights,w",			po::value<string>(),					"current weight vector (same as for generate-instances)")
			("batch-size,b",		po::value<int>()->default_value(0),		"batch size for updates")
			("output,o",			po::value<string>(),					"output weight vector file")
			("iterations,t",		po::value<int>()->default_value(1000),	"number of iterations for boosting")
			("epsilon,e",			po::value<double>()->default_value(0.00001), "stopping criterion: delta change of likelihood < epsilon")
			("learning-rate,l",		po::value<double>()->default_value(0.000001),	"learning rate / step size")
			("verbose,v",			po::value<bool>()->zero_tokens(),		"verbose output to STDERR.")
			("help,h",				po::value<bool>()->zero_tokens(),		"print this help message.");

	po::store(parse_command_line(argc, argv, cl), *cfg);
	po::notify(*cfg);

	if (cfg->count("help")) {
		cerr << cl << endl;
		return false;
	}

	if (!cfg->count("input") || !cfg->count("weights")) {
		cerr << cl << endl;
		return false;
	}
	
	return true;
}

namespace SGD {

class Learner {
public:
	Learner(const vector<Instance>& X, WeightVector w,
			const int T, const double eps, double eta,
			const bool verbose, int batch_size,
			ostream* l_out)
		: X_(X), m_(X.size()), T_(T), eps_(eps), eta_(eta), w_(w), u_(0),
		  ll_(0.0), t_(0), v_(verbose), lo_(l_out) {
		
		bs_ = (batch_size >= m_ || batch_size == 0) ? m_ - 1 : batch_size;

		cerr << "\nStochastic Gradient Descent"
			 << "\ninstances:\t" << m_
			 << "\niterations:\t" << T_
			 << "\nepsilon:\t" << eps_
			 << "\nlearning rate:\t" << eta_
			 << "\nbatch size:\t" << bs_
			 << "\n\n";
	}

	void learn() {

		cerr << "running ...\n";

		double prev_ll = PlackettLuce::likelihood( X_, w_ ); // previous log likelihood
		double change;
		g_.clear();

		cerr << "Initial Log Likelihood: " << prev_ll << "\n";

		for (t_ = 1; t_ <= T_ ; ++t_) {

			if (v_) cerr << "T" << t_ << endl;
			
			for ( short i=0 ; i<m_ ; ++i ) {
				if (X_[i].size<2) { continue; }
				g_ += PlackettLuce::calculate_gradient( X_[i], w_ );
				if ((i!=0 && i%bs_ == 0) || i==m_-1)
					update();
			}

			// get new likelihood
			ll_ = PlackettLuce::likelihood( X_, w_ );
			change = ll_ - prev_ll;
			if (v_) cerr << "Log Likelihood: " << ll_ << " (" << change << ")\n";
			*lo_ << ll_ << "\n";
			
			// if change in likelihood is too small, stop
			if ( fabs(change) < eps_ ) {
				stopped();
				return;
			}

			prev_ll = ll_;

		}

		stopped();

	}
	
	void inline update() {
		g_ *= eta_;
		w_ += g_;
		if (v_) cerr << "G ||| " << g_ << "\n"
					 << "W ||| " << w_ << "\n";
		u_++;
		g_.clear();
	}

	WeightVector getWeights() { return w_; }
	
	
private:

	void stopped() const {
		cerr << "stopped after " << t_ << " iterations ("
			 << u_ << " updates, final log likelihood: " << ll_ << ")\n\n";
	}

	const vector<Instance> X_; // vector of TrainingInstances
	const int m_; // number of instances
	const int T_; // number of iterations
	const double eps_; // stop training when likelihood does not grow anymore
	double eta_; // learning rate
	int bs_; // batch size before each update
	
	WeightVector w_;
	Gradient g_;
	unsigned int u_; // number of updates
	double ll_; // current log likelihood
	short t_; // current iteration
	const bool v_; // verbosity
	ostream* lo_; // out stream for likelihood
};

}

#endif /* SGD_H_ */
