/*
 * AdaRank.h
 *
 *  Created on: Apr 22, 2013
 */

#ifndef ADARANK_H_
#define ADARANK_H_

#include <vector>
#include "TrainingInstance.h"
#include "losses.h"
#include <math.h>
#include <numeric>
#include <omp.h>


using namespace std;

class AdaRank {
public:
	AdaRank(const vector<TrainingInstance> S,
			const int& S_size,
			const int& no_iterations,
			const double& epsilon,
			DenseWeightVector& weights,
			ListwiseLossFunction* loss,
			const bool& verbose)
		: S_(S), m_(S_size), T_(no_iterations), eps_(epsilon), J_(weights.size()), W_(weights), scorer_(loss), v_(verbose)
	{
		exp_losses_.reserve(no_iterations);
		deltas_.reserve(no_iterations);

		initializeDistributionUniformly();
		precomputeLoss();

	};

	DenseWeightVector GetWeightVector() { return W_; };

	void run() {

		unsigned short t=0;
		double sum_exp_losses = 0.0; // sum of exp losses
		double sum_exp_losses_prev = 0.0; // previous sum of exp losses
		double delta_loss = 0.0; // loss change between iterations

		if (v_) {
			cerr << " W0: ";
			for (short j=1; j<J_; ++j)
				cerr << W_[j] << " ";
			cerr << endl;
		}

		for (t=1; t<T_; ++t) {

			if (v_) cerr << " T="<<t<<":\n";

			/*
			 * calculate alphas
			 */
			vector<double> alphatop(J_,0.0);
			vector<double> alphabot(J_,0.0);
			vector<double> feature_score(J_,0.0);


#pragma omp parallel
			{ // begin of parallel section
#pragma omp for
			for (unsigned short j=1; j<J_; ++j) {
				for (short i=0; i<m_; ++i) {
					alphatop[j] += P_[i] * ( 1 + L_[j][i] );
					alphabot[j] += P_[i] * ( 1 - L_[j][i] );
					feature_score[j] += P_[i] * L_[j][i];
				}
			}
#pragma omp master
			{ // begin master

			for (int u=0; u<10; ++u)
				cerr << "P: " << P_[u] << " ";
			cerr << endl;

			int idx = 0;
			double max = -5000;
			for (int j=1; j<J_; ++j) {
				if (feature_score[j] > max) {
					max = feature_score[j];
					idx = j;
				}
			}
			cerr << "Best performing feature: " << idx << " " << FD::Convert(idx) << " " << feature_score[idx] << "\n";

			double alpha = 0.5 * log( alphatop[idx] / alphabot[idx] );

			if (v_)
				cerr << "  alpha: " << alpha << "\n";

			/*
			 * create combined ranker w_
			 */
			W_[idx] += alpha;

			if (v_) {
				cerr << "  W"<<t<<": ";
				for (short j=1; j<J_; ++j)
					cerr << W_[j] << " ";
				cerr << "\n";
			}

			sum_exp_losses = 0.0;

			} // end of master exclusive block
			/*
			 * update query distribution P_
			 */
#pragma omp barrier
#pragma omp for reduction(+:sum_exp_losses)
			for (short i=0; i<m_; ++i) {
				double L_i_W = exp( - (*scorer_)(S_[i].GetWeightedPermutation(W_), *(S_[i].GetGoldPermutation())) );
				P_[i] = L_i_W;
				sum_exp_losses += L_i_W;
			}
#pragma omp for
			for (short i=0; i<m_; ++i) {
				P_[i] /= sum_exp_losses;
			}
			} // end parallel section
			
			delta_loss = fabs(sum_exp_losses_prev - sum_exp_losses);

			if (v_) cerr << "  loss=" << sum_exp_losses << " delta=" << delta_loss << "\n";
			if (!v_ && t%100==0) cerr << "."; 
			sum_exp_losses_prev = sum_exp_losses;

			exp_losses_.push_back(sum_exp_losses);
			deltas_.push_back(delta_loss);

			if ( delta_loss <= eps_ )
				break;

		} // end of iterations loop

		for (unsigned short x=0; x<exp_losses_.size(); ++x) 
			cout << exp_losses_[x] << "\t" << deltas_[x] << "\n";

		cerr << "\nfinished after " << t << " iterations.\n";
		for (short j=1; j<J_; ++j)
			W_[j] /= t;

	}

	void initializeDistributionUniformly() {
		P_ = vector<double>(m_, 1.0/m_);
		if (v_) cerr << " initialized uniform distribution over instances (p="<<1.0/m_<<").\n";
	}

	void precomputeLoss() {
		L_.clear();
		L_ = vector<vector<double> >(J_);
		vector<double> sums(J_);
#pragma omp parallel for
		for (unsigned short j=1; j<J_; ++j) {
			L_[j] = vector<double>(m_);
			for (unsigned short i=0; i<m_; ++i) {
				L_[j][i] = (*scorer_)(*(S_[i].GetFeaturePermutation(j)), *(S_[i].GetGoldPermutation()));
				sums[j] += L_[j][i];
			}
		}

		if (v_) {
			cerr << " loss precomputation finished. loss sum: "
				 << std::accumulate(sums.begin(), sums.end(), 0.0)
				 << "  average loss per feature: ";
			for (unsigned short j=1; j<J_; ++j)
				cerr << FD::Convert(j) << "=" << sums[j] / m_ << " ";
			cerr << "\n";
		}

	}

private:
	const vector<TrainingInstance> S_; // vector of TrainingInstances
	const int m_; // number of training instances
	const int T_; // number of iterations TODO determine T by delta of changes
	const double eps_; // stop boosting when change of loss is smaller than eps_

	const unsigned short J_; // number of features (fids_.size())
	DenseWeightVector W_; // weight vector (from previous epoch or initialized)
	vector<double> P_; // weight distribution over instances
	vector<vector<double> > L_; // matrix of losses. constant across iterations
	ListwiseLossFunction* scorer_; // pointer to scoring / loss function (NDCG, MAP etc)

	const bool v_;
	// for output & plotting	
	vector<double> exp_losses_; // for output & plotting
	vector<double> deltas_; // for output & plotting


};

#endif /* ADARANK_H_ */
