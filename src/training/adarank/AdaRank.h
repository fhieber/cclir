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
#include <float.h>
#ifdef _OPENMP
	#include <omp.h>
#else
	#define omp_get_thread_num() 0
#endif


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
		// ATTENTION!
		W_ = DenseWeightVector(weights.size(),0.0); // init W_ with zeros
		exp_losses_.reserve(no_iterations);
		deltas_.reserve(no_iterations);

		initializeDistributionUniformly();
		precomputeAccuracies();

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
			 * calculate alphas / find best feature
			 */
			vector<double> alphatop(J_,0.0);
			vector<double> alphabot(J_,0.0);
			vector<double> feature_score(J_,0.0);


			for (unsigned short j=1; j<J_; ++j) {
				for (short i=0; i<m_; ++i) {
					alphatop[j] += P_[i] * ( 1 + Acc_[j][i] );
					alphabot[j] += P_[i] * ( 1 - Acc_[j][i] );
					feature_score[j] += P_[i] * Acc_[j][i];
				}
			}

			for (int u=0; u<10; ++u)
				cerr << "P: " << P_[u] << " ";
			cerr << endl;

			int idx = 0;
			double max = -DBL_MAX;
			cerr << "phi values: ";
			for (int j=1; j<J_; ++j) {
				cerr << FD::Convert(j) << "=" << feature_score[j] << " ";
				if (feature_score[j] > max) {
				//if (FD::Convert(j) == "LanguageModel") {
					max = feature_score[j];
					idx = j;
				}
			}
			cerr << "\n";
			cerr << "Best performing feature: " << idx << " " << FD::Convert(idx) << " " << feature_score[idx] << "\n";

			double alpha = 0.5 * log( alphatop[idx] / alphabot[idx] );
			//alpha = -alpha;
			double phi_t = feature_score[idx]; // phi value of current best feature

			if (v_)
				cerr << "  alpha: " << alpha << "\n";

			DenseWeightVector W_prev(W_); // save previous W_ to own vector

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
			double min_delta = DBL_MAX; // set to max num
			int acc_improves = 0; // number of accuracy improvements between W_prev and W
			int acc_decr = 0;
			int acc_same = 0;
			double acc_sum = 0.0;
			/*
			 * update query distribution P_
			 */
			for (short i=0; i<m_; ++i) {
				double Acc_iW = scorer_->score(S_[i].GetWeightedPermutation(W_), *(S_[i].GetGoldPermutation())); // Ranking Acc. under current W_t for instance i
				double Acc_iW_prev = scorer_->score(S_[i].GetWeightedPermutation(W_prev), *(S_[i].GetGoldPermutation())); // Ranking Acc. under previous W_ for instance i
				double Acc_i_j = scorer_->score(*(S_[i].GetFeaturePermutation(idx)), *(S_[i].GetGoldPermutation())); // Ranking Acc. for winning Feature for instance i

				double L_i_W = exp( -Acc_iW ); // Loss = exp(-Acc)
				
				acc_sum += Acc_iW;
				
				if (Acc_iW > Acc_iW_prev) { 
					++acc_improves;
					//cerr << P_[i] << " cur=" << Acc_iW << " prev=" << Acc_iW_prev << endl;
				} else if (Acc_iW < Acc_iW_prev) { ++acc_decr; } else { ++acc_same; }
				
				
				P_[i] = L_i_W;
				sum_exp_losses += L_i_W;
				double delta_ti = Acc_iW - Acc_iW_prev - alpha * Acc_i_j;
				if (delta_ti < min_delta) {
					min_delta = delta_ti;
				}
			}
			double psum = 0.0;
			for (short i=0; i<m_; ++i) {
				P_[i] /= sum_exp_losses;
				psum += P_[i];
			}
			
			cerr << "psum=" << psum << endl;
			
			
			for (short i=0; i<m_; ++i) {
				double Acc_iW = scorer_->score(S_[i].GetWeightedPermutation(W_), *(S_[i].GetGoldPermutation())); // Ranking Acc. under current W_t for instance i
				double Acc_iW_prev = scorer_->score(S_[i].GetWeightedPermutation(W_prev), *(S_[i].GetGoldPermutation())); // Ranking Acc. under previous W_ for instance i
				double Acc_i_j = scorer_->score(*(S_[i].GetFeaturePermutation(idx)), *(S_[i].GetGoldPermutation())); // Ranking Acc. for winning Feature for instance i

				double L_i_W = exp( -Acc_iW ); // Loss = exp(-Acc)
				
				if (Acc_iW > Acc_iW_prev) { 
					//cerr << P_[i] << endl;
				}
			}
			
			cerr << "average (decoder) accuracy with current W: " << acc_sum / m_ << endl;
			cerr << "Accuracy +" << acc_improves << ", -" << acc_decr << ", =" << acc_same << endl;
			cerr << "minimum delta: " << min_delta << endl;
			double magic_num = exp(-min_delta) * sqrt(1 - pow(phi_t,2));
			cerr << "THE MAGIC NUMBER: " << magic_num << endl;

			delta_loss = sum_exp_losses_prev - sum_exp_losses;

			if (v_) cerr << "  loss=" << sum_exp_losses << " delta=" << delta_loss << "\n";
			if (!v_ && t%100==0) cerr << "."; 
			sum_exp_losses_prev = sum_exp_losses;

			exp_losses_.push_back(sum_exp_losses);
			cout << sum_exp_losses << "\t" << delta_loss << "\n";
			deltas_.push_back(delta_loss);

			//if ( delta_loss <= eps_ )
				//break;

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

	void precomputeAccuracies() {
		Acc_.clear();
		Acc_ = vector<vector<double> >(J_);
		vector<double> sums(J_);
		for (unsigned short j=1; j<J_; ++j) {
			Acc_[j] = vector<double>(m_);
			for (unsigned short i=0; i<m_; ++i) {
				Acc_[j][i] = scorer_->score(*(S_[i].GetFeaturePermutation(j)),*(S_[i].GetGoldPermutation()));
				sums[j] += Acc_[j][i];
			}
		}

		if (v_) {
			cerr << " Feature Accuracy precomputation finished. Accuracy sum: "
				 << std::accumulate(sums.begin(), sums.end(), 0.0)
				 << "  average acc per feature: ";
			for (unsigned short j=1; j<J_; ++j)
				cerr << FD::Convert(j) << "=" << sums[j] / m_ << " ";
			cerr << "\n";

		double l = 0.0;
		double acc_sum = 0.0;
		for (unsigned short i=0; i<m_; ++i) {
			double Acc_iW = scorer_->score(S_[i].GetWeightedPermutation(W_), *(S_[i].GetGoldPermutation())); // Ranking Acc. under current W_t for instance i
			acc_sum += Acc_iW;
			double Li = exp(-Acc_iW);
			l += Li;
		}
		cerr << "initial loss: " << l << endl;
		cerr << "average decoder accuracy: " << acc_sum / m_ << endl;

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
	vector<vector<double> > Acc_; // matrix of feature accuracies. constant across iterations
	ListwiseLossFunction* scorer_; // pointer to scoring / loss function (NDCG, MAP etc)

	const bool v_;
	// for output & plotting	
	vector<double> exp_losses_; // for output & plotting
	vector<double> deltas_; // for output & plotting


};

#endif /* ADARANK_H_ */
