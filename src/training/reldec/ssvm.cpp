#include <vector>
#include <numeric>
#include <cfloat>
#include <sys/time.h>

#include <stdlib.h>
#include <stdio.h>

#ifdef _OPENMP
	#include <omp.h>
#else
	#define omp_get_thread_num() 0
#endif

#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

#include "hg.h"
#include "hg_io.h"
#include "viterbi.h"
#include "feature_vector.h" // typedefs FeatureVector,WeightVector, DenseWeightVector
#include "weights.h"
#include "filelib.h"

using namespace std;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

typedef unsigned long long timestamp_t;

static timestamp_t get_timestamp () {
	struct timeval now;
	gettimeofday (&now, NULL);
	return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

// this way lr for all features is equal to the commandline specified lr in 1st iter
const double beta = 1.0; // AdaGrad beta parameter;
int J = 0; // number of features
int H = 0; // number of hypergraphs
int I = 0; // number of iterations
unsigned T = 1; // number of threads
WeightVector w_smt, w_rel, w_rel_scaled, w_hope, w_fear;
vector<FeatureVector> hopes, fears; // vector of hope and fear derivations
vector<double> rel_oracles; // relevance of oracles
bool perceptron, ramploss, freeze;
vector<int> frozen_features;
unsigned scaling;
double scalingfactor;

bool init_params(int argc, char** argv, po::variables_map* cfg) {
	stringstream ss;
	ss << "\nssvm for relevance decoding";
	po::options_description cl(ss.str());
	cl.add_options()
		("input,i", po::value<string>(), "* input directory w/ hypergraphs")
		("weights,w", po::value<string>(), "* smt weights (to be optimized)")
		("rweights,r", po::value<string>(), "* relevance weights")
		("perceptron,p", po::value<bool>()->zero_tokens(), "use perceptron instead of SVM. No margin")
		("ramploss", po::value<bool>()->zero_tokens(), "use ramp loss")
		("output,o", po::value<string>(), "* output file for final model")
		("iters", po::value<int>()->default_value(10), "number of iterations")
		("learningrate,n", po::value<double>()->default_value(0.001), "learning rate")
		("reg,l", po::value<double>()->default_value(0.0), "l1 regularization w/ clipping (Tsuoroka et al). specify regularization strength")
		("adagrad,a", po::value<bool>()->zero_tokens(), "use per-coordinate learning rate (Duchi et al, 2010 / Green et al, ACL'13)")
		#ifdef _OPENMP
		("jobs,j", po::value<int>(), "Number of threads. Default: number of cores")
		#else
		#endif
		("eval,e", po::value<int>()->default_value(1), "compute statistics every e-th iteration.")
		("dump,d", po::value<int>()->default_value(500), "dump model to <o>.d at every d-th iteration")
		("viterbi,v", po::value<bool>()->zero_tokens(), "write viterbi translations to disk when dumping model (requires rules to be in the hypergraphs)")
		("freeze", po::value<vector<string> >()->multitoken(), "specify set of frozen features (excluded from the gradient)")
		("scaling,s", po::value<int>()->default_value(1), "scale rw to ||w||: 0 (disabled), 1 (rescale at initializiation), 2 (rescale after each update of w).")
		("scalingfactor", po::value<double>()->default_value(1), "x in rw = (rw / ||w||) * x")
		("help,h", po::value<bool>()->zero_tokens(), "prints this help message");
	po::store(parse_command_line(argc, argv, cl), *cfg);
	po::notify(*cfg);

	if(cfg->count("help")) {
		cerr << cl << endl;
		return false;
	}
	if(!cfg->count("input") || !cfg->count("weights") || !cfg->count("rweights") || !cfg->count("output")) {
		cerr << cl << endl;
		return false;
	}
	return true;
}
	

void normalize(WeightVector& w) { w /= w.l2norm(); }

void scaleRelevanceWeights(const double scalingfactor) {
	w_rel_scaled /= w_rel_scaled.l2norm(); // to length 1: w_rel / ||w_rel||
	const double cur_w_smt_norm = w_smt.l2norm(); // ||w_smt||
	if (cur_w_smt_norm == 0) return;
	w_rel_scaled *= cur_w_smt_norm * scalingfactor; // scale by ||w_smt||*scalingfactor
}

/*
 * converts sparse to dense while keeping size of dense intact (contrary to feature_vector.h::sparse_to_dense())
 */
int sparse2dense(const WeightVector& v, DenseWeightVector* dv) {
	int nonzero=0;
	for (int i=0;i<dv->size();++i) dv->at(i) = .0;
	for(WeightVector::const_iterator i=v.begin();i!=v.end();++i) {
		assert(i->first < dv->size());
		(*dv)[i->first] = i->second; 
		if (i->second != 0)
			nonzero++;
	}
	return nonzero;
}

void loadWeights(const string& fname, WeightVector& w) {
	DenseWeightVector dw;
	Weights::InitFromFile(fname, &dw);
	Weights::InitSparseVector(dw, &w);
}

void loadRelevanceWeights(const string& fname, WeightVector& rw) {
	rw.clear();
	ReadFile in_file(fname);
	istream& in = *in_file.stream();
	assert(in);
	string f;
	double v;
	while(in>>f) {
		in>>v;
		if (f.empty()) continue;
		rw.set_value(FD::Convert(f), v);
	}
}

void writeWeights(const string& fname, const WeightVector& w) {
	WriteFile out(fname);
	ostream& o = *out.stream();
	assert(o);
	o.precision(17);
	for (WeightVector::const_iterator i=w.begin(); i!=w.end(); ++i) {
		if (i->second != 0.0)
			o << FD::Convert(i->first) << " " << i->second << "\n";
	}
	//cerr << "Weights written to '" << fname << "'.\n";
}

void loadHypergraphs(const string& dir, vector<Hypergraph>& hgs) {
	timestamp_t t0 = get_timestamp();
	vector<fs::path> paths;
	fs::directory_iterator end_itr;
    for (fs::directory_iterator itr(dir); itr != end_itr; ++itr)
    {
        if (fs::is_regular_file(itr->path())) {
            paths.push_back(itr->path().string());
        }
    }
	std::sort(paths.begin(), paths.end());
	int edges=0,nodes=0;
	hgs.resize(paths.size());
	int x = paths.size() < 10 ? 1 : paths.size() / 10;
	cerr << "Loading " << paths.size() << " hypergraphs ";
	for (int i=0;i<paths.size();++i) {
		ReadFile rf(paths[i].string());
		string raw;
		rf.ReadAll(raw);
		Hypergraph hg;
		HypergraphIO::ReadFromPLF(raw, &hg);
		edges += hg.NumberOfEdges();
		nodes += hg.NumberOfNodes();
		hgs[i] = hg;
		if (paths.size()%x==0) {
			cerr << '.';
			cerr.flush();
		}
	}
	J = FD::NumFeats();
	timestamp_t t1 = get_timestamp();
	cerr << " ok " 
		 << J << " features, ~" 
		 << edges / (double) hgs.size() 
		 << " edges/hg, ~" 
		 << nodes / (double) hgs.size() 
		 << " nodes/hg (" << (t1 - t0) / 1000000.0L << "s)\n";
	H = hgs.size();
	hopes.resize(H);
	fears.resize(H);
	rel_oracles.resize(H);
}

/*
 * write intermediate model and viterbi translations
 */
void dumpModel(vector<Hypergraph>& hgs, const WeightVector& w, const int i, const string& output, const bool viterbi) {
	// weights
	stringstream ss;
	ss << output << "." << i;
	writeWeights(ss.str(), w);

	// translations
	if (viterbi) {
		stringstream ts;
		ts << output << "." << i << ".viterbi";
		WriteFile tout(ts.str());
		for ( int h=0; h<hgs.size(); ++h) {
			Hypergraph& hg = hgs[h];
			hg.Reweight(w);
			vector<WordID> trans;
			// if this fails an assertion about TRule, the hypergraphs
			// have no rules in them to construct the output sentence
			ViterbiESentence(hg, &trans);
			*tout << TD::GetString(trans) << endl;
		}
	}
}

/*
 * removes dimensions from g that are present in w_hope.
 * we dont want to update relevance weights
 */
void filterGradient(FeatureVector& f, const WeightVector& rw) {
	for (WeightVector::const_iterator it=rw.begin(); it!=rw.end(); ++it)
		f.set_value(it->first, .0);
}

/*
 * regular FOBOS 2-step update with constant learning rate
 */
void update(DenseWeightVector& dw, const DenseWeightVector& dg, const double lr, const double lambda, const bool l1) {
	// for all feature ids in parallel
	#pragma omp parallel for
	for ( int j=1;j<J;++j ) { // feature ids start with 1
		// FOBOS (1) gradient step: wj = wj - lr * gj
		if (dg[j] != 0.0)
			dw[j] -= lr * dg[j];
		// FOBOS (2) l1 regularization
		if (l1) {
			if (dw[j] > 0)
				dw[j] = max(.0, dw[j] - lambda * lr);
			else if (dw[j] < 0)
				dw[j] = min(.0, dw[j] + lambda * lr);
		}
	}
}

/*
 * Green et al, Fast and Adaptive Online Training of
 * Feature-Rich Translation Models (ACL'13)
 * AdaGrad applies per-coordinate (j) learning rate to the gradient:
 * lr_j = 1 / (beta + sqrt(G^t)
 *
 * FOBOS (1) [eq. 13,14]
 * gradient = gradient .* (lr / (beta + sqrt(G^t)))
 * beta is a fixed parameter (>lr) so that denominator is not 0 for t=0;
 *
 * w is updated with the per coordinate learning rate
 *
 * FOBOS (2) [eq. 15]
 * l1 regularization with w/ adaptive learning rate
 */
void update_adaGrad(DenseWeightVector& dw, const DenseWeightVector& dg, DenseWeightVector& G, const double lr, const double lambda, const bool l1) {
	// for all features in parallel do
	#pragma omp parallel for
	for ( int j=1;j<J;++j ) { // feature ids start with 1
		// FOBOS (1) gradient step: wj = wj - lrj * gj
		if (dg[j])
			dw[j] -= (lr / (beta + sqrt(G[j]))) * dg[j];
		// FOBOS (2) l1 regularization
		if (l1) {
			if (dw[j] > 0)
				dw[j] = max(.0, dw[j] - lambda * (lr / (beta + sqrt(G[j]))));
			else if (dw[j] < 0)
				dw[j] = min(.0, dw[j] + lambda * (lr / (beta + sqrt(G[j]))));
		}
		// update sum of squared gradient values
		if (dg[j])
			G[j] += dg[j] * dg[j];
	}
}

int main(int argc, char** argv) {
	po::variables_map cfg;
	if (!init_params(argc,argv,&cfg)) exit(1);
	T = cfg["jobs"].as<int>();
	#ifdef _OPENMP
	omp_set_num_threads(T);
	#endif
	I = cfg["iters"].as<int>();
	perceptron = cfg.count("perceptron");
	ramploss = cfg.count("ramploss");
	freeze = cfg.count("freeze");
	if (freeze) {
		const vector<string>& ffstrs = cfg["freeze"].as<vector<string> >();
		stringstream ffss;
		ffss << "frozen features: ";
		for (vector<string>::const_iterator ffit=ffstrs.begin();ffit!=ffstrs.end();++ffit) {
			frozen_features.push_back(FD::Convert(*ffit));
			ffss << *ffit << " ";
		}
		cerr << ffss.str() << endl;
	}
	scaling = cfg["scaling"].as<int>();
	scalingfactor = cfg["scalingfactor"].as<double>();
	cerr << "scaling=" << scaling << " scalingfactor=" << scalingfactor << endl;

	// the SMT weights (to be optimized)
	loadWeights(cfg["weights"].as<string>(), w_smt);
	// the relevance weights
	loadRelevanceWeights(cfg["rweights"].as<string>(), w_rel);
	w_rel_scaled = w_rel;
	// initial scaling
	if (scaling != 0) scaleRelevanceWeights(scalingfactor);

	// output some vector stats
	cerr << "W_REL=" << w_rel << endl;
	cerr << "W_REL_SCALED=" << w_rel_scaled << endl;
	cerr << "|W_REL|=" << w_rel_scaled.size() << endl;
	cerr << "|W_SMT|=" << w_smt.size() << endl;

	if (ramploss) { cerr << "RAMPLOSS\n"; }

	// load hypergraphs
	vector<Hypergraph> hgs;
	loadHypergraphs(cfg["input"].as<string>(), hgs);

	cerr << "I\tLoss\tCost\tMargin\tR_fear\tR_vit\tR_hope\tR_oracle\tFeatures\tTime\n";

	const double lr = cfg["learningrate"].as<double>(); // learning rate
	const double lambda = cfg["reg"].as<double>(); // l1 regularization strength
	const bool l1 = (lambda != 0.0); // l1? yes/no
	const int eval = cfg["eval"].as<int>();
	const int dump = cfg["dump"].as<int>();
	double loss, vrel, hrel, frel, orel=0, margin, cost, prev_loss = DBL_MAX;
	//FeatureVector g; // gradient
	DenseWeightVector G(J, .0); // squared gradient coordinates over time
	DenseWeightVector dg(J, .0); // dense gradient
	DenseWeightVector dw(J, .0); // dense weights
	timestamp_t t0,t1;
	vector<FeatureVector> pg(T, FeatureVector()); // partial gradients (parts == # threads)

	// compute oracles and relevances once
	# pragma omp parallel for reduction(+:orel)
	for ( int h=0; h<H; ++h ) {
		Hypergraph& hg = hgs[h];
		hg.Reweight(w_rel);
		hopes[h] = ViterbiFeatures(hg);
		rel_oracles[h] = hopes[h].dot(w_rel);
		orel += rel_oracles[h];
	}
	orel /= (double) H;

	for ( unsigned int i=0; i<I; ++i ) {

		// the weight vector that gives you fear
		w_fear = perceptron ? w_smt : (w_smt - w_rel_scaled) ;

		for (unsigned t=0;t<T;++t) pg[t].clear(); // clear thread gradients
		loss=0, vrel=0, hrel=0, frel=0, margin=0, cost=0;
		t0 = get_timestamp();
		# pragma omp parallel for reduction(+:loss,vrel,hrel,frel,margin,cost)
		for ( int h=0; h<H; ++h) {

			Hypergraph& hg = hgs[h];
			// if first iter or ramploss, compute hopes/oracles
			if (ramploss) {
				hg.Reweight(w_smt + w_rel_scaled);
				hopes[h] = ViterbiFeatures(hg);
			}
			FeatureVector& hope = hopes[h];
			// get fear
			hg.Reweight(w_fear);
			fears[h] = ViterbiFeatures(hg);
			FeatureVector& fear = fears[h];
			// accumulate gradient
			pg[omp_get_thread_num()] += fear - hope;

			if (i%eval==0) { // get viterbi derivations, calculate loss
				hg.Reweight(w_smt);
				FeatureVector normal = ViterbiFeatures(hg);
				double this_orel = rel_oracles[h];
				double this_vrel = w_rel.dot(normal); // viterbi relevance
				double this_hrel = w_rel.dot(hope);   // hope relevance
				double this_frel = w_rel.dot(fear);   // fear relevance
				double c = (fear-hope).dot(w_smt); // cost: w(fear - hope)
				double m = (this_orel - this_frel) + (this_orel - this_hrel); // margin: delta_fear + delta_hope
				if (perceptron) m = .0;
				margin += m;
				cost += c;
				loss += max( c + m , .0 );
				vrel += this_vrel;
				hrel += this_hrel;
				frel += this_frel;
			}
			
		}
		// accumulate gradients from threads into master gradient (0)
		FeatureVector& g = pg[0];
		for (vector<FeatureVector>::iterator git=pg.begin()+1;git!=pg.end();++git) { g += *git; }
		filterGradient(g, w_rel);

		if (i%eval==0) {
			if (l1) { loss += lambda * w_smt.pnorm(1); }
			loss   /= (double) H;
			margin /= (double) H;
			cost   /= (double) H;
			vrel   /= (double) H;
			hrel   /= (double) H;
			frel   /= (double) H;
			t1 = get_timestamp();
			cout.precision(6);
			cout << i      << '\t' 
			     << loss   << '\t' 
			     << cost   << '\t' 
			     << margin << '\t'
			     << frel   << '\t'
			     << vrel   << '\t' 
			     << hrel   << '\t'
			     << orel   << '\t'
			     << w_smt.num_nonzero() << '\t'
			     << (t1-t0) / 1000000.0L << "s\n";
			cout.flush();
			if (loss <= 0 ||  fabs(prev_loss - loss) < 1e-10 ) {// converged
				cerr << "converged...\n";
				break;
			}
			prev_loss = loss;
		}

		if (cfg.count("adagrad")) {

			g /= H;
			// get dense vectors
			sparse2dense(g, &dg);
			sparse2dense(w_smt, &dw);
		
			if (freeze) { // if frozen features
				for (int ff=0;ff<frozen_features.size();++ff) dg[frozen_features[ff]] = 0.0;
			}
			update_adaGrad( dw, dg, G, lr, lambda, l1 );
			Weights::InitSparseVector(dw, &w_smt);

		} else {
			// remove frozen features from gradient
			if(freeze) {
				for (int ff=0;ff<frozen_features.size();++ff) g.set_value(frozen_features[ff], .0);
			}
			// update
			w_smt.plus_eq_v_times_s(g, -lr/H);
			// l1
			if (l1) {
				for (WeightVector::iterator it=w_smt.begin();it!=w_smt.end();++it) {
					if (it->second > .0) it->second = max(.0, it->second - lambda * lr);
					else if (it->second < .0) it->second = min(.0, it->second + lambda * lr);
				}
			}
		}

		// rescale relevance weights to balance with new model after the update
		if (scaling == 2) scaleRelevanceWeights(scalingfactor);
		// dumpModel
		if (i%dump==0) dumpModel(hgs, w_smt, i, cfg["output"].as<string>(), cfg.count("viterbi"));

	} // end of iteration

	cerr << "\ndone.\n";
	writeWeights(cfg["output"].as<string>(), w_smt);

}

