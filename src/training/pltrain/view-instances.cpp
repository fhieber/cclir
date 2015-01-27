/*
 * view-instances.cpp
 *
 *  Created on: Apr 29, 2013
 */

#include "view-instances.h"

int main(int argc, char** argv) {

	if (argc < 2 || argc > 3) {
		cerr << "USAGE: view-instances <binary instance file> [<weights>]\n";
		exit(1);
	}

	bool has_w = (argc==3);
		
	// load instances
	vector<Instance> instances;
	loadInstances(string(argv[1]), instances);
	
	// weights
	WeightVector weights;
	if (has_w) {
		DenseWeightVector dense_weights;
		Weights::InitFromFile(string(argv[2]), &dense_weights);
		Weights::InitSparseVector(dense_weights, &weights);
		cerr << "Current Weight Vector:\n";
		for (WeightVector::iterator i=weights.begin(); i!=weights.end(); ++i)
			cerr << i->first << " " << FD::Convert(i->first) << "=" << i->second << endl;
	}

	double likelihood = 0.0;
	double likelihood_i=0.0;
	for (int i=0;i<instances.size();++i) {
		if (instances[i].ir_sorted) {
			if (has_w)
				likelihood_i = PlackettLuce::pl_likelihood(instances[i], weights);
			else
				likelihood_i = PlackettLuce::pl_likelihood(instances[i]);
			cout << "P(y|x;";
			if (has_w) cout << "w)=";
			else cout << "D)=";
			cout << likelihood_i << "\n";
		}
		
		if (has_w)
			cout << instances[i].AsString(weights) << endl;
		else
			cout << instances[i].AsString() << endl;
		likelihood +=likelihood_i;
		

		
	}
	
	cerr << "Likelihood=" << likelihood << "\n";

}

/*
// test instance
Derivation::Derivation d1,d2,d3;
d1.mtscore = 10;
d2.mtscore = 5;
d3.mtscore = 1;

d1.irscore = 5;
d2.irscore = 6;
d3.irscore = 7;

Instance x;
x.derivations.push_back(d1);
x.derivations.push_back(d2);
x.derivations.push_back(d3);
x.mt_sorted = true;
x.size = 3;
x.has_irscores = true;
x.SortByRetrievalScores();

cerr << x.AsString() << endl;
cerr << PlackettLuce::pl_loss(x) << endl;
double myloss = d1.mtscore + d2.mtscore + d3.mtscore - (log(exp(d3.mtscore)+exp(d2.mtscore)+exp(d1.mtscore)) + log(exp(d2.mtscore)+exp(d1.mtscore)) + log(exp(d1.mtscore)));
cerr << myloss << endl;
exit(0);
*/
	
	
/*
// test gradients
Gradient g1;
Gradient g2;
for (WeightVector::iterator it=weights.begin();it!=weights.end();++it)
	g1[it->first] = PlackettLuce::gradient_naive(instances[i], weights, it->first);
for (WeightVector::iterator it=weights.begin();it!=weights.end();++it)
	g2[it->first] = PlackettLuce::gradient(instances[i], weights, it->first);
Gradient g3 = PlackettLuce::calculate_gradient(instances[i], weights);

cerr << "Gradient 1: ";
for (Gradient::iterator it=g1.begin(); it!=g1.end(); ++it)
	cerr << FD::Convert(it->first) << "("<<it->first<<")=" << it->second << " ";
cerr << endl;

cerr << "Gradient 2: ";
for (Gradient::iterator it=g2.begin(); it!=g2.end(); ++it)
	cerr << FD::Convert(it->first) << "("<<it->first<<")=" << it->second << " ";
cerr << endl;

cerr << "Gradient 3: ";
for (Gradient::iterator it=g3.begin(); it!=g3.end(); ++it)
	cerr << FD::Convert(it->first) << "("<<it->first<<")=" << it->second << " ";
cerr << endl;
*/	


