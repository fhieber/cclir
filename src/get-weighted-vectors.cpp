#include "get-weighted-vectors.h"


int main(int argc, char** argv) {

	// handle parameters
	po::variables_map cfg;
	if (!init_params(argc,argv, &cfg)) exit(1); // something is wrong

	// load df table
	DfTable dft (cfg["dftable"].as<string>());
	cerr << "DF table loaded (" << dft.size() << " entries)." << endl;
	
	const int N = cfg.count("N") ? cfg["N"].as<int>() : dft.mMaxDf;
	const double avg_len = cfg.count("avg_len") ? cfg["avg_len"].as<double>() : 1.0;
	Scorer* scorer = setupScorer(cfg["weight_metric"].as<string>(), N,avg_len, &dft);
	cerr << "Term weighting metric: " 
		 << cfg["weight_metric"].as<string>()
		 << " [normalization=" << cfg.count("normalize") << "]" << endl;

	int c = 0;
	string line;
	while (getline(cin, line)) {
		if (line.empty()) continue;
		Document d(line, true);
		if (!d.parsed_ || d.v_.empty())
			cerr << "WARNING: input vector (id=" << TD::Convert(d.id_) << ",len=" << d.len_ << ") empty!" << endl;
		scorer->score(d, cfg.count("normalize"));
		cout << d.asVector() << "\n";
		c++;
		c%1000==0 ? cerr << "." << c << "." : cerr ;
	}
	delete scorer;
	cerr << "done.\n\n"
		 << c << "\tlines read/written.\n";

}



