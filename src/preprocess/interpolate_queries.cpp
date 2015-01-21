#include "interpolate_queries.h"

int main(int argc, char** argv) {

	// handle parameters
	po::variables_map cfg;
	if (!init_params(argc,argv, &cfg)) exit(1); // something is wrong

	ReadFile input1(cfg["1"].as<string>());
	ReadFile input2(cfg["2"].as<string>());
	double lambda = cfg["lambda"].as<double>();
	prob_t L = prob_t(cfg["LOWER"].as<double>());
	prob_t C = prob_t(cfg["CUMULATIVE"].as<double>());


	string line1, line2;
	while (getline(*input1, line1)) {
		getline(*input2, line2);
		if (line1.size() == 0 || line2.size() == 0) continue;

		Query q1(cfg.count("psq"),line1);
		Query q2(cfg.count("psq"),line2);

		assert(q1.size() == q2.size());
		assert(q1.id() == q2.id());

		Query I = q1.interpolate(q2, lambda);
		I.constrain(L,C, false);

		if (I.size() == 0)
			cerr << "WARNING: interpolated option table for id " << I.idstr() << " is empty!\n";

		stringstream tmp;
		I.save(std::cout);
		std::cout << "\n";

	}

}

