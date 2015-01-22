#ifndef TRAIN_BOWFD_H_
#define TRAIN_BOWFD_H_

#include <algorithm>
#include <climits>
#include <unordered_map>
#include <unordered_set>


#include "src/core/bowfd.h"

// cdec imports
#include "sampler.h"
boost::shared_ptr<MT19937> rng;

using namespace CLIR;

Decoder* setupDecoder(const po::variables_map& cfg) {
	register_feature_functions();
	SetSilent(true);
	ReadFile ini_rf(cfg["decoder_config"].as<string>());
	return new Decoder(ini_rf.stream());
}

bool init_params(int argc, char** argv, po::variables_map* conf) {
	po::options_description opts("Trains a BOW-FD model on preference pairs from input (default STDIN) using SGD with Adadelta (Zeiler'05). Can be parallelized via qsub.\nCommand Line Options");
	opts.add_options()
			("input,i", po::value<string>()->default_value("-"), "input (file or '-' for STDIN)")
			("decoder_config,c", po::value<string>(),"cdec configuration")
			("weights,w", po::value<string>(),"initial smt weights")
			("ir_weights", po::value<string>(), "initial ir weights")
			("iterations,I", po::value<unsigned>()->default_value(1), "# of iterations over input (cannot be STDIN if >1)")
			("default_ir_weight", po::value<double>()->default_value(0.0),"default ir weight for term match")
			("word_classes", po::value<string>()->default_value(""), "a file containing word2class mappings for extended matching")
			("dftable", po::value<string>(), "* df table")
			("stopwords,s",po::value<string>(), "stopword file")
			("epsilon,e", po::value<double>()->default_value(1e-6), "Adadelta constant")
			("decay_rate", po::value<double>()->default_value(0.95), "Adadelta decay_rate for sliding window of squared gradients")
			("tune_dense", po::value<bool>()->default_value(0), "tune a dense IR feature instead of sparse")
			("tune_smt", po::value<bool>()->default_value(1), "tune SMT features jointly with IR feature(s)")
			("freeze", po::value<vector<string> >()->multitoken(), "specify set of features to freeze")
			("output,o", po::value<string>(),"output directory to write learned weights to.")
			("hold_out", po::value<double>(),"percentage of input to hold out for loss calculation")
			("perceptron,P", po::value<bool>()->zero_tokens(), "use only a perceptron (zero margin required)")
			("fixed_margin,M", po::value<bool>()->zero_tokens(), "require a fixed margin of 1 for each example")
			("quiet", po::value<bool>()->zero_tokens(), "quiet ranker");

	po::options_description clo("Command line options");
	clo.add_options()("config", po::value<string>(),
			"Configuration file (cdec.ini)")("help,H",
			"Print this help message and exit");
	po::options_description dconfig_options, dcmdline_options;
	dconfig_options.add(opts);
	dcmdline_options.add(opts).add(clo);
	po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
	if (conf->count("config")) {
		ifstream config((*conf)["config"].as<string>().c_str());
		po::store(po::parse_config_file(config, dconfig_options), *conf);
	}
	po::notify(*conf);
	if (conf->count("help") 
		|| !conf->count("decoder_config") 
		|| !conf->count("dftable")
		|| !conf->count("output")) {
		cerr << dcmdline_options << endl;
		return false;
	}
	if ((*conf)["iterations"].as<unsigned>() > 1 && (*conf)["input"].as<string>() == "-") {
		cerr << "\nCannot iterate multiple times over STDIN!\n";
		return false;
	}
	return true;
}

void writeWeights(const string& fname, const WeightVector& w, const string& msg="") {
	WriteFile out(fname);
	ostream& o = *out.stream();
	assert(o);
	o.precision(12);
	if (!msg.empty()) {
		o << msg << "\n";
	}
	for (WeightVector::const_iterator i = w.begin(); i != w.end(); ++i) {
		if (i->second != 0.0)
			o << FD::Convert(i->first) << " " << i->second << "\n";
	}
	cerr << "Weights written to '" << fname << "'.\n";
}


#endif /* TRAIN_BOWFD_H_ */
