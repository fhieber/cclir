#include "vectorize.h"

bool init_params(int argc, char** argv, po::variables_map* cfg) {
	po::options_description cl("\nTransforms text input into term vectors. Can write document frequency table as side product.\nCommand Line Options");
	cl.add_options()
			("input,i",	po::value<string>()->default_value("-"),"input file. Use '-' for STDIN.")
			("output,o",po::value<string>()->default_value("-"),"output file for term frequency vectors. Use '-' for STDOUT.")
			("stopwords,s",po::value<string>(),		"file with stopwords to filter")
			("dftable,d",po::value<string>(),		"output file for document frequency table")
			("skip-empty",po::value<bool>()->zero_tokens(),	"do not print empty vectors in output")
			("boolean", po::value<bool>()->zero_tokens(), "compute boolean vectors")
			("verbose,v",po::value<bool>()->zero_tokens(),	"verbose output to STDERR.")
			("help,h",	po::value<bool>()->zero_tokens(),	"print this help message.");

	po::store(parse_command_line(argc, argv, cl), *cfg);
	po::notify(*cfg);

	if (cfg->count("help")) {
		cerr << cl << endl;
		return false;
	}

	return true;
}

int main(int argc, char** argv) {

	// handle parameters
	po::variables_map cfg;
	if (!init_params(argc,argv, &cfg)) exit(1); // something is wrong

	// input / output
	ReadFile input(cfg["input"].as<string>());
	WriteFile out(cfg["output"].as<string>());

	// load stopwords
	if (cfg.count("stopwords")) {
		sw::loadStopwords(cfg["stopwords"].as<string>());
		cerr << sw::stopwordCount() << " stopwords loaded from '"
			 << cfg["stopwords"].as<string>() << "'\n";
	}

	DfTable dft;
	int c = 0;
	int sum_tokens = 0;
	int empty_output = 0;
	string docid, text;
	string line;
	while(getline(*input, line)) {
		if (line.empty()) continue;
		Document d(line);
		if (!d.parsed_) {
			cerr << "could not parse input!" << endl;
			continue;
		}
		if (cfg.count("boolean")) d.computeBooleanVector(cfg.count("stopwords"));
		else d.computeTfVector(cfg.count("stopwords"));
		if ( d.v_.size() == 0 ) {
			cerr << "WARNING: vector for docid " << TD::Convert(d.id_) << " is empty!" << endl;
			empty_output++;
			if (cfg.count("skip-empty"))
				continue;
		}
		// update df table
		dft.update(d.v_);
		// write to output
		*out << d.asVector() << "\n";
		sum_tokens += d.len_;
		c++;
		c%1000==0 ? cerr << "." << c << "." : cerr ;
	}

	cerr << "done." << endl << endl
		 << c << "\tlines read/written." << endl
		 << empty_output << "\ttf vectors empty." << endl
		 << (sum_tokens / (1.0*c)) << "\taverage doc length" << endl;

	if (cfg.count("dftable"))
		dft.writeToFile(cfg["dftable"].as<string>());

	return 0;

}

