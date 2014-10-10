#include "clir.h"

bool init_params(int argc, char** argv, po::variables_map* cfg) {
	po::options_description cl("\nCreates an Index from Documents\nCommand Line Options");
	cl.add_options()
			("input,i", po::value<string>()->default_value("-"), "* Input documents (stdin)")
			("output,o", po::value<string>()->default_value("-"), "* Output File (stdout)")
			("text-input,t", po::value<bool>()->zero_tokens(), "indicate if input is <id>TAB<text> and not <id>TAB<tfvec>")
			("stopwords,s", po::value<string>(), "stopword file for filtering if text-input")
			("help,h", po::value<bool>()->zero_tokens(), "Show this help message");
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
	if (!init_params(argc,argv, &cfg)) exit(1);
	ReadFile in(cfg["input"].as<string>());
	WriteFile out(cfg["output"].as<string>());
	cerr << "CREATE-INDEX::Reading input from " << cfg["input"].as<string>() << endl;
	cerr << "CREATE-INDEX::Writing index to   " << cfg["output"].as<string>() << endl;
	if (cfg.count("text-input")) cerr << "CREATE-INDEX::Textual input from " << cfg["input"].as<string>() << endl;
	if (cfg.count("stopwords")) {
		sw::loadStopwords(cfg["stopwords"].as<string>());
		cerr << "CREATE-INDEX::" << sw::stopwordCount() << " stopwords loaded from '" << cfg["stopwords"].as<string>() << "'\n";
	}
	istream *ins = in.stream();
	ostream *outs = out.stream();
	assert(*ins);
	assert(*outs);
	Index::Index idx;
	string line;
	unsigned c=0;
	const bool is_text = cfg.count("text-input");
	const bool swf = cfg.count("stopwords");
	cerr << "CREATE-INDEX::Parsing documents.";
	while (getline(*ins, line)) {
		if (line.empty()) continue;
		CLIR::Document d(line, !is_text);
		if (is_text) d.computeTfVector(swf);
		if (!d.parsed_ || d.v_.empty())
			cerr << "CREATE-INDEX::WARNING: input vector (id=" << TD::Convert(d.id_) << ",len=" << d.len_ << ") empty!" << endl;
		else
			idx.AddDocument(d);
		if (c%10000==0) cerr << "." << c << ".";
		c++;
	}
	cerr << endl;
	cerr.flush();
	idx.ComputeAverageDocumentLength();
	idx.SortPostingsLists();
	cerr << "CREATE-INDEX::" << idx << endl;
	cerr << "CREATE-INDEX::Saving Index\n";
	idx.Save(*outs);
	cerr << "CREATE-INDEX::Done.\n";
	return 0;
}
