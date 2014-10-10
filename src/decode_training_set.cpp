#include <unordered_map>
#include "ranker.h"
#include <climits>

using namespace CLIR;

Decoder* setupDecoder(const po::variables_map& cfg) {
	register_feature_functions();
	SetSilent(true);
	ReadFile ini_rf(cfg["decoder_config"].as<string>());
	return new Decoder(ini_rf.stream());
}

bool init_params(int argc, char** argv, po::variables_map* conf) {
	po::options_description opts;
	opts.add_options()
		("decoder_config,c",po::value<string>(),"Decoder configuration file")
		("weights,w", po::value<string>(), "initial weights")
		("ir_weights", po::value<string>(), "ir weights")
		("default_ir_weight", po::value<double>()->default_value(1.0), "default ir weight for term match")
		("word_classes", po::value<string>()->default_value(""), "a file containing word2class mappings for extended matching")
		("tune_dense", po::value<bool>()->default_value(0), "tune a dense IR feature instead of sparse")
		("tune_smt", po::value<bool>()->default_value(1), "tune SMT features jointly with IR feature(s)")
		("dftable", po::value<string>(), "* DF table to load")
		("stopwords,s",po::value<string>(),		"stopword file")
		("model,m", po::value<string>(), "outputs SMT and IR combined model from this run.");

	po::options_description clo("Command line options");
	clo.add_options()
		("config", po::value<string>(), "Configuration file (cdec.ini)")
		("help,H", "Print this help message and exit");
	po::options_description dconfig_options, dcmdline_options;
	dconfig_options.add(opts);
	dcmdline_options.add(opts).add(clo);
	po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
	if (conf->count("config")) {
		ifstream config((*conf)["config"].as<string>().c_str());
		po::store(po::parse_config_file(config, dconfig_options), *conf);
	}
	po::notify(*conf);
	if(conf->count("help") || !conf->count("decoder_config") || !conf->count("avg_len") || !conf->count("model")) {
		cerr << dcmdline_options << endl;
		return false;
	}
	return true;
}

void writeWeights(const string& fname, const WeightVector& w) {
	WriteFile out(fname);
	ostream& o = *out.stream();
	assert(o);
	o.precision(12);
	for (WeightVector::const_iterator i = w.begin(); i != w.end(); ++i) {
		if (i->second != 0.0)
			o << FD::Convert(i->first) << " " << i->second << "\n";
	}
	cerr << "Weights written to '" << fname << "'.\n";
}

// vowpal wabbit feature id escape: vw requires fid:val format
static inline string VWEscape(const string& x) {
	string y = x;
	for (int i = 0; i < y.size(); ++i) {
	  if (y[i] == ':') y[i]='@'; // change existing ':' to something else
	  if (y[i] == '|') y[i]='\\';
	}
	return y;
}

// vowpal wabbit feature id unescape
static inline string VWUnescape(const string& x) {
	string y = x;
	for (int i = 0; i < y.size(); ++i) {
	  if (y[i] == '@') y[i]=':';
	  if (y[i] == '\\') y[i]='|';
	}
	return y;
}

static string ConvertToVWVector(const FeatureVector& x) {
	stringstream ss;
	for (FeatureVector::const_iterator i=x.begin();i!=x.end();++i) {
		if (i->second != 0) ss << VWEscape(FD::Convert(i->first)) << ":" << i->second << " ";
	}
	return ss.str();
};

int main(int argc, char** argv) {
	po::variables_map cfg;
	if (!init_params(argc,argv,&cfg)) return 1;

	// load stopwords
	if (cfg.count("stopwords")) {
		sw::loadStopwords(cfg["stopwords"].as<string>());
		cerr << sw::stopwordCount() << " stopwords loaded from '"
			 << cfg["stopwords"].as<string>() << "'\n";
	}
	// setup decoder
	Decoder* decoder = setupDecoder(cfg);
	if (!decoder) {
		cerr << "error while loading decoder with" << cfg["decoder_config"].as<string>() << "!\n";
		return 1;
	}
	// get reference to decoder weights
	vector<weight_t>& decoder_weights = decoder->CurrentWeightVector();
	WeightVector w_smt;
	// the SMT weights
	if (cfg.count("weights")) {
		Weights::InitFromFile(cfg["weights"].as<string>(), &decoder_weights);
		Weights::InitSparseVector(decoder_weights, &w_smt);
	} else {
		cerr << "starting with EMPTY weights!\n";
	}
	WeightVector w_ir;
	if (cfg.count("ir_weights")) {
		vector<weight_t> dirw;
		Weights::InitFromFile(cfg["ir_weights"].as<string>(), &dirw);
		Weights::InitSparseVector(dirw, &w_ir);
	}

	// load documents
	const double avg_len = cfg["avg_len"].as<double>();
	DfTable dft (cfg["dftable"].as<string>());
	cerr << "DF table loaded (" << dft.size() << " entries)." << endl;
	IRFeatureExtractor extractor(&w_ir, cfg["default_ir_weight"].as<double>(), cfg["word_classes"].as<string>());
	IRFeatureIndicator indicator(&extractor, &dft);

	TrainingRanker ranker(decoder,
						  &indicator,
						  &w_smt,
						  &w_ir,
						  cfg.count("tune_dense"),
						  cfg.count("quiet"));
	ranker.extract_smt = cfg.count("tune_smt");
	ranker.extract_sparse = !cfg.count("tune_dense");
	ranker.extract_dense = cfg.count("tune_dense");

	WordID prev_qid=-1;
	WordID prev_d_rel_id=-1;
	string qid, qstr, d_rel_id, d_rel_tfstr, d_irel_id, d_irel_tfstr;
	int margin, d_rel_len, d_irel_len;
	FeatureVector f_rel, f_irel;
	double s_rel=0, s_irel=0; // scores
	CLIR::FDocument* d_rel=NULL;
	unsigned cp = 0;  // # examples
	unsigned cco = 0; // # examples correctly ordered
	unsigned cne = 0; // # negative examples
	cerr << "reading from STDIN...\n";
	while(cin >> qid) {
		cin.ignore(1, '\t');
		getline(cin, qstr, '\t');
		cin >> margin;
		cin.ignore(1, '\t');
		cin >> d_rel_id;
		cin.ignore(1, '\t');
		cin >> d_rel_len;
		cin.ignore(1, '\t');
		getline(cin, d_rel_tfstr, '\t');
		cin >> d_irel_id;
		cin.ignore(1, '\t');
		cin >> d_irel_len;
		cin.ignore(1, '\t');
		getline(cin, d_irel_tfstr);
		if (qid.empty() || qstr.empty() || margin==0 || d_rel_id.empty() || d_rel_len==0 || d_rel_tfstr.empty() || d_irel_id.empty() || d_irel_len==0 || d_irel_tfstr.empty()) continue;

		if (TD::Convert(qid) != prev_qid) { // new query, produce hypergraph, apply irff
			prev_qid = TD::Convert(qid);
			cerr << "New Query: " << qid << endl;
			ranker.translate(qid, qstr);
			ranker.applyIndicator(); // might add unseen IR features to w_ir, creates w_ir_q_ in ranker instance
			cerr << "|W_IR|="<<w_ir.size() << endl;
		}
		if (TD::Convert(d_rel_id) != prev_d_rel_id) { // new relevant document, read it
			prev_d_rel_id = TD::Convert(d_rel_id);
			delete d_rel;
			d_rel = new CLIR::FDocument(d_rel_id, d_rel_len, extractor.convert(vutils::read_vector(d_rel_tfstr)));
			s_rel  = log( ranker.score(*d_rel, f_rel) ); // score relevant document and get feature vector
		}
		// read irrelevant document
		CLIR::FDocument d_irel(d_irel_id, d_irel_len, extractor.convert(vutils::read_vector(d_irel_tfstr)));
		s_irel = log( ranker.score(d_irel, f_irel) ); // score irrelevant document and get feature vector

		cerr << "D+: " << d_rel_id << " ||| D-: " << d_irel_id << endl;

		FeatureVector gradient = (f_rel - f_irel).erase_zeros();
		double prediction = s_rel - s_irel;

		cerr << "Q(" << qid << ") ||| D+(" << d_rel_id << "): " << s_rel 
		     << " ||| D-(" << d_irel_id << "):" << s_irel << " ||| Desired Margin: " 
		     << margin << " ||| My Prediction: " << prediction << " ||| ";

		if (s_rel > s_irel) cco++; // D+ > D-?
		if (prediction < margin) cne++;
		cerr << "Scores: D+=" <<s_rel<<" ||| D-="<<s_irel<<endl;
		cerr << "Desired Margin: " << margin << " ||| Actual Margin: " << prediction << endl ;
		// VW exampke format: <margin> <importance weight> | <features>
		cout << margin << " 1.0 " << qid<<"-"<<d_rel_id<<"-"<<d_irel_id<<"| " << ConvertToVWVector(gradient) << endl;
		cp++;
	}

	cerr << "# of total pairs seen: " << cp << endl
		 << "# of pairs correctly ordered: " << cco << " (" << (cco/(double)cp)*100 << "%)" << endl
		 << "# of pairs to learn from: " << cne << " (" << (cne/(double)cp)*100 << "%)" << endl;

	// write vw initvector (lame but necessary)
	writeWeights(cfg["model"].as<string>(), w_smt+w_ir);

	delete decoder;
	delete d_rel;
}

