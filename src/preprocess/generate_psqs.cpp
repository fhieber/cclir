#include "generate_psqs.h"

/*
 * loads and parses commandline parameters
 */
bool init_params(int argc, char** argv, po::variables_map* cfg) {
	po::options_description cl("\nCreates Probabilistic Structured Queries (PSQs) from source input (translates using cdec).\nCommand Line Options");
	cl.add_options()
			("input,i",				po::value<string>()->default_value("-"), "input file. Use '-' for STDIN.")
			("decoder_config,c",	po::value<string>(),					"* decoder config for cdec.")
			("weights,w",			po::value<string>(),					"decoder weights. overwrites weights specified in the decoder config.")
			("kbest,k", po::value<int>()->default_value(1000), "nbest list size for nbest ttable estimation")
			("unique_kbest,r", po::value<bool>()->zero_tokens(), "use unique nbest list")
			("ignore_derivation_scores", po::value<bool>()->default_value(false), "ignore derivation scores for nbest ttable")
			("add_passthrough_rules", po::value<bool>()->default_value(true), "add passthrough translation for nbest ttable")
			("lex",					po::value<string>(),					"* lexical translation table for interpolation.")
			("lambda,l",			po::value<double>()->default_value(0.6,"0.6"),"interpolation coefficient for option tables.")
			("output,o",			po::value<string>()->default_value("-"), "output file for the (interpolated) Probabilistic Structured Queries. Use '-' for STDOUT.")
			("LOWER,L",				po::value<double>()->default_value(0.005,"0.005"),"lower bound on interpolated weights")
			("CUMULATIVE,C",		po::value<double>()->default_value(0.95,"0.95"),"cumulative upper bound on interpolated weights")
			("source-stopwords,s",	po::value<string>(),					"stopword file for query (source) language")
			("target-stopwords",	po::value<string>(),					"stopword file for document (target) language")
			("tf_vectors,t",		po::value<string>(),					"output file for the source term frequency vectors. Use '-' for STDOUT.")
			("dftable,d",			po::value<string>(),					"output target for DF table (optional).")
			("verbose,v",			po::value<bool>()->zero_tokens(),		"verbose output to STDERR.")
			("help,h",				po::value<bool>()->zero_tokens(),		"print this help message.");

	po::store(parse_command_line(argc, argv, cl), *cfg);
	po::notify(*cfg);

	if (cfg->count("help")) {
		cerr << cl << endl;
		return false;
	}

	if (cfg->at("lambda").as<double>() > 0.0 && !cfg->count("decoder_config")) {
		cerr << cl << endl << "\nI require --decoder_config!\n";
		return false;
	}

	if (cfg->at("lambda").as<double>() < 1.0 && !cfg->count("lex")) {
		cerr << cl << endl << "\nI require --lex!\n";
		return false;
	}

	return true;
}

int main(int argc, char** argv) {

	// handle parameters
	po::variables_map cfg;
	if (!init_params(argc,argv, &cfg)) exit(1); // something is wrong

	ReadFile input(cfg["input"].as<string>());
	WriteFile psq_out(cfg["output"].as<string>());
	WriteFile tf_out;
	if (cfg.count("tf_vectors"))
		tf_out = WriteFile(cfg["tf_vectors"].as<string>());

	// load lexical ttable
	LexicalTTable lex;
	if (cfg.count("lex") && cfg["lambda"].as<double>() < 1.0) {
		lex = LexicalTTable(cfg["lex"].as<string>());
		cerr << lex.size() << " f types in lexical ttable loaded from '"
			 << cfg["lex"].as<string>() << "'" << endl;
	}

	// setup decoder
	Decoder* decoder = NULL;
	if (cfg.count("decoder_config") && cfg["lambda"].as<double>() > 0.0) {
		decoder = SetupDecoder(cfg);
	}

	// setup decoder observer
	NbestTTableGetter* observer = new NbestTTableGetter(
									cfg["kbest"].as<int>(),
									cfg["ignore_derivation_scores"].as<bool>(),
									cfg["add_passthrough_rules"].as<bool>(),
									prob_t(cfg["LOWER"].as<double>()),
									prob_t(cfg["CUMULATIVE"].as<double>()),
									cfg.count("unique_kbest"),
									cfg.count("target-stopwords")
									);

	// load stopwords
	if (cfg.count("source-stopwords")) {
		sw::loadStopwords(cfg["source-stopwords"].as<string>());
		cerr << sw::stopwordCount() << " source language stopwords loaded from '"
			 << cfg["source-stopwords"].as<string>() << "'" << endl;
	}
	if (cfg.count("target-stopwords")) {
		sw::loadStopwords(cfg["target-stopwords"].as<string>());
		cerr << sw::stopwordCount() << " target language stopwords loaded from '"
			 << cfg["target-stopwords"].as<string>() << "'" << endl;
	}

	if (cfg.count("source-stopwords") && cfg.count("target-stopwords"))
		cerr << "WARNING: loaded source and target language stopwords into one set!" << endl;


	int c = 0;
	int empty_vecs = 0;
	int empty_queries = 0;

	string docid, text;
	DfTable dft;
	prob_t L = prob_t(cfg["LOWER"].as<double>());
	prob_t C = prob_t(cfg["CUMULATIVE"].as<double>());

	/*
	 * FOR EACH DOCUMENT
	 */
	while(*input >> docid) {
		input->ignore(1, '\t');
		getline(*input, text);

		if (cfg.count("verbose"))
			cerr << "ID: " << docid << "\t" << "TEXT: " << text << endl;

		if (docid.size() == 0 || text.size() == 0)
			continue;

		Document d(text, docid);
		if (!d.parsed_) {
			cerr << "could not parse input!" << endl;
			*psq_out << "\n";
			psq_out->flush();
			continue;
		}

		// obtain source tf vec
		d.computeTfVector(cfg.count("source-stopwords"));
		if ( d.v_.size() == 0 ) {
			cerr << "WARNING: tf-vector for docid " << TD::Convert(d.id_) << " is empty!" << endl;
			empty_vecs++;
		}

		Query query(true, d.s_count, TD::Convert(d.id_)); // new PSQ query with s_count NbestTTables
		// obtain translations from decoder
		if ( cfg.count("decoder_config") && cfg["lambda"].as<double>() > 0.0 ) {
			for (int s = 0 ; s < d.s_count ; ++s) {
				decoder->Decode( d.sentence(s, true), observer ); // decode document (single sentence)
				query.set(s, observer->GetNbestTTable());
				if (cfg.count("verbose")) cerr << query.nbt(s).as_string() << endl;
			}
		}

		// interpolate with lexical ttable
		if ( cfg.count("lex") && cfg["lambda"].as<double>() < 1.0 ) {
			NbestTTable lex_t(lex, d.v_);
			query.interpolate(lex_t, cfg["lambda"].as<double>());
		}

		// constrain query
		query.constrain(L, C, cfg.count("target-stopwords"));

		// set tfvec
		query.set(d.v_);
		
		if (query.empty()) ++empty_queries;

		// write query to output file
		query.save(*psq_out);
		*psq_out << "\n";
		psq_out->flush();

		// write results to outputs
		if (cfg.count("tf_vectors")) {
			*tf_out << TD::Convert(d.id_) << "\t" << d.len_ << "\t";
			vutils::write_vector(d.v_, *tf_out);
			*tf_out << endl;
		}

		// write types to df table if wanted
		if (cfg.count("dftable")) {
			dft.update(d.v_);
		}

		c++;
		c%1000==0 ? cerr << "." << c << "." : cerr ;

	}
	/*
	 * END FOR EACH DOCUMENT
	 */

	// we are done using the decoder. destroy it.
	delete decoder;
	delete observer;

	// if a df table should be written to disk:
	if (cfg.count("dftable"))
		dft.writeToFile(cfg["dftable"].as<string>());

	cerr << "done." << endl << endl
		 << c << "\tlines read/written." << endl
		 << empty_vecs << "\ttf vectors empty." << endl
		 << empty_queries << "\tqueries empty." << endl;

	return 0;
}

