/*
 * generate-instances.cpp
 *
 *  Created on: Apr 20, 2013
 */

#include "generate-instances.h"

int main(int argc, char** argv) {
	// handle parameters
	po::variables_map cfg;
	if (!init_params(argc,argv, &cfg)) exit(1); // something is wrong

	// setup decoder
	Decoder* decoder = SetupDecoder(cfg);

	// init weights
	DenseWeightVector& dense_weights = decoder->CurrentWeightVector();
	cerr << " \nweights in cdec:\n";
	for (int i = 1 ; i < dense_weights.size(); ++i)
		cerr << "  " << FD::Convert(i) << " " << dense_weights[i] << endl;
	cerr << "\n";
	WeightVector weights;
	Weights::InitSparseVector(dense_weights, &weights);

	// load stopwords
	if (cfg.count("target-stopwords")) {
		sw::loadStopwords(cfg["target-stopwords"].as<string>());
		cerr << sw::stopwordCount() << " target language stopwords loaded from '"
			 << cfg["target-stopwords"].as<string>() << "'" << endl;
	}

	// setup decoder observer
	InstanceGetter* observer;
	if (cfg["sample_from"].as<string>() == "kbest") {
		observer = dynamic_cast<KbestInstanceGetter*>(new KbestInstanceGetter(
				cfg["k"].as<int>(),
				cfg.count("unique_k_best"),
				cfg.count("ignore_derivation_scores"),
				cfg.count("target-stopwords"),
				cfg["chunksize"].as<int>(),
				prob_t(cfg["LOWER"].as<double>()),
				prob_t(cfg["CUMULATIVE"].as<double>()),
				weights));
	} else if (cfg["sample_from"].as<string>() == "forest") {
		MT19937 rng; // only for forest sampling
		observer = dynamic_cast<ForestSampleInstanceGetter*>( new ForestSampleInstanceGetter(
				cfg["k"].as<int>(),
				&rng,
				cfg.count("ignore_derivation_scores"),
				cfg.count("target-stopwords"),
				cfg["chunksize"].as<int>(),
				prob_t(cfg["LOWER"].as<double>()),
				prob_t(cfg["CUMULATIVE"].as<double>()),
				weights));
	} else {
		cerr << "use either 'kbest' or 'forest' !\n";
		exit(1);
	}

	cerr << cfg["k"].as<int>() << " derivations from " << cfg["sample_from"].as<string>() << " per instance.\n";

	// setup input
	ReadFile input(cfg["input"].as<string>());

	// load relevance file
	boost::unordered_map<string,Relevance> trec_rels;
	if (cfg.count("rels"))
		loadTrecRels(cfg["rels"].as<string>(), trec_rels);

	vector<Instance> instances;
	string input_line, raw;
	string qid;
	int c = 0;
	cerr << "\ndecoding";
	while(getline(*input, input_line)) {
		stringstream qss(input_line);
		qss >> qid;
		qss.ignore(1, '\t');
		getline(qss,raw);
		Document d(raw, qid);
		if (!d.parsed_) {
			cerr << "could not parse input!" << endl;
			continue;
		}
		assert(d.s_count == 1); // we can only optimize for single-sentence queries!
		decoder->Decode(d.sentence(0,true), observer);
		Instance& i = observer->GetTrainingInstance();
		if (i.size == 0) {
			cerr << "  Warning: query " << qid << " produced empty Instance!" << endl;
			continue;
		}
		i.SetID(qid);
		i.SetRelevance(trec_rels[qid]);
		if (cfg.count("verbose"))
			cerr << i.AsString() << endl;
		
		instances.push_back(i);
		
		if (c%500 == 0) cerr << ".";

	}
	cerr << "ok.\n\n";

	delete decoder;
	delete observer;

	if (cfg.count("ascii"))
		saveInstancesText(cfg["output"].as<string>(), instances);
	else
		saveInstances(cfg["output"].as<string>(), instances);

}
