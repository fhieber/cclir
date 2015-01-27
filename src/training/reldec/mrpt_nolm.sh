#!/bin/bash

PARALLELIZE=/toolbox/cdec_fh/training/utils/parallelize.pl
CDEC=/toolbox/cdec_fh
SSVM=$CDEC/training/reltrain/ssvm
GETTF=$CDEC/vector-projection/get-tf-vectors
CLIRSERVER=$CDEC/vector-projection/clir-server
CLIRCLIENT=$CDEC/vector-projection/clir-client
TREC=/toolbox/trec_eval
JOIN=/workspace/reldec/data/join-jp.py

if [[ $# != 1 ]]; then
	echo "USAGE: mrpt.sh <settings.ini>"
	exit 1
fi
if [[ ! -f $1 ]]; then
	echo "$1 does not exist!"
	exit 1
fi

source $1

if [[ -d $WORKDIR ]]; then
	echo "$WORKDIR exists"
	exit 1
fi

if [[ ! -f $INIT_WEIGHTS ]]; then echo "$INIT_WEIGHTS doesnt exist!"; exit 1; fi
if [[ ! -f $RELEVANCE_WEIGHTS ]]; then echo "$RELEVANCE_WEIGHTS doesnt exist!"; exit 1; fi
if [[ ! -f $INPUT ]]; then echo "$INPUT doesnt exist!"; exit 1; fi
if [[ ! -f $DOCS ]]; then echo "$DOCS doesnt exist!"; exit 1; fi
if [[ ! -f $QUERIES ]]; then echo "$QUERIES doesnt exist!"; exit 1; fi
if [[ ! -f $DFT ]]; then echo "$DFT doesnt exist!"; exit 1; fi
if [[ ! -f $QREL ]]; then echo "$QREL doesnt exist!"; exit 1; fi

mkdir -p $WORKDIR
cd $WORKDIR

# make cdec ini
:> cdec.ini
echo "formalism=scfg" >> cdec.ini
echo "add_pass_through_rules=true" >> cdec.ini
echo "scfg_max_span_limit=15" >> cdec.ini
#echo "intersection_strategy=cube_pruning" >> cdec.ini
#echo "cubepruning_pop_limit=200" >> cdec.ini
echo "feature_function=WordPenalty" >> cdec.ini
echo "feature_function=ArityPenalty" >> cdec.ini
#echo "feature_function=KLanguageModel /workspace/reldec/data/smt/ntc7.lm.ken.5" >> cdec.ini

# sparse features
if [[ $SPARSE -eq 1 ]]; then
echo "feature_function=RuleIdentityFeatures" >> cdec.ini
echo "feature_function=RuleShape" >> cdec.ini
echo "feature_function=RuleSourceBigramFeatures" >> cdec.ini
echo "feature_function=RuleTargetBigramFeatures" >> cdec.ini
fi

# start clir server
$CLIRSERVER -c $DOCS -d $DFT -j $CLIR_JOBS --port $PORT 2> server.err > server.out &
CLIRSERVER_PID=$!

for e in `seq 1 $epochs`; do
	if [[ $e == 1 ]]; then
		W=$INIT_WEIGHTS
	fi
	echo "EPOCH=$e"
	echo "- getting hypergraphs..."
	mkdir $e.decode.tmp
	mkdir $e.hgs
	cut -f2 $INPUT | \
	$PARALLELIZE --workdir $e.decode.tmp --use-fork -e $e.decode.logs -j $DECODE_JOBS --no-which \
		"$CDEC/decoder/cdec -c cdec.ini -w $W --feature_function Relevance --feature_function 'DocumentFrequency $DECODE_DFT $DECODE_N' -O $e.hgs" \
		> $e.transl1 2> $e.decode.err
	#rm -rf $e.decode.tmp $e.decode.logs
	echo "- optimizing w/ ssvm..."
	$SSVM -i $e.hgs/ --weights $W --rweights $RELEVANCE_WEIGHTS --iters $SSVM_ITERS -n $SSVM_LEARNING_RATE -l $SSVM_LAMBDA $SSVM_PARAMS -o $e.w -j $SSVM_JOBS -v > $e.ssvm.loss 2> $e.ssvm.err
	# set w
	W=$e.w
	echo "- running retrieval..."
	mkdir -p $e.tmp
	paste <(cut -f1 $QUERIES) <(cut -f2 $QUERIES | $PARALLELIZE --workdir $e.tmp --use-fork -e $e.logs -j $DECODE_JOBS --no-which \
	"$CDEC/decoder/cdec -c cdec.ini -w $W" 2>> /dev/null) | tee $e.transl2 | $JOIN | $GETTF 2> $e.tf.err | tee $e.tf | $CLIRCLIENT -q - -k $K -r $e --port $PORT > $e.scores 2> client.err
	rm -rf $e.tmp $e.logs 
	/toolbox/trec_eval -m all_trec $QREL $e.scores > $e.eval
	grep "^ndcg " $e.eval
	grep "^map " $e.eval

done
# stopping clirserver
kill $CLIRSERVER_PID

#--feature_function2 'DocumentFrequency /workspace/reldec/dff/dev.docs.stop-small.df 100000'
#--forest_output_no_rules
#--weights2 $RELEVANCE_WEIGHTS
