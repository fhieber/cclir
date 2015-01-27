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
if [[ ! -f $DECODE_DFT ]]; then echo "$DECODE_DFT doesnt exist!"; exit 1; fi

mkdir -p $WORKDIR
cp $INIT_WEIGHTS $WORKDIR/0.w
cp $RELEVANCE_WEIGHTS $WORKDIR/weights.relevance
cp $INPUT $WORKDIR/decode.input
cd $WORKDIR

# make cdec ini
:> cdec.ini
echo "formalism=scfg" >> cdec.ini
echo "add_pass_through_rules=true" >> cdec.ini
echo "scfg_max_span_limit=15" >> cdec.ini
if [[ $PRUNE -eq 1 ]]; then
	echo "intersection_strategy=cube_pruning" >> cdec.ini
	echo "cubepruning_pop_limit=200" >> cdec.ini
	echo "feature_function=KLanguageModel /workspace/reldec/data/smt/ntc7.lm.ken.5" >> cdec.ini
else
	echo "intersection_strategy=full" >> cdec.ini
fi
echo "feature_function=WordPenalty" >> cdec.ini
echo "feature_function=ArityPenalty" >> cdec.ini

# sparse features
if [[ $SPARSE -eq 1 ]]; then
echo "feature_function=RuleIdentityFeatures" >> cdec.ini
echo "feature_function=RuleShape" >> cdec.ini
echo "feature_function=RuleSourceBigramFeatures" >> cdec.ini
echo "feature_function=RuleTargetBigramFeatures" >> cdec.ini
fi

for e in `seq 1 $epochs`; do
	if [[ $e == 1 ]]; then
		W=0.w
	fi
	echo "EPOCH=$e"

	# decode with current weights for RS1, do RS2 with Relevance feature to annotate the hypergraphs
	CDECCMD="$CDEC/decoder/cdec -c cdec.ini -w $W --feature_function2 Relevance --intersection_strategy2=full -O $e.hgs"
	if [[ $DF -eq 1 ]]; then
		CDECCMD="$CDECCMD --feature_function2 'DocumentFrequency $DECODE_DFT $DECODE_N'"
	fi
	echo "- getting hypergraphs with command '$CDECCMD'..."
	mkdir $e.decode.tmp $e.hgs
	cut -f2 decode.input | $PARALLELIZE --workdir $e.decode.tmp --use-fork -e $e.decode.logs -j $DECODE_JOBS --no-which "$CDECCMD" > $e.transl1 2> $e.decode.err
	echo "- optimizing w/ ssvm..."
	$SSVM \
		-i $e.hgs/ \
		--weights $W \
		--rweights weights.relevance \
		--iters $SSVM_ITERS \
		-n $SSVM_LEARNING_RATE \
		-l $SSVM_LAMBDA $SSVM_PARAMS \
		-o $e.w \
		-j $SSVM_JOBS \
		-e 10 \
		-d 100 \
		-v \ 
		> $e.ssvm.loss 2> $e.ssvm.err
	
	# set w
	W=$e.w
done
