#!/bin/bash

PARALLELIZE=/toolbox/cdec_fh/training/utils/parallelize.pl
CDEC=/toolbox/cdec_fh
GRIDSEARCH=$CDEC/training/reltrain/grid_search
GETTF=$CDEC/vector-projection/get-tf-vectors
CLIRSERVER=$CDEC/vector-projection/clir-server
CLIRCLIENT=$CDEC/vector-projection/clir-client
TREC=/toolbox/trec_eval
JOIN=/workspace/reldec/data/join-jp.py

if [[ $# != 1 ]]; then
	echo "USAGE: find-oracles.sh <settings.ini>"
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


mkdir -p $WORKDIR
cd $WORKDIR

# make cdec ini
:> cdec.ini
echo "formalism=scfg" >> cdec.ini
echo "add_pass_through_rules=true" >> cdec.ini
echo "scfg_max_span_limit=15" >> cdec.ini
if [[ $PRUNE -eq 1 ]]; then # use mert features
	echo "intersection_strategy=cube_pruning" >> cdec.ini
	echo "cubepruning_pop_limit=200" >> cdec.ini
	echo "feature_function=KLanguageModel /workspace/reldec/data/smt/ntc7.lm.ken.5" >> cdec.ini
	echo "feature_function=WordPenalty" >> cdec.ini
	echo "feature_function=ArityPenalty" >> cdec.ini
else
	echo "intersection_strategy=full" >> cdec.ini
fi
echo "feature_function2=Relevance" >> cdec.ini
echo "feature_function2=DocumentFrequency $DECODE_DFT $DECODE_N" >> cdec.ini


if [[ $PRUNE -eq 1 ]]; then
	CDECCMD="$CDEC/decoder/cdec -c cdec.ini -w $MERT_WEIGHTS -O hgs"
else
	CDECCMD="$CDEC/decoder/cdec -c cdec.ini -O hgs"
fi
mkdir decode.tmp hgs
cut -f2 $INPUT | $PARALLELIZE --workdir decode.tmp --use-fork -e decode.logs -j $DECODE_JOBS --no-which "$CDECCMD" > decode.out 2> decode.err
mkdir oracles
$GRIDSEARCH -i hgs/ -o oracles/o > gs.out 2> gs.err
for f in oracles/o*; do
	paste <(cut -f1 $INPUT) $f | $JOIN | $GETTF | $CLIRCLIENT -q - -k 1000 -r $f --servers $SERVER > $f.scores
	echo -n $f
	$TREC -m ndcg $QRELS $f.scores > $f.eval
	SCORE=`cat $f.eval`
	echo "$f $SCORE"
done
