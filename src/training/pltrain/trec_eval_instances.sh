#!/bin/bash

if [[ $# != 2 ]]; then
	echo "USAGE: trec_eval_instances.sh <inputdir> <measure>"
	exit 1
fi

BIN=/toolbox/trec_eval
INPUTDIR=$1
MEASURE=$2

cd $INPUTDIR
i=0
j=0
echo -n "trec_eval -m $MEASURE for all instances "
for f in *.ranks; do
	$BIN ${f%.*}.rels ${f%.*}.ranks -q -m $MEASURE | awk '$2 != "all"' | cut -f3 > ${f%.*}.labels & 
	j=$(( j+1 ))
	if [[ $j == 12 ]]; then
		wait
		j=0
	fi
	i=$(( i+1 ))
	x=$(($i % 500))
	if [[ $x == 0 ]]; then
		echo -n "."
	fi
done
echo
echo "ok ($i instances)."

wait
