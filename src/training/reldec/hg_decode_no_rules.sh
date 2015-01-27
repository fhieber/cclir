#!/bin/bash


# decodes in parallel and saves hypergraphs to specified folder. Does not support a second set of weights


PARALLELIZE=/toolbox/cdec_fh/training/utils/parallelize.pl
CDEC=/toolbox/cdec_fh


USAGE="$0 <queries> <cdec.ini> <weights> <hg output folder> <jobs>"

INPUT=$1
INI=$2
if [[ $# == 5 ]]; then
	W=$3
	HGO="$4"
	J=$5
else
	echo $USAGE
	exit 1
fi

if [[ -d $HGO ]]; then
	echo "hg output folder exists!"
	exit 1
fi

mkdir -p $HGO
tmp=`mktemp -d`
paste <(cut -f1 $INPUT) <(cut -f2 $INPUT | $PARALLELIZE --workdir $tmp --use-fork -e $tmp -j $J --no-which "$CDEC/decoder/cdec -c $INI -w $W -O $HGO --forest_output_no_rules" 2> /dev/null ) 
rm -rf $tmp
