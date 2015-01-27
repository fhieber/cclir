#!/bin/bash

PARALLELIZE=/toolbox/cdec_fh/training/utils/parallelize.pl
CDEC=/toolbox/cdec_fh


USAGE="$0 <queries> <cdec.ini> <weights> [<weights2>] <jobs>"

INPUT=$1
INI=$2
if [[ $# == 4 ]]; then
W=$3
W2=""
J=$4
elif [[ $# == 5 ]]; then
W=$3
W2="--weights2 $4"
J=$5
else
echo $USAGE
exit 1
fi

tmp=`mktemp -d`
paste <(cut -f1 $INPUT) <(cut -f2 $INPUT | $PARALLELIZE --workdir $tmp --use-fork -e $tmp -j $J --no-which "$CDEC/decoder/cdec -c $INI -w $W $W2" 2> /dev/null ) 
rm -rf $tmp
