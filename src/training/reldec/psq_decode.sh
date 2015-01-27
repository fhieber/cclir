#!/bin/bash

PARALLELIZE=/toolbox/cdec_fh/training/utils/parallelize.pl
CDEC=/toolbox/cdec_fh
JOIN=/workspace/reldec/data/join-jp-xml.py

USAGE="$0 <queries> <cdec.ini> <weights> <jobs>"

INPUT=$1
INI=$2
if [[ $# == 4 ]]; then
W=$3
J=$4
else
echo $USAGE
exit 1
fi

tmp=`mktemp -d`
cat $INPUT | $JOIN | $PARALLELIZE --workdir $tmp --use-fork -e $tmp -j $J --no-which "$CDEC/vector-projection/get-psqs -i - -c $INI -w $W -k 200 -r -l 1.0 -o -" 2> /dev/null
rm -rf $tmp
