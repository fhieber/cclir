#!/bin/bash

CDEC=/toolbox/cdec_fh
TREC=/toolbox/trec_eval
PRES=/toolbox/PRESeval.pl
CLIRCLIENT=$CDEC/vector-projection/clir-client

USAGE="$0 <qrels> <host:port [host:port host:port]>  < translated queries"

if [[ $# -lt 2 ]]; then
echo $USAGE
exit 1
fi

QRELS=$1
SERVERS=${@:2}

tmp=`mktemp`
tmp2=`mktemp`
# read queries from stdin
$CLIRCLIENT -q - -p -k 1000 -r 0 --servers $SERVERS > $tmp
# EVAL
avglen=`cat $tmp2 | grep "average doc length" | cut -f1`
qno=`cat $tmp2 | grep "lines read/written." | cut -f1`
echo "# of queries	$qno"
echo "avg. query length	$avglen"
$TREC -m ndcg -m map -m num_rel_ret -m P.5,100,1000 -m recall.5,100,1000 $QRELS $tmp | sed 's/  */\ /g' | cut -f1,3
$TREC -m ndcg.2=0,1=0 $QRELS $tmp | sed 's/  */\ /g' | cut -f1,3
$TREC -m ndcg.3=0,1=0 $QRELS $tmp | sed 's/  */\ /g' | cut -f1,3
$TREC -m ndcg.3=0,2=0 $QRELS $tmp | sed 's/  */\ /g' | cut -f1,3
$PRES $QRELS $tmp | grep "^PRES" | cut -d" " -f1,3
rm $tmp $tmp2
