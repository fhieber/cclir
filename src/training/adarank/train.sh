#!/bin/bash

if [[ $# != 1 ]]; then
	echo "USAGE train.sh <ini file>"
	exit 1
fi

if [[ ! -f $1 ]]; then
	echo "ini file does not exist!"
	exit 1
fi

source $1

# file checks
if [[ ! -f $INIT_WEIGHTS || ! -f $SOURCE_QUERIES || ! -f $TREC_RELS || ! -f $DOCUMENT_VECTORS || ! -f $DFTABLE || ! -f $CDEC_INI ]]; then
	echo "one ore more files given in the .ini file are missing!"
	exit 1
fi
if [[ -d $WORKDIR ]]; then
	echo "workdir '$WORKDIR' exists!"
	exit 1
fi

# binary folder
BINDIR=/toolbox/cdec_fh/training/clirtrain

# create workdir
echo "creating and entering '$WORKDIR'..."
mkdir $WORKDIR

# getting files
cp $CDEC_INI $WORKDIR/cdec.ini
cp $INIT_WEIGHTS $WORKDIR/weights.0
ln -s $SOURCE_QUERIES $WORKDIR/queries
cp $TREC_RELS $WORKDIR/rels.trec
ln -s $DOCUMENT_VECTORS $WORKDIR/documents.tf
ln -s $DFTABLE $WORKDIR/documents.df

# switches
UNIQUE_STR=
if [[ $UNIQUE_KBEST == 1 ]]; then
	UNIQUE_STR="-r"
fi
SWF_STR=
if [[ $TARGET_SWF == 1 ]]; then
	SWF_STR="-s $TRG_STOPWORDS"
fi


echo "CLIRTRAIN - start $(date)"

cd $WORKDIR

# creating decoder config TODO


:> err.log
:> out.log

for E in `seq 1 $EPOCHS`;
do

echo "Epoch $E - start $(date)"

	PREV_E=$(($E-1)) # previous epoch
	
	echo " generating instances ..."
	$BINDIR/generate-instances -i queries -o $E.instances -w weights.$PREV_E -c cdec.ini -k $K --sample_from $DERIVATIONS_FROM $UNIQUE_STR $SWF_STR --chunksize $CHUNKSIZE -L $LOWER -C $CUMULATIVE --rels rels.trec >> out.log 2>> err.log
	echo " running retrieval ..."
	$BINDIR/retrieval -q $E.instances -c documents.tf -d documents.df -o $E.clir -k $TOPK -n $N -a $AVG_LEN -r $E >> out.log 2>> err.log
	echo " getting ir evaluation ..."
	$BINDIR/trec_eval_instances.sh $E.clir $TREC_SCORE  >> out.log 2>> err.log
	echo " setting gold permutations ..."
	$BINDIR/set-gold-permutation -q $E.instances -y $E.clir  >> out.log 2>> err.log
	echo " optimizing ..."
	$BINDIR/optimize -i $E.instances.labels -v -w weights.$PREV_E -l $LOSSFUNC -o weights.$E -t $MAX_ITER -e $EPS > losses.$E 2>> err.log 

	echo "Epoch $E - end $(date)"

done

echo "CLIRTRAING - end $(date)"

exit 0
