#!/usr/bin/python

from collections import defaultdict
import sys
import random
from itertools import groupby
from operator import itemgetter

INFO = """
samples training examples for the forced decoding ranking algorithm.
For each query, for each relevant document, it samples N irrelevant documents and creates triples: Q,D+,D-.
Output format is:
QID TAB #SENTENCES TAB S1 TAB S2 TAB [...] MARGIN TAB D+ID TAB D+LEN TAB D+TF TAB D-ID TAB D-LEN D-TF
"""


def loadQrels(fname):
	QRELS = defaultdict(dict)
	for line in open(fname):
		qid, _, did, rl = line.strip().split('\t',3)
		QRELS[qid][did] = int(rl)
	return QRELS

def loadQueries(fname):
	Q = defaultdict(list)
	for line in open(fname):
		qid, q = line.strip().split('\t', 1)
		Q[qid].append(q)
	return dict(Q)

def loadDocs(fname):
	D = {}
	for line in open(fname):
		did, dlen, dcbm25 = line.strip().split('\t', 2)
		D[did] = (dlen, dcbm25)
	return D

def read_input(fd):
	for line in fd:
		yield line.strip().split("\t",1)


import argparse
parser = argparse.ArgumentParser(description=INFO)
parser.add_argument('-qrels', required=True, help='qrels')
parser.add_argument('-ir', required=True, type=int, help='number of irrelevant docs to sample per doc')
parser.add_argument('-only_level', type=int, default=None, help='use only minimum relevance level x for positive pairs')
parser.add_argument('-max_rel', type=int, default=None, help='use a maximum number of relevant documents per query')
parser.add_argument('-documents', required=True, help='document .cbm25 file')
parser.add_argument('-margin', type=int, required=True, help='minimum margin for pair', default=3)
args = parser.parse_args(sys.argv[1:])

qrels =  loadQrels(args.qrels)
N = int(args.ir) # number of samples per relevant document
D = loadDocs(args.documents)
Dids = D.keys()
MINMARGIN = int(args.margin)
rl_filter = args.only_level
sys.stderr.write("Sampling pairs with minimum margin %d\n"%MINMARGIN)

i = 0
for qid, group in groupby(read_input(sys.stdin), itemgetter(0)):
	if i%1000==0: sys.stderr.write("%d."%i)
	i += 1
	# determine # of query sentences
	q = [q for _, q in group]
	qlen = len(q) # number of query sentences
	q = "\t".join(q)

	# sampler
	if qrels.get(qid, None):
		rels_used = 0
		for d_rel in qrels[qid]:
			rl = qrels[qid][d_rel]

			if args.max_rel and rels_used == args.max_rel:
				break
				
			if rl_filter and rl < rl_filter:
				continue
			d_irels = []

			# sample
			while True:
				d_irel = random.choice(Dids)
				rl_irel = qrels[qid].get(d_irel,0)
				margin = rl - rl_irel
				if margin < min([MINMARGIN,rl]): continue # cannot sample a pair with MINMARGIN when rl is below MINMARGIN
				d_irels.append( (d_irel,margin) )
				if len(d_irels) == N: break;

			rels_used += 1

			# get strings
			d_rel_len, d_rel_tf = D.get(d_rel,(None,None))
			if not d_rel_len: continue;
			for d_irel, margin in d_irels:
				d_irel_len,d_irel_tf = D.get(d_irel,(None,None))
				if not d_irel_len: continue;

				# write output:
				sys.stdout.write(qid + "\t" + str(qlen) + "\t" + q + "\t" + str(margin) + "\t" + d_rel + "\t" + d_rel_len + "\t" + d_rel_tf + "\t" + d_irel + "\t" + d_irel_len + "\t" + d_irel_tf + "\n")
sys.stderr.write("\ndone.\n")			
