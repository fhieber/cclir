#!/usr/bin/python

import sys,argparse,math,os,random
import logging, operator
from collections import defaultdict, Counter

def batch_gen(data, batch_size):
	for i in range(0, len(data), batch_size):
		yield data[i:i+batch_size]

def parseVector(s):
	V = {}
	for t in batch_gen(s.split(),2):
		if len(t) != 2:
			sys.stderr.write(s + "\n")
			continue
		if t[1].startswith("(-)"):
			V[ t[0] ] = math.exp(float(t[1][3:])) * -1
		else:
			V[ t[0] ] = math.exp(float(t[1]))
	return V

def loadDocumentVectors(fname):
	with open(fname) as f:
		D = {}
		for line in f:
			did, _, dvec = line.strip().split("\t") # dont need length
			D[did] = parseVector(dvec)
	return D

def loadRels(fname):
	with open(fname) as f:
		R = defaultdict(list)		
		for line in f:
			qid, _, did, rl = line.strip().split("\t")
			R[qid].append( (did,rl) )
		f.close()
	return dict(R)

def loadRelsByLevel(fname):
	with open(fname) as f:
		R = defaultdict(set)
		for line in f:
			qid, _, did, rl = line.strip().split("\t")
			R[rl].add( did )
		f.close()
	return dict(R)

def sampleIr(qid, D, R, sample):
	"""
	samples irrelevant document bm25 values for a given qid.
	BM@5 values are normalized by sample_size.
	"""
	drel = set([x[0] for x in R.get(qid,[])]) # relevant dids for qid
	IRR = {}
	S = set(random.sample(D.keys(), sample)) - drel
	for s in S:
		for t,val in D[s].iteritems():
			IRR[t] = IRR.get(t,0.0) + val
	# normalize by sample_size
	return {key: value/float(sample) for (key, value) in IRR.iteritems()}

def relevances(qid, D, R, args):
	"""
	compute (normalized) bm25 values for separate relevance levels
	"""
	vals = defaultdict(Counter)
	rlcounts = defaultdict(int) 
	for did, rl in R.get(qid,[]): # for each document and its rl given the qid
		rlcounts[rl] += 1
		for term, bm25 in D.get(did,{}).iteritems(): # sum up bm25 values for each term in the document
			vals[rl][term] += bm25
	rlcounts = dict(rlcounts)
	vals = dict(vals)
	# normalize bm25 vals by number of relevant documents in the relevance level
	for rl, bm25s in vals.iteritems():
		l = [ ( term, value/float(rlcounts[rl]) ) for term, value in list(bm25s.iteritems()) ]
		l.sort(key=operator.itemgetter(1), reverse=True)
		if args.cutoff:
			yield rl, l[ : int(math.ceil(len(l)/args.cutoff)) ]
		else:
			yield rl, l 
	return

def process(line, D, R, O, args):
	qid, q = line.strip().split("\t")
	out = ""
	for rl, bm25s in relevances(qid, D, R, args):
		out += rl + "\t"
		for term, bm25 in bm25s:
			if bm25 > 1e-07:
				out += "%s %.7f "%(term,bm25)
		out += '\n'
	if args.sample > 0:
		out += "-1\t"
		for term,bm25 in sampleIr(qid, D, R, args.sample).iteritems():
			out += "%s %.7f "%(term,bm25)
		out += '\n'
	return qid, qid + "\t" + q.replace('<seg grammar', '<seg rel="%s" grammar'%os.path.join(O, qid)), out

def createCommon(args, D, R):
	for rl, docidset in R.iteritems():
		v = defaultdict(int)
		for docid in docidset:
			for term, bm25 in D.get(docid,{}).iteritems():
				v[term] += bm25
		s = float(len(docidset))
		l = [ (term, value/s) for term, value in list(dict(v).iteritems()) ]
		l.sort(key=operator.itemgetter(1), reverse=True)
		sys.stdout.write(rl + "\t")
		for term, bm25 in l:
			sys.stdout.write("%s %.7f "%(term,bm25))
		sys.stdout.write("\n")
	return
	
def main():
	parser = argparse.ArgumentParser(description='Create relevance score files for each query from STDIN. Outputs updated query markup to STDOUT.')
	parser.add_argument('-r', '--rels', required=True, help='relevance judgements for queries')
	parser.add_argument('-d', '--docs', required=True, help='document bm25 vectors')
	parser.add_argument('-o', '--output', help='output directory for query specific files')
	parser.add_argument('-c', '--cutoff', type=float, help='use only top c percentage of bm25 values for each relevance level (focus on most important words)')
	parser.add_argument('-s', '--sample', type=int, default=0, help='sample S irrelevant document bm25 values for Relevance_-1')
	parser.add_argument('--common', action="store_true", help='do not use queries and write a single (big) file containing average bm25 values for each term in the document collection TO STDOUT(!)')
	args = parser.parse_args()


	if args.common: # create one big bm25 file
		D = loadDocumentVectors(args.docs)
		sys.stderr.write("%d document vectors loaded.\n"%len(D))
		R = loadRelsByLevel(args.rels)
		sys.stderr.write("picked up %d relevance levels from qrels.\n"%len(R))
		createCommon(args, D, R)
		
	else:
		if not args.output:
			sys.stderr.write("specify output directory!\n")
			sys.exit(1)
		# output dir
		if os.path.exists(args.output):
			sys.stderr.write("output dir exists!\n")
			sys.exit(1)
		os.makedirs(args.output)
		O = os.path.abspath(args.output)
	
		# rels
		R = loadRels(args.rels)
		sys.stderr.write("%d query relevance judgements loaded.\n"%len(R))
		# docs
		D = loadDocumentVectors(args.docs)
		sys.stderr.write("%d document vectors loaded.\n"%len(D))
	
		for line in sys.stdin:
			qid, qstr, relstr = process(line, D, R, O, args)
			with open( os.path.join(O, qid), "w") as out:
				out.write(relstr)
				out.close()
			sys.stdout.write(qstr + '\n')

if __name__=='__main__': main()

