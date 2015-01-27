#!/usr/bin/python

import sys,argparse,math,os,random
import logging, operator
from collections import defaultdict, Counter
from itertools import groupby
import mmap

TFT = Counter() # document tf table
DFT = Counter() # document df table
DOCS= {} # document tf vectors
DOCFILE=None
DOCMMAP=None
DOCFNAME=None
RELS= defaultdict(lambda : defaultdict(list)) # relevance judgements
TOTALTF = 0
AVGLEN = 0
N = 0

k1 = 1.2

def batch_gen(data, batch_size):
	for i in range(0, len(data), batch_size):
		yield data[i:i+batch_size]

def getDocumentFromMMap(docid):
	if DOCS and docid in DOCS: # in cache no need to access disk
		return DOCS[docid]
	else:
		global DOCFILE, DOCMMAP
		if not DOCMMAP:
			DOCFILE = open(DOCFNAME, "r+b")
			DOCMMAP = mmap.mmap(DOCFILE.fileno(),0)
		DOCMMAP.seek(0)
		curid = ""
		while curid != docid:
			try:
				curid, length, dvec = DOCFILE.readline().strip().split("\t",2)
			except ValueError:
				print DOCFILE.readline()
		if curid != docid: return None, None
		DOCS[docid] = (length, parseVector(dvec))
		return DOCS[docid] 
		

def parseVector(s):
	global TFT
	V = Counter()
	for t in batch_gen(s.split(),2):
		w, tf = t[0], round(math.exp(float(t[1])))
		V[ w ] = tf
		TFT[ w ] += tf
	return V

def loadDocumentVectors(fname):
	global TFT
	global DOCS
	global AVGLEN
	global TOTALTF
	global N
	with open(fname) as f:
		N = 0
		for line in f:
			did, length, dvec = line.strip().split("\t",2) # dont need length
			DOCS[did] = (length, parseVector(dvec)) # mapping did : (length,vector)
			TOTALTF += int(length)
			N += 1
		AVGLEN = TOTALTF/float(N)
	sys.stderr.write("TFT loaded, %d documents, %.4f avg. length\n"%(N,AVGLEN))
	assert AVGLEN > 0
	assert N > 0

def loadDFTable(fname):
	global DFT
	with open(fname) as f:
		for line in f:
			try:
				w, df = line.strip().split()
				DFT[w] += round(float(df))
			except ValueError:
				sys.stderr.write("DFT: Error loading %s"%line)
				continue
	sys.stderr.write("DFT loaded. %d unique tokens\n"%(len(DFT)))
	
def loadRels(fname):
	global RELS
	with open(fname) as f:
		for line in f:
			qid, _, did, rl = line.strip().split("\t")
			RELS[qid][int(rl)].append( did )
		f.close()
	sys.stderr.write("RELS loaded. %d queries\n"%(len(RELS)))

def readQueryInput(fd):
	for line in fd:
		qid, q = line.strip().split('\t',2)
		yield qid,q

def computeRelevantCounts(args, qid):
	rel_doc_ids = RELS.get(qid, {})
	fname = os.path.join(os.path.abspath(args.output), str(qid))
	out = open(fname, 'w+')
	for rel_level, rel_doc_ids in RELS.get(qid, {}).iteritems():
		tf_vector = Counter() # sum of tf values from relevant tf vectors
		df_vector = Counter() # df values for terms in relevant documents
		R = 0
		for doc_id in rel_doc_ids:
			length, doc = DOCS.get(doc_id, (None,None))
			if doc:
				R += 1
				tf_vector += doc
				df_vector += Counter(doc.iterkeys())
		tfsum = sum(tf_vector.values())
		dfsum = sum(df_vector.values())
		assert(len(tf_vector)==len(df_vector))
		# compute weights for each term in the relevant document set
		weights = {}
		for w in tf_vector.iterkeys():
			w_rtf = tf_vector[w]
			w_rdf = df_vector[w]
			w_tf  = TFT[w]
			w_df  = DFT[w]
			# rsj (complement method)
			assert R>0; assert N>0; assert w_rdf <= R
			rsj = math.log((w_rdf + .5)*(N - R - w_df + w_rdf + .5) / ((w_df - w_rdf + 0.5)*(R - w_rdf + 0.5)))
			if rsj < 0:
				val = rsj
			else:
				p_w_rel   = float(w_rtf) / ( k1 + (float(tfsum)/AVGLEN) + w_rtf ) # P(w|rel)_BM25
				p_w_irrel = float(w_tf - w_rtf) / ( k1 + (float(TOTALTF - tfsum)/AVGLEN) + float(w_tf - w_rtf) ) # P(w|irrel)_BM25
				val = rsj * max(p_w_rel - p_w_irrel, 0)
			if args.normalize:
				weights[w] = 2.0/(1.0+math.exp(-val)) - 1.0
			else:
				weights[w] = val
		out.write("%s\t"%(rel_level) + " ".join(["%s %.7f"%(k,w) for k,w in sorted([ (k, weights[k]) for k in weights.iterkeys() ], key=operator.itemgetter(1), reverse=True) ]) + "\n")
	out.close()
	return fname

def main():
	global DFT, TFT
	parser = argparse.ArgumentParser(description='Create offer weights for queries from STDIN; outputs updated query markup to STDOUT.')
	parser.add_argument('-r', '--rels', required=True, type=str, help='relevance judgements')
	parser.add_argument('-d', '--docs', required=True, type=str, help='document tf vectors')
	parser.add_argument('--dftable',    required=True, type=str, help='document dftable')
	parser.add_argument('-o', '--output', required=True, type=str, help='output directory for query specific files')
	parser.add_argument('--normalize', action="store_true", help='sigmoid normalization')
	args = parser.parse_args()

	if not args.output:
		sys.stderr.write("specify output directory!\n")
		sys.exit(1)
	# output dir
	if os.path.exists(args.output):
		sys.stderr.write("output dir exists!\n")
		sys.exit(1)
	os.makedirs(args.output)

	loadRels(args.rels)
	loadDFTable(args.dftable)
	loadDocumentVectors(args.docs)
	
	# process queries
	queries = readQueryInput(sys.stdin)
	for qid, group in groupby(queries, operator.itemgetter(0)):
		fname = computeRelevantCounts(args,qid)
		for qid, query in group:
			sys.stdout.write(qid + "\t" + query.replace('<seg grammar', '<seg rel="%s" grammar'%fname) + "\n")
	
	return

if __name__=='__main__': main()

