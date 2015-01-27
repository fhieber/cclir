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
	with open(fname) as f:
		for line in f:
			did, length, dvec = line.strip().split("\t",2) # dont need length
			DOCS[did] = (length, parseVector(dvec)) # mapping did : (length,vector)
	sys.stderr.write("TFT loaded\n")

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
	sys.stderr.write("DFT loaded\n")
	
def loadRels(fname):
	global RELS
	with open(fname) as f:
		for line in f:
			qid, _, did, rl = line.strip().split("\t")
			RELS[qid][int(rl)].append( did )
		f.close()
	sys.stderr.write("RELS loaded\n")

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
		doccount = 0
		for doc_id in rel_doc_ids:
			length, doc = DOCS.get(doc_id, (None,None))
			if doc:
				doccount  += 1
				tf_vector += doc
				df_vector += Counter(doc.iterkeys())
		tfsum = sum(tf_vector.values())
		dfsum = sum(df_vector.values())
		assert(len(tf_vector)==len(df_vector))
		out.write(
			"%s %d %d %d "%(rel_level,doccount,tfsum,dfsum) + 
			" ".join([ "%s %d %d"%(k,v1,v2) for k,v1,v2 in sorted([ ( k, tf_vector[k], df_vector[k] ) for k in tf_vector.iterkeys() ], key=operator.itemgetter(1), reverse=True) ]) + "\n"
		)
	out.close()
	return fname

def main():
	global DFT, TFT
	parser = argparse.ArgumentParser(description='Create offer weights for queries from STDIN; outputs updated query markup to STDOUT.')
	parser.add_argument('-r', '--rels', required=True, type=str, help='relevance judgements')
	parser.add_argument('-d', '--docs', required=True, type=str, help='document tf vectors')
	parser.add_argument('--dftable',    required=True, type=str, help='document dftable')
	parser.add_argument('-o', '--output', required=True, type=str, help='output directory for query specific files')
	parser.add_argument('--table-out', required=True, type=str, help='output file for table containing total tfs and df')
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
			sys.stdout.write(qid + "\t" + query.replace('<seg grammar', '<seg relcounts="%s" grammar'%fname) + "\n")
	
	# write total df/tf value file
	with open(args.table_out, 'w+') as f:
		for term, df in DFT.iteritems():
			tf = TFT.get(term,0)
			if tf==0: sys.stderr.write("Warning: zero tf for term '%s'\n"%term)
			if df==0: sys.stderr.write("Warning: zero df for term '%s'\n"%term)
			f.write("%s %d %d\n"%(term,tf,df))
		f.close()
	sys.stderr.write("TFDF table written\n")
	return

if __name__=='__main__': main()

