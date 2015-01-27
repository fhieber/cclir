#!/usr/bin/python

import sys,argparse, math

def loadWeights(fname):
	w = {}
	with open(fname) as f:
		for line in f:
			f,v = line.strip().split(' ',1)
			f = int(f.split('_')[1])
			w[f] = float(v)
	return w

def batch_gen(data, batch_size):
	for i in range(0, len(data), batch_size):
		yield data[i:i+batch_size]

def loadVector(s):
	v = {}
	for k,val in batch_gen(s.split(),2):
		v[k] = float(val)
	return v
	
def loadRelevanceFile(fname):
	r = {}
	with open(fname) as f:
		for line in f:
			rl, rawvec = line.strip().split('\t',1)
			r[int(rl)] = loadVector(rawvec)
	return r
	
def loadDFT(fname):
	dft = {}
	maxdf = 0.0
	with open(fname) as f:
		for line in f:
			w,df = line.strip().split(' ',1)
			dft[w] = float(df)
			if df > maxdf:
				maxdf = df
	return dft, maxdf

def normalize(x, sigma=1.0, beta=1.0):
	return ( (2*sigma) / (1+math.exp(-x*beta)) ) - sigma;

def scoreRelevance(hypotheses, relevance_files, weights, dft={}, N=0.0):
	relevance = {}
	n = 0
	idx = 0
	for h in hypotheses:
		rels = loadRelevanceFile(relevance_files[idx])
		idx +=1
		for word in h.strip().split():
			isrel = False
			for rl, vals in rels.iteritems():
				if word in vals:
					isrel = True
					relevance[rl] = relevance.get(rl,0.0) + vals.get(word, 0.0)
			#if not isrel: # is a junk word, assign negative relevance
			#	if dft and N:
			#		relevance[0] = relevance.get(0,0.0) - dft.get(word,0.0)/N
			#	else:
			#		relevance[0] = relevance.get(0,0.0) - 1
			relevance[0] = relevance.get(0,0.0) - (1/math.log(10))
		n += 1

	weighted_relevance = {}
	for rl,weight in weights.iteritems():
		weighted_relevance[rl] = weight * relevance.get(rl,0.0)
		
	return sum(weighted_relevance.values())/n , [(rl, rel/n) for rl,rel in sorted(weighted_relevance.iteritems(), reverse=True)]
	

def main():
	parser= argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-i', required=True, help='hypotheses to score')
	parser.add_argument('-r', required=True, help='references: file containing filenames of relevance files')
	parser.add_argument('-w', required=True, help='relevance weights for relevance levels')
	parser.add_argument('-d', help='optional df table for graceful penalty on irrelevant words')
	parser.add_argument('-n', type=int, help='number of documents in collection')
	args = parser.parse_args()
	
	w = loadWeights(args.w)
	if args.d:
		dft, N = loadDFT(args.d)
	else:
		dft, N = {}, 0
	if args.n:
		N = args.n
	
	hyps = open(args.i)
	hypotheses = [line.strip() for line in hyps]
	hyps.close()
	refs = open(args.r)
	relevance_files = [line.strip() for line in refs]
	refs.close()
	if len(relevance_files) < len(hypotheses):
		sys.stderr.write("Error: only %d relevance files for %d hypotheses!\n"%(len(relevance_files),len(hypotheses)))
		return 1
	elif len(hypotheses) < len(relevance_files):
		sys.stderr.write("Warning: scoring partial set: %d hypotheses. Found %d relevance files.\n"%(len(hypotheses),len(relevance_files)))
	
	score, score_detail = scoreRelevance(hypotheses,relevance_files,w,dft, N)
	
	sys.stdout.write("R=%.4f\t"%score)
	sys.stdout.write("\t".join("R%d=%.4f"%(rl,rel) for rl,rel in score_detail) + '\n')
	return 0
	
	
if __name__=="__main__": main()
