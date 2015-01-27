#!/usr/bin/python

import sys, os, re, shutil, math
import subprocess, shlex, glob
import argparse
import logging
import random, time
import gzip, itertools

import RelevanceScorer
import average_weights
    
def DecodeAndRetrieve(args, script_dir, weightfile, iteration):
	parallelize = script_dir+'/../utils/parallelize.pl'
	decoder = script_dir+'/decode.sh'
	retriever = script_dir+'/retrieval.sh'
	inputfile = os.path.join(args.output_dir,'test')
	config = os.path.join(args.output_dir,'cdec.ini')
	if args.rescoring_model:
		logging.info('======= DECODING with weights.smt, RESCORING with: %s ======='%weightfile)
		decoder_cmd = '{0} {1} {2} {3} {4} {5}'.format(decoder, inputfile, config, os.path.join(args.output_dir,'weights.smt'), weightfile, args.testjobs)
	else:
		logging.info('======= DECODING with weights: %s ======='%weightfile)
		decoder_cmd = '{0} {1} {2} {3} {4}'.format(decoder, inputfile, config, weightfile, args.testjobs )
	if args.retrieve:
		decoder_cmd += " | tee " + inputfile+".%02d"%iteration + " | {0} {1} {2} 2> {3}.err > {3}.eval".format(retriever, args.qrels, " ".join(args.servers), inputfile+".%02d"%iteration)
	else:
		decoder_cmd += " > " + inputfile+".%02d"%iteration
	logging.info('DECODER COMMAND: {}'.format(decoder_cmd))
	return subprocess.Popen(decoder_cmd, shell=True)
	
def scaleRelevanceWeightsToSMTWeights(relevanceweightfile, weightfile):
	s = 0.0
	with open(weightfile) as f:
		for l in f:
			if l.startswith('#'):
				continue
			f,v = l.strip().split()
			s += float(v) * float(v)
	wnorm = math.sqrt(s)
	if wnorm == 0.0:
		return relevanceweightfile
	r_old = open(relevanceweightfile)
	r_new = open(relevanceweightfile+'.scaled', 'w')
	for l in r_old:
		if l.startswith('#'):
			continue
		f,v = l.strip().split()
		r_new.write( f + ' ' + "%.7f\n"%(float(v)/wnorm) )
	r_old.close()
	r_new.close()
	return relevanceweightfile+'.scaled'

def optimize(args, script_dir, source, source_size, references):

	parallelize = script_dir+'/../utils/parallelize.pl'
	decoder = script_dir+'/online_ssvm'
	best_score_iter = -1
	best_score = -1
	i = 0
	hope_best_fear = {'hope':[],'best':[],'fear':[]}
	decode_processes = []
	if args.test: # start decode of test set with initial weights
		decode_processes.append( DecodeAndRetrieve(args, script_dir, args.output_dir+'/weights.%02d'%i, i) )
	
	# scale relevance weights to length of smt weights
	relevanceweightsfile = scaleRelevanceWeightsToSMTWeights(os.path.join(args.output_dir,'weights.relevance'), os.path.join(args.output_dir,'weights.00'))
	# or not :)
	if args.noscale:
		relevanceweightsfile = args.output_dir+'/weights.relevance'
	
	relevanceweights = RelevanceScorer.loadWeights(relevanceweightsfile)
	
	while i<args.max_iterations:
	
		logging.info('======= STARTING ITERATION {} ======='.format(i))
		logging.info('Starting at {}'.format(time.asctime()))
		
		#iteration specific files
		runfile = args.output_dir+'/run.%02d'%i
		logdir = args.output_dir+'/logs.%02d'%i
		decoderlog = logdir+'/decoder.sentserver.log.%02d'%i
		weightdir = args.output_dir+'/pass.%02d'%i
		os.mkdir(logdir)
		os.mkdir(weightdir)
		weightsfile = args.output_dir+'/weights.%02d'%i
		logging.info('  log directory={}'.format(logdir))
		curr_pass = '%02d'%i
		decoder_cmd = ('{0} -c {1} -w {2} -r {3} -n {4} -o {5}').format(
				   decoder, args.config, weightsfile, relevanceweightsfile,
				   args.learningrate, weightdir)
		if args.ramploss:
			decoder_cmd += " --ramploss"
		if args.perceptron:
			decoder_cmd += " --perceptron"
		if args.adagrad:
			decoder_cmd += " -a"
		if args.regularization != 0.0:
			decoder_cmd += " -l {}".format(args.regularization)
		if args.freeze != []:
			decoder_cmd += " --freeze %s"%(" ".join(args.freeze))
		if args.sample:
			decoder_cmd += " --sample %d"%(args.sample)
		if args.rescoring_model:
			decoder_cmd += " --firstpass-model %s"%(os.path.join(args.output_dir,'weights.smt'))
		if args.merge_forests:
			decoder_cmd += " --forest-output %s"%(os.path.join(args.output_dir,'hgs'))
		if args.batch_update:
			decoder_cmd += " --no-redecode"
		if args.conservative:
			decoder_cmd += " --conservative"
		
		#always use fork 
		parallel_cmd = '{0} --use-fork -e {1} -j {2} --'.format(parallelize, logdir, args.jobs)
		
		cmd = parallel_cmd + ' ' + decoder_cmd
		logging.info('OPTIMIZATION COMMAND: {}'.format(cmd))
		
		dlog = open(decoderlog,'w')
		runf = open(runfile,'w')
		retries = 0
		num_processed = 0
		while retries < 6:
			#call decoder through parallelize.pl
			p1 = subprocess.Popen(['cat', source], stdout=subprocess.PIPE)
			exit_code = subprocess.call(shlex.split(cmd), stderr=dlog, stdout=runf, stdin=p1.stdout)
			p1.stdout.close()
			if exit_code:
				logging.error('Failed with exit code {}'.format(exit_code))
				sys.exit(exit_code)
			try:
				f = open(runfile)
			except IOError:
				logging.error('Unable to open {}'.format(runfile))
				sys.exit()

			num_processed = sum(1 for line in f)
			f.close()
			if num_processed == source_size: break
			logging.warning('Incorrect number of processed output. Sleeping for 10 seconds and retrying...')
			time.sleep(10)
			retries += 1

		if source_size != num_processed:
			logging.error("source set contains "+source_size+" sentences, but we don't have topbest for all of these. Decoder failure? Check "+decoderlog)
			sys.exit()
		dlog.close()
		runf.close()
		
		time.sleep(3)

		#write best, hope, and fear translations
		run = open(runfile)
		H = open(runfile+'.H', 'w')
		B = open(runfile+'.B', 'w')
		F = open(runfile+'.F', 'w')
		hopes = []
		bests = []
		fears = []
		for line in run:
			qid, fear, best, hope = line.strip().split(' ||| ', 3)
			hopes.append(hope)
			bests.append(best)
			fears.append(fear)
			H.write('%s\t%s\n'%(qid,hope))
			B.write('%s\t%s\n'%(qid,best))
			F.write('%s\t%s\n'%(qid,fear))
		run.close()
		H.close()
		B.close()
		F.close()
		
		# score H/B/F with fast_score using RelevanceScorer
		dec_score, dec_score_detail = RelevanceScorer.scoreRelevance(bests, references, relevanceweights)
		dec_score_f, dec_score_detail_f = RelevanceScorer.scoreRelevance(fears, references, relevanceweights)
		dec_score_h, dec_score_detail_h = RelevanceScorer.scoreRelevance(hopes, references, relevanceweights)
		
		if dec_score > best_score:
			best_score_iter = i
			best_score = dec_score
		hope_best_fear['hope'].append(dec_score)
		hope_best_fear['best'].append(dec_score_h)
		hope_best_fear['fear'].append(dec_score_f)
		logging.info('VITERBI SCORE: {0} HOPE: {1} FEAR: {2}'.format(dec_score, dec_score_h, dec_score_f))
		# write to scores file
		scoresfile = open(args.output_dir+'/scores', 'a')
		scoresfile.write("%d %.4f %.4f %.4f\n"%(i,dec_score_f, dec_score, dec_score_h))
		scoresfile.close()
		
		new_weights_file = '%s/weights.%02d'%(args.output_dir, i+1)
		last_weights_file = '%s/weights.%02d'%(args.output_dir, i)
		i += 1
		weight_files = glob.glob(weightdir +'/weights.*')
		average_weights.average_weights(new_weights_file, weight_files, args.foldaverage)
		
		if args.test: # start decode of test set with new weights
			decode_processes.append( DecodeAndRetrieve(args, script_dir, new_weights_file, i) )
		
	logging.info('BEST ITERATION: {} (SCORE={})'.format(best_score_iter, best_score))
	weights_final = args.output_dir+'/weights.final'
	logging.info('WEIGHTS FILE: {}'.format(weights_final))
	shutil.copy(last_weights_file, weights_final)
	
	if args.test:
		logging.info('======= WAITING FOR DECODING PROCESSES =======')
		for p in decode_processes:
			logging.info('wait for %s'%(str(p)))
			p.wait()
	logging.info('======= FINISHED =======')
	
	return weights_final
		
    
def main():
	logging.basicConfig(level=logging.INFO)
	script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
	
	parser= argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-i', '--input', required=True, help='training set input queries with relevance annotation')
	parser.add_argument('-w','--weights', help='(initial) SMT weights file')
	parser.add_argument('--rescoring-model', action="store_true", help='if set, learn a model on top of the initially specified SMT weights.')
	parser.add_argument('-r','--relevanceweights', required=True, help='relevance weights for hope paths')
	parser.add_argument('-c', '--config', required=True, help='decoder configuration file for training (cdec.ini)')
	parser.add_argument('-o','--output-dir', required=True, help='directory for intermediate and output files.')
	parser.add_argument('--max-iterations', type=int, default=20, metavar='N', help='maximum number of iterations to run')
	parser.add_argument('--ramploss', action="store_true", help='use ramploss to select hope: cost+margin')
	parser.add_argument('--perceptron', action="store_true", help='use perceptron: fear = viterbi')
	parser.add_argument('-a','--adagrad', action="store_true", help='use adaptive learning rates for each feature (AdaGrad)')
	parser.add_argument('-l','--learningrate', type=float, default=0.01, help='learning rate / step size')
	parser.add_argument('--regularization', type=float, default=0.0, help='l1 regularization strength. Disabled by default.')
	parser.add_argument('--foldaverage', action="store_true", help='average weights by number of folds/jobs, not by no of lines.')
	parser.add_argument('--batch-update', action="store_true", help='use initial epoch weights for each input sentence. Update only after epoch.')
	parser.add_argument('--freeze', type=str, nargs='+', default=[], help='list of frozen features.')
	parser.add_argument('-j','--jobs', type=int, default=1, help='number of jobs')
	parser.add_argument('--sample', type=int, help='sample hope and fear x times from forest for each input sentence')
	parser.add_argument('--merge-forests', action="store_true", help='merge search space with previous iterations')
	parser.add_argument('-t','--test', type=str, help='decode test set with new weights after each iteration')
	parser.add_argument('--testjobs', type=int, default=1, help='number of jobs for decoding the test set (running in background)')
	parser.add_argument('--retrieve', action="store_true", help='whether to run retrieval')
	parser.add_argument('--qrels', type=str, help='qrels for testing')
	parser.add_argument('--servers', type=str, nargs='+', default={}, help='list of retrieval servers')
	parser.add_argument('--noscale', action='store_true', help='do not scale relevance weights to init weights')
	parser.add_argument('--conservative', action='store_true', help='use conservative MIRA updates')
	args = parser.parse_args()
	
	if not os.path.isabs(args.output_dir):
		args.output_dir = os.path.abspath(args.output_dir)
	if os.path.exists(args.output_dir):
		if len(os.listdir(args.output_dir))>2:
			logging.error('Error: working directory {0} already exists\n'.format(args.output_dir))
			sys.exit()
	else:
		os.mkdir(args.output_dir)
		
	# get initial weight file
	if args.weights:
		shutil.copy(args.weights,os.path.join(args.output_dir,'weights.00'))
	else: #if no weights given, use Glue 0 as default
		weights = open(args.output_dir+'/weights.00','w')
		weights.write('Glue 0\n')
		weights.write('WordPenalty 0\n')
		weights.close()
		args.weights = args.output_dir+'/weights.00'
	if args.rescoring_model:
		shutil.copy(os.path.join(args.output_dir,'weights.00'), os.path.join(args.output_dir,'weights.smt'));
	# copy relevance weights
	shutil.copy(args.relevanceweights, os.path.join(args.output_dir,'weights.relevance'))
		
	#copy ini file
	shutil.copy(args.config,'{0}/cdec.ini'.format(args.output_dir))
	
	# copy input
	shutil.copy(args.input, os.path.join(args.output_dir,'input'))
	
	# copy test set
	if args.test:
		shutil.copy(args.test, os.path.join(args.output_dir,'test'))
		
	# make hypergraph dir
	if args.merge_forests:
		os.mkdir(os.path.join(args.output_dir,'hgs'))
	
	# get size of training set
	inputfile = open(args.input)
	referencefile = open(os.path.join(args.output_dir,'references'), 'w')
	references = []
	source_size = 0
	for line in inputfile:
		source_size += 1
		r = re.search(' rel="(.*?)" ',line).group(1)
		referencefile.write(r + '\n')
		references.append(r)
	inputfile.close()
	referencefile.close()
	assert len(references) == source_size
	
	# run optimization
	args.weights = optimize(args, script_dir, args.input, source_size, references)
	
	return 0
	

if __name__=="__main__": main()
