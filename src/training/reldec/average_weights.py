#!/usr/bin/python
import sys, logging, glob, os

def average_weights(new_weights, weight_files, fold_avg=False):
    logging.info('AVERAGE {} {}'.format(new_weights, weight_files))
    feature_weights = {}
    total_mult = 0.0
    for path in weight_files:
        score = open(os.path.abspath(path))
        mult = 0
        logging.info('  FILE {}'.format(os.path.abspath(path)))
        try:
            msg, ran, mult = score.readline().strip().split(' ||| ')
        except ValueError:
            logging.info('   no header line found.')
            msg, ran, mult = "", path, "1.0"
        if fold_avg:
            mult ="1.0"
        logging.info('  Processing {} {}'.format(ran, mult))
        for line in score:
            f,w = line.split(' ',1)
            if f in feature_weights:
                feature_weights[f]+= float(mult)*float(w)
            else: 
                feature_weights[f] = float(mult)*float(w)
        total_mult += float(mult)
        score.close()
    
    #write new weights to outfile
    logging.info('Writing averaged weights to {}'.format(new_weights))
    if new_weights == "-": out = sys.stdout
    else: out = open(new_weights, 'w+')
    out.write("# Online SSVM tuned and averaged weights ||| avg ||| %d\n"%(total_mult))
    for f,v in feature_weights.iteritems():
        avg = v/total_mult
        out.write('{} {}\n'.format(f,avg))
    if new_weights != "-": out.close()

def main():
	logging.basicConfig(level=logging.INFO)
	if len(sys.argv) < 2:
		sys.stderr.write("Usage: %s [-s] <model1> <model2> [<model3>...]\n"%sys.argv[0])
		sys.exit(1)
	if sys.argv[1] == '-s':
		average_weights("-", sys.argv[2:], True)
	else:
		average_weights("-", sys.argv[1:], False)
	
if __name__=="__main__": main()

