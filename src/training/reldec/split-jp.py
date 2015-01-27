#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys

for line in sys.stdin:
	l = line.strip().split("\t")
	pid = l[0]
	text = l[1].split() 
	sent = []
	for word in text:
		word = word.decode('utf-8')
		sent.append(word)
		if word == u"。":
			sentence = u' '.join(sent).encode('utf-8')
			sys.stdout.write( pid+ "\t"+ sentence+ "\n" )
			sent = []
		#else: sys.stdout.write( word.encode('utf-8') + " " )
	if text[ -1 ].decode('utf-8') != u"。":
		sentence = u' '.join(sent).encode('utf-8')
		sys.stdout.write( pid+ "\t"+ sentence+ "\n" )
		
		#sys.stdout.write( line )
	
	
	
