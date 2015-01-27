#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys

prev_id = ""
i = 0
for line in sys.stdin:
	l = line.strip().split("\t")
	curr_id = l[0]
	text = l[1] 
	if curr_id != prev_id:
		if i > 0:
			sys.stdout.write( "\n"+curr_id+ "\t"+ text )
		else:
			sys.stdout.write( curr_id+ "\t"+ text )
			
	else:
		sys.stdout.write( " "+text )
		sys.stdout.flush()
	prev_id = curr_id
	i += 1
sys.stdout.write("\n")
sys.stdout.flush()
	
	
	
