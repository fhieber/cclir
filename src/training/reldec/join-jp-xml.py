#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys, re
from xml.sax.saxutils import escape

def clean(s):
	return re.sub('">', '"><![CDATA[', re.sub('</seg>',']]></seg>', s))

prev_id = ""
i = 0
for line in sys.stdin:
	l = line.strip().split("\t")
	curr_id = l[0]
	text = l[1] 
	if curr_id != prev_id:
		if i>0: sys.stdout.write("</d>\n");
		sys.stdout.write( curr_id+ "\t" + "<d>"+ "<s>" + clean(text) + "</s>")
			
	else:
		sys.stdout.write( "<s>"+clean(text)+"</s>" )
	prev_id = curr_id
	i += 1
sys.stdout.write("</d>\n")
	
	
	
