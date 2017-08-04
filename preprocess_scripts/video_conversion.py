#!/bin/env python
import os
import subprocess

if __name__ == "__main__":
	#foldername = "./13"
	#a = os.listdir("./")
	#for j in range(0,len(a)):
		#print("./%s" % a[j])
		#b=os.listdir(a[j])
		#for i in b:
			#print (i.rsplit('-',1)[1])
			#os.rename(a[j]+"/"+i,a[j]+"/"+i.rsplit('-',1)[1])
	try:
		subprocess.Popen('for j in *; do ffmpeg -start_number 0 -framerate 15 -i $j/%d.jpg ../videos/"$j".webm;done', shell=True)
	#subprocess.Popen('for j in *;do ls $j/;done', shell=True)
	except: pass