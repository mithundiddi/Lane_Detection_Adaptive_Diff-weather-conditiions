#!/bin/env python
import os
import subprocess

if __name__ == "__main__":
	foldername = "./13"
	a = os.listdir(foldername)
	for i in a:
		print (i.rsplit('-',1)[1])
		os.rename(foldername+"/"+i,foldername+"/"+i.rsplit('-',1)[1])
		#subprocess.Popen('ffmpeg -start_number 0 -framerate 15 -i "$foldername"\%d.jpg videos\"$foldername".webm', shell=True)
		
