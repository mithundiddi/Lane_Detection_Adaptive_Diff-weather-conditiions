#! bin/env/python
import cv2
import os
import sys
import time
import numpy as np
from matplotlib import pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy import misc
from moviepy.editor import *
from moviepy.video.tools.segmenting import findObjects
from polydrawer import Polydrawer
from polyfitter import Polyfitter
from thresholder import Thresholder
from undistorter import Undistorter
from warper import Warper
from roi import ROI


undistorter = Undistorter()
thresholder = Thresholder()
warper = Warper()
polyfitter = Polyfitter()
polydrawer = Polydrawer()
roi=ROI()

def main():
#input name here..

	clip_name='two'
	clip_fullname='{}.png'.format(clip_name)
	path='./inputimages/'
    clip_in=path+clip_fullname
    clip1=ImageClip(clip_in)
    time_name= time.strftime("%Y%m%d-%H%M%S")
	white_clip = clip1.fl_image(process_image)
	a='{}_{}.jpg'.format(clip_name,time_name)
	white_clip.save_frame(a) 
	#clip1=ImageClip(sys.argv[1])
	#clip2=cv2.imread(sys.argv[1],0)
	#cv2.imshow('image',clip1)
	#cv2.waitKey(0)
def process_image(base):
	i = 1

	time_name= time.strftime("%Y%m%d-%H%M%S")

	img = thresholder.threshold(base)
	c='threshold_{}.jpg'.format(time_name)
	c1="./output_images/"+c
	misc.imsave(c1, img)

	img = warper.warp(img)
	misc.imsave('output_images/warped.jpg', img)
	
	img=roi.maskImage(img)
	misc.imsave('output_images/roi.jpg',img)
	
	left_fit, right_fit = polyfitter.polyfit(img)

    img = polydrawer.draw(undistorted, left_fit, right_fit, warper.Minv)
    misc.imsave('output_images/final.jpg', img)
   
    lane_curve, car_pos = polyfitter.measure_curvature(img)

    if car_pos > 0:
        car_pos_text = '{}m right of center'.format(car_pos)
    else:
        car_pos_text = '{}m left of center'.format(abs(car_pos))

    cv2.putText(img, "Lane curve: {}m".format(lane_curve.round()), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color=(255, 255, 255), thickness=2)
    cv2.putText(img, "Car is {}".format(car_pos_text), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255),
                thickness=2)

    return img


if __name__=="__main__":
	main()

