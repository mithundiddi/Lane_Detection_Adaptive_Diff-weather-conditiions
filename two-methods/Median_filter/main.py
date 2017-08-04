import cv2
from matplotlib import pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy import misc

from polydrawer import Polydrawer
from polyfitter import Polyfitter
from thresholder import Thresholder
from undistorter import Undistorter
from warper import Warper
import numpy as np

from filterlib import Filterer

undistorter = Undistorter()
thresholder = Thresholder()
warper = Warper()
polyfitter = Polyfitter()
polydrawer = Polydrawer()
leftFilter = Filterer(4)
rightFilter = Filterer(4)
leftFilter2 = Filterer(6)
rightFilter2 = Filterer(6)

def main():
    path = './inputvideos/'
    video = 'project_video_1'
    white_output = 'outputvideos/{}_done.mp4'.format(video)
    clip1 = VideoFileClip('{}.mp4'.format(path+video))#.subclip(0,6)
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)





last_car_pos = 0
chase_state = 0
left_fit_prev = [0,0,0]
right_fit_prev = [0,0,0]
last_mask = 0
first = 1
pl_prev = 0
pr_prev = 0
failed = 0

def process_image(base):
    global last_car_pos
    global chase_state
    global left_fit_prev
    global right_fit_prev
    global last_mask
    global first
    global pl_prev, pr_prev, failed
    img = thresholder.color_thresh(base)
    img = thresholder.select_region(img)
    if first != 1 and failed == 0:
        img[(last_mask==0)] = (0,0,0)
    img = warper.warp(img)
    img = thresholder.remove_small_contours(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        left_fit, right_fit = polyfitter.polyfit(img)
        if failed > 0:
            leftFilter.state = 0
            rightFilter.state = 0
            leftFilter2.state = 0
            rightFilter2.state = 0
        left_fit = leftFilter.medianfilter(left_fit)
        right_fit = rightFilter.medianfilter(right_fit)
        left_fit = leftFilter2.meanfilter(left_fit)
        right_fit = rightFilter2.meanfilter(right_fit)
        pl, pr = polydrawer.points(img,left_fit,right_fit)
        if first == 1:
            pl_prev = pl
            pr_prev = pr
        msel = (pl-pl_prev)
        msel = np.multiply(msel,msel)
        msel = np.mean(msel)
        mser = (pr-pr_prev)
        mser = np.multiply(mser,mser)
        mser = np.mean(mser)
        if mser > 170 or abs(left_fit[0]-right_fit[0])>0.0002:
            right_fit = right_fit_prev
            failed = 1
        else:
            # right_fit_prev = right_fit
            # pr_prev = pr
            failed = 0
        if msel > 170 or abs(left_fit[0]-right_fit[0])>0.0002:
            left_fit = left_fit_prev
            if failed = 1:
                failed = 3
            else:
                failed = 2
        else:
            # left_fit_prev = left_fit
            # pl_prev = pl
            failed = 0
        pl_prev = pl
        pr_prev = pr
    except TypeError:
        left_fit = left_fit_prev
        right_fit = right_fit_prev
        failed = 2
    if failed == 1 or failed == 2:
        color = (255,69,0)
    elif failed == 3:
        color = (255,0,0)
    else:
        color = (0,255,0)
    img,last_mask = polydrawer.draw(base, left_fit, right_fit, warper.Minv,color)    
    if failed == 0:
        cv2.putText(img, "Both lanes good", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0), thickness=2)
    elif failed == 1:
        cv2.putText(img, "Right lane bad", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 69, 0), thickness=2)
    elif failed == 2:
        cv2.putText(img, "Left lane bad", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 69, 0), thickness=2)
    else:
        cv2.putText(img, "No lanes detected", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0), thickness=2)
    
    right_fit_prev = right_fit
    left_fit_prev = left_fit
    first = 0
    return img    

def show_image(fig, i, img, title, cmap=None):
    a = fig.add_subplot(2, 2, i)
    plt.imshow(img, cmap)
    a.set_title(title)
    return i + 1

if __name__ == '__main__':
    main()
