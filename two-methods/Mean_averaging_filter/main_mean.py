import cv2
from matplotlib import pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy import misc
import os,sys,time
from polyfitfilter import Polyfitfilter
from polydrawer import Polydrawer
from polyfitter import Polyfitter
from thresholder import Thresholder
from warper import Warper
from roi import ROI


polyfitfilter = Polyfitfilter()
thresholder = Thresholder()
warper = Warper()
polyfitter = Polyfitter()
polydrawer = Polydrawer()
roi=ROI()

def main():
    #input file name without extension type
    
    video ='project_video'
    
    path='./inputvideos/'
    video_in=path+video
    time_name= time.strftime("%Y%m%d-%H%M%S")
    white_output = 'output_videos/{}_{}.webm'.format(video,time_name)
    #change the subclip size
    clip1 = VideoFileClip('{}.mp4'.format(video_in)).subclip(1, 6)
    
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False, codec='libvpx')


def process_image(base):
    fig = plt.figure(figsize=(10,8))
    i = 1
    imgpoly=base
    img = thresholder.threshold(base)
    misc.imsave('output_images/thresholded.jpg', img)
    

    img = warper.warp(img)
    misc.imsave('output_images/warped.jpg', img)

    #img=roi.maskImage(img)
    #misc.imsave('output_images/roi.jpg',img)

    
    # i = show_image(fig, i, img, 'Thresholded', 'gray')

    
    # i = show_image(fig, i, img, 'Warped', 'gray')

    left_fit, right_fit = polyfitter.polyfit(img)

    # Linear filtering and rejection of bogus data for line coefficients
    left_fit, right_fit = polyfitfilter.filterLineCoefficients(left_fit,right_fit)
    print(polyfitfilter.confidence)

    img = polydrawer.draw(imgpoly, left_fit, right_fit, warper.Minv, polyfitfilter.confidence)
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

    # show_image(fig, i, img, 'Final')
    # plt.imshow(img)
    # plt.show()
    return img


def show_image(fig, i, img, title, cmap=None):
    a = fig.add_subplot(2, 2, i)
    plt.imshow(img, cmap)
    a.set_title(title)
    return i + 1


if __name__ == '__main__':
   main()
