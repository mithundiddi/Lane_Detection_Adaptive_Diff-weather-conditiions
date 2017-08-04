import cv2
import numpy as np
from numpy.polynomial import Polynomial as P
import sys
from scipy import misc

class Thresholder:
    def __init__(self):
        self.sobel_kernel = 15

        self.thresh_dir_min = 0.7
        self.thresh_dir_max = 1.2

        self.thresh_mag_min = 50
        self.thresh_mag_max = 255
    def threshold(self,img):

        imag = img
        imn=img
		#saturation
        img[0:364,:]=0
        misc.imsave('output_images/skyskip.jpg', img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
        (h, s, v) = cv2.split(img)
        s = s*2.4
        s = np.clip(s,0,255)
        img = cv2.merge([h,s,v])
        img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_HSV2BGR)
        misc.imsave('output_images/sat.jpg', img)
		#blur
        #img=cv2.bilateralFilter(img,9,75,75)
        img = cv2.GaussianBlur(img, (3,3), 0)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        yellow_min = np.array([10, 0, 100], np.uint8)
        yellow_max = np.array([30, 255, 255], np.uint8)
        yellow_mask = cv2.inRange(img, yellow_min, yellow_max)
        yellow_shell = cv2.blur(yellow_mask,(5,5))
        misc.imsave('output_images/ym.jpg', yellow_shell)

        img = cv2.cvtColor(imag, cv2.COLOR_RGB2HLS)
        white_min1 = np.array([0,0, 12], np.uint8)#190
        white_max1 = np.array([255, 255, 255], np.uint8)#255
        white_mask1 = cv2.inRange(img, white_min1, white_max1)
        white_shell1 = -(cv2.blur(white_mask1,(10,10)))
        misc.imsave('output_images/wk1.jpg', white_shell1)

        white_min2 = np.array([0,0, 20], np.uint8)#190
        white_max2 = np.array([255, 255, 255], np.uint8)#255
        white_mask2 = cv2.inRange(img, white_min2, white_max2)
        white_shell2 = -(cv2.blur(white_mask2,(10,10)))
        misc.imsave('output_images/wk2.jpg', white_shell2)

        img = cv2.cvtColor(imag, cv2.COLOR_RGB2HLS)
        white_min3 = np.array([0, 190, 0], np.uint8)
        white_max3 = np.array([255, 255, 255], np.uint8)
        white_mask3 = cv2.inRange(img, white_min3, white_max3)
        white_shell3 = cv2.blur(white_mask3,(10,10))
        misc.imsave('output_images/wk.jpg', white_shell3)
        #edge detection
		#canny
        #img=cv2.bilateralFilter(img,9,75,75)
        img = cv2.GaussianBlur(imag, (1,1), 0)
        img1=cv2.Canny(img, 100,150)
        edges=img1
        rows, cols = img1.shape
        imgn=img1.nonzero()
        imgny = np.array(imgn[0])
        imgnx = np.array(imgn[1])
        imgz=[[],[]]
        for i in range(0,len(imgnx)):
            imgz.append([imgny[i],imgnx[i]])
        vertices = np.array([imgz[2:]], dtype=np.int32)
        mask = np.zeros_like(img1)
        if len(mask.shape)==2:
            #cv2.fillPoly(mask, vertices, 255)
            cv2.polylines(mask, vertices, 0, 255, 8, 8, 0)
            #print("1")
        else:
            cv2.fillPoly(mask,vertices , (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
        misc.imsave('output_images/masker.jpg', mask)
        imgtrail=cv2.bitwise_and(img1, mask)
        misc.imsave('output_images/canny.jpg', edges)
        misc.imsave('output_images/canny_improves.jpg', imgtrail)


        #sobel
        sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=15)
        sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=15)
        abs_sobelx = np.abs(sobelx)
        abs_sobely = np.abs(sobely)
        scaled_sobel = np.arctan2(abs_sobely, abs_sobelx)
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >=0.7) & (scaled_sobel <= 1.2)] = 1
        #misc.imsave('output_images/sx.jpg', sxbinary)

        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >=50) & (gradmag <= 255)] = 1
        #misc.imsave('output_images/binary.jpg', binary_output)
		#sobel grad and mag
        sobel_combined = np.zeros_like(sxbinary)
        sobel_combined[(binary_output==1) | (sxbinary == 1)] = 1
        sobel_out = np.zeros_like(sxbinary)
        sobel_out[np.abs((sobel_combined==1)-(binary_output==1))]=1
        sobel_out=cv2.cvtColor(sobel_out, cv2.COLOR_HSV2RGB)
        sobel_out=cv2.cvtColor(sobel_out, cv2.COLOR_BGR2GRAY)
        #misc.imsave('output_images/sobel_out.jpg', sobel_out)
        #COMBINE BOTH
        #final_edge=np.zeros_like(edges)
        #final_edge[(img1==1)&(sobel_out==1)]=1
        #misc.imsave('output_images/final_edge.jpg', final_edge)
        final_edge=img1
        #masking with edges

        img1 = np.zeros_like(edges)
        img2 = np.zeros_like(edges)
        img3 = np.zeros_like(edges)

        img1[(white_shell1 != 0) & (final_edge != 0)] = 255
        img1[(yellow_shell != 0) & (final_edge != 0)] = 255

        misc.imsave('output_images/colour_masks1.jpg', img1)

        img2[(white_shell2 != 0) & (final_edge != 0)] = 255
        img2[(yellow_shell != 0) & (final_edge != 0)] = 255
        misc.imsave('output_images/colour_masks2.jpg', img2)


        img3[(white_shell3 != 0) & (final_edge != 0)] = 255
        img3[(yellow_shell != 0) & (final_edge != 0)] = 255
        misc.imsave('output_images/colour_masks3.jpg', img3)
        
        #region of interst

        rows, cols = img.shape[:2]
        bottom_left  = [cols*0.01, rows*0.86]
        top_left     = [cols*0.4, rows*0.60]
        bottom_right = [cols*0.75, rows*0.86]
        top_right    = [cols*0.65, rows*0.60] 
        # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

        #Create the mask using the vertices and apply it to the input image

        mask = np.zeros_like(img1)
        if len(mask.shape)==2:
        	cv2.fillPoly(mask, vertices, 255)
        else:
        	cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        

        img1=cv2.bitwise_and(img1, mask)
        misc.imsave('output_images/thresholder1.jpg', img1)

        img2=cv2.bitwise_and(img2, mask)
        misc.imsave('output_images/thresholder2.jpg', img2)
	       
        img3=cv2.bitwise_and(img3, mask)
        misc.imsave('output_images/thresholder3.jpg', img3)
        imy=np.zeros_like(img3)
        imy=cv2.bitwise_or(img1,img2,img3)
        misc.imsave('output_images/thresholdercombined.jpg', imy)

	               
        imgx=len(img1.nonzero())
        imgy=len(img2.nonzero())
        imgc=len(img3.nonzero())
        im=[imgx,imgy,imgc]
        im.sort()

        if im[2]==imgx:
              img=img1
        elif im[2]==imgy:
              img=img2
        else:
        	   img=img3
               
        return img