import cv2
import numpy as np
from scipy import misc

class Thresholder:
    def color_thresh(self, img):
        imag = img
        img = self.saturate(img)
        # misc.imsave('output_images/saturated.jpg', img)
        img = cv2.GaussianBlur(img, (3,3), 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        yellow_min = np.array([10, 0, 100], np.uint8)
        yellow_max = np.array([30, 255, 255], np.uint8)
        yellow_mask = cv2.inRange(img, yellow_min, yellow_max)
        yellow_shell = cv2.blur(yellow_mask,(5,5))
        # yellow_kills_white = cv2.blur(yellow_mask,(100,100))

        img = cv2.cvtColor(imag, cv2.COLOR_RGB2HLS)

        white_min = np.array([0, 190, 0], np.uint8)
        white_max = np.array([255, 255, 255], np.uint8)
        white_mask = cv2.inRange(img, white_min, white_max)
        white_shell = cv2.blur(white_mask,(10,10))
        # white_shell[(yellow_kills_white!=0)] = 0

        img = cv2.GaussianBlur(imag, (1,1), 0)
        edges = self.detect_edges(img)
        edges = cv2.blur(edges,(20,20))
        # misc.imsave('output_images/edges.jpg', edges)
        img = np.zeros_like(imag)
        img[(white_shell != 0) & (edges != 0)] = (255,255,255)
        img[(yellow_shell != 0) & (edges != 0)] = (255,255,255)

        white_shell[(white_shell!=0)] = 255
        # misc.imsave('output_images/whitemask.jpg', white_shell)
        yellow_shell[(yellow_shell!=0)] = 255
        # misc.imsave('output_images/yellowmask.jpg', yellow_shell)

        colored = img
        colored[(white_shell == 0) & (yellow_shell == 0)] = 0
    
        return img
    def saturate(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
        (h, s, v) = cv2.split(img)
        s = s*2.4
        s = np.clip(s,0,255)
        img = cv2.merge([h,s,v])
        img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_HSV2BGR)
        # misc.imsave('output_images/saturated.jpg', img)
        return img
    def detect_edges(self, img, low_threshold=100, high_threshold=150):
        img = cv2.Canny(img, low_threshold, high_threshold)
        return img
    def filter_region(self, image, vertices):
        """
        Create the mask using the vertices and apply it to the input image
        """
        mask = np.zeros_like(image)
        if len(mask.shape)==2:
            cv2.fillPoly(mask, vertices, 255)
        else:
            cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
        return cv2.bitwise_and(image, mask)
    def select_region(self,image):
        """
        It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
        """
        # first, define the polygon by vertices
        # rows, cols = image.shape[:2]
        # bottom_left  = [cols*0.01, rows*0.86]
        # top_left     = [cols*0.35, rows*0.60]
        # bottom_right = [cols*0.99, rows*0.86]
        # top_right    = [cols*0.75, rows*0.60] 
        rows, cols = image.shape[:2]
        bottom_left  = [cols*0.01, rows*0.86]
        top_left     = [cols*0.44, rows*0.55]
        bottom_right = [cols*0.99, rows*0.86]
        top_right    = [cols*0.52, rows*0.55] 
        # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        return self.filter_region(image, vertices)
    def remove_small_contours(self,img):
        base = img.copy()
        base2 = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.ones(img.shape[:2], dtype="uint8") * 255
         
        for c in contours:
            if cv2.contourArea(c)<400:
                cv2.drawContours(mask, [c], -1, 0, -1)
                cv2.drawContours(base2, [c], -1, (0,255,0), 3)
        # misc.imsave('output_images/contours.jpg', base2)
        mask = cv2.blur(mask,(6,6))
        base[(mask!=255)] = (0,0,0)
        return base

    # def addlinesweight(self, img, base):
    #     img = cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)
    #     pos_slopes = []
    #     neg_slopes = []
    #     pos_intercepts = []
    #     neg_intercepts = []
    #     for x in range(0,len(img)):
    #         for x1,y1,x2,y2 in img[x]:
    #             slope = (y2-y1)/(x2-x1)
    #             intercept = y1 - slope*x1
    #             if slope<0:
    #                 neg_slopes.append(np.arctan2((y2-y1),(x2-x1)))
    #                 neg_intercepts.append(intercept)
    #             elif slope>0:
    #                 pos_slopes.append(np.arctan2((y2-y1),(x2-x1)))
    #                 pos_intercepts.append(intercept)
    #     pos_slopes = np.sort(pos_slopes)
    #     neg_slopes = np.sort(neg_slopes)
    #     pos_intercepts = np.sort(pos_intercepts)
    #     neg_intercepts = np.sort(neg_intercepts)
    #     pos_slopes = steepestblock(pos_slopes,1/180*np.pi)
    #     neg_slopes = steepestblock(neg_slopes,1/180*np.pi)
    #     for x in range(0,len(img)):
    #         for x1,y1,x2,y2 in img[x]:
    #             slope = (y2-y1)/(x2-x1)
    #             intercept = y1 - slope*x1
    #             if slope<0:
    #                 slope = np.arctan2((y2-y1),(x2-x1))
    #                 if (slope in neg_slopes) and (intercept in neg_intercepts):
    #                     cv2.line(base,(x1,y1),(x2,y2),(0,0,255),4)
    #             elif slope>0:
    #                 slope = np.arctan2((y2-y1),(x2-x1))
    #                 if (slope in pos_slopes) and (intercept in pos_intercepts):
    #                     cv2.line(base,(x1,y1),(x2,y2),(0,0,255),4)
    #     return base
    def addlinesweight(self, img, base):
        img = cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)
        pos_slopes = []
        neg_slopes = []
        for x in range(0,len(img)):
            for x1,y1,x2,y2 in img[x]:
                slope = (y2-y1)/(x2-x1)
                if slope<0:
                    neg_slopes.append(np.arctan2((y2-y1),(x2-x1)))
                elif slope>0:
                    pos_slopes.append(np.arctan2((y2-y1),(x2-x1)))
        pos_slopes = np.sort(pos_slopes)
        neg_slopes = np.sort(neg_slopes)
        pos_slopes = steepestblock(pos_slopes,1/180*np.pi)
        neg_slopes = steepestblock(neg_slopes,1/180*np.pi)
        pos_intercepts = []
        neg_intercepts = []
        filtered_lines = []
        # print(img)
        for x in range(0,len(img)):
            for x1,y1,x2,y2 in img[x]:
                slope = (y2-y1)/(x2-x1)
                intercept = y1 - slope*x1
                if slope<0:
                    slope = np.arctan2((y2-y1),(x2-x1))
                    if slope in neg_slopes:
                        neg_intercepts.append(intercept)
                        filtered_lines.append([[x1,y1,x2,y2]])
                        # filtered_lines = np.vstack([filtered_lines,[[x1,y1,x2,y2]]])
                elif slope>0:
                    slope = np.arctan2((y2-y1),(x2-x1))
                    if slope in pos_slopes:
                        pos_intercepts.append(intercept)
                        filtered_lines.append([[x1,y1,x2,y2]])
                        # filtered_lines = np.vstack([filtered_lines,[[x1,y1,x2,y2]]])
        # print(filtered_lines)
        pos_intercepts = np.sort(pos_intercepts)
        neg_intercepts = np.sort(neg_intercepts)
        pos_intercepts = largestBlock(pos_intercepts,10)
        neg_intercepts = largestBlock(neg_intercepts,10)
        for x in range(0,len(filtered_lines)):
            for x1,y1,x2,y2 in filtered_lines[x]:
                slope = (y2-y1)/(x2-x1)
                intercept = y1 - slope*x1
                if slope<0:
                    if intercept in neg_intercepts:
                        cv2.line(base,(x1,y1),(x2,y2),(255,255,255),4)
                elif slope>0:
                    if intercept in pos_intercepts:
                        cv2.line(base,(x1,y1),(x2,y2),(255,255,255),4)
        return base

    def addlinesthresh(self, img, base):
        img = cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)
        pos_slopes = []
        neg_slopes = []
        for x in range(0,len(img)):
            for x1,y1,x2,y2 in img[x]:
                slope = (y2-y1)/(x2-x1)
                if abs(slope)>1/3:
                    cv2.line(base,(x1,y1),(x2,y2),(0,0,255),4)
        return base

    def addlinesall(self, img, base):
        img = cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)
        pos_slopes = []
        neg_slopes = []
        for x in range(0,len(img)):
            for x1,y1,x2,y2 in img[x]:
                cv2.line(base,(x1,y1),(x2,y2),(0,0,255),4)
        return base
    def addlines2(self, img, base):
        # img = cv2.HoughLines(img,1,np.pi/90,200)
        img = cv2.HoughLines(img, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)
        pslope = 0
        nslope = 0
        px1 = 0
        py1 = 0
        px2 = 0
        py2 = 0
        nx1 = 0
        ny1 = 0
        nx2 = 0
        ny2 = 0
        for x in range(0,len(img)):
            for rho,theta in img[x]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                slope = 0
                if x2-x1!=0:
                    slope = (y2-y1)/(x2-x1)
                if slope>0:
                    if slope>pslope:
                        pslope = slope
                        px1 = x1
                        py1 = y1
                        px2 = x2
                        py2 = y2
                elif slope<0:
                    if slope<nslope:
                        nslope = slope
                        nx1 = x1
                        ny1 = y1
                        nx2 = x2
                        ny2 = y2
        cv2.line(base,(px1,py1),(px2,py2),(0,0,255),4)
        cv2.line(base,(nx1,ny1),(nx2,ny2),(0,0,255),4)
        return base
def largestBlock(vec,thresh):
    sample_vec = [vec[0]]
    ret_vec = []
    for x in range(1,len(vec)):
        if(abs(vec[x-1]-vec[x])<thresh):
            sample_vec.append(vec[x])
            if len(sample_vec)>len(ret_vec):
                ret_vec = sample_vec
            # print(ret_vec)
        else:
            sample_vec = [vec[x]]
    return ret_vec
def blocks(vec,thresh):
    sample_vec = [vec[0]]
    ret_vec = []
    for x in range(1,len(vec)):
        if(abs(vec[x-1]-vec[x])<thresh):
            sample_vec.append(vec[x])
        else:
            ret_vec.append(sample_vec)
            sample_vec = [vec[x]]
    ret_vec.append(sample_vec)
    return ret_vec
def steepestblock(vec,thresh):
    vec = blocks(vec,thresh)
    maxSlope = 0
    for x in range(0,len(vec)):
        slope = abs(np.mean(vec[x]))
        if slope > maxSlope and len(vec[x]) >= 20:
            maxSlope = slope
            ret_block = vec[x]
    return ret_block
