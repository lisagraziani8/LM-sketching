"""
Image processing module that extract segments from skecth images.
Sketch matrices are (N+2,4), where each segment is represented as [x0,y0,x1,y1].

HEURISTIC 4:
- Extract contours from the image
- Order contours by bigger area
- Build polygons on the contours
- Order segments of each polygon taking as the first segment the one with the smallest abscissa and traverse
the polygon in anti-clock wise direction
- Select only the sketches with less than N segments.
"""


import cv2
import numpy as np
from utils import normalization
#import matplotlib.pyplot as plt
#import os
from utils import scale_color

class SegmentExtractionModule:

    def __init__(self, approx=0.004, min_area=10, N=50): 
        self.N = N  # n. max of segments
        self.w = 200  # image width
        self.h = 200  # image height
        self.approx_par = approx  # parameter to approximate contour into polygon
        self.min_area = min_area  # minimum area of a contour
        self.lower = 100  # canny edges detector
        self.upper = 200  # canny edges detector
        self.max_value = 199.0  # maximum value that can assume a segment coordinate
        print('**** [SEM] initialized ***')

    def detect_shape(self, c):
        #approximate a contour with polygon
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, self.approx_par * peri, True)
        return approx

    def build_segment_matrix(self, z):
        """
        Build segment matrix from sketch image
        :param z: grayscale image
        :return: norm_matrix: normalized sketch matrix (N+2,4)
                 up: number of lines of the sketch
                 segm_img: image representation of the sketch with segments
                 save: (boolean) if saving the segments matrix
        """

        save = True
        gray = z.astype(np.uint8)
        edges = cv2.Canny(gray, self.lower, self.upper)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) #cv2.CHAIN_APPROX_SIMPLE removes all redundant points and compresses the contour,

        filtered_contours = []
        count = 0
        for c in contours:  # select contours with area greater than a threshold
            if cv2.contourArea(c) > self.min_area and hierarchy[:,count,3]==-1:
                filtered_contours.append(c)
            count = count + 1

        filtered_contours.sort(reverse=True, key=cv2.contourArea)  # order contours by bigger area

        norm_matrix = np.zeros((self.N + 2, 4), dtype=float)
        segm_img = np.ones((self.w, self.h,3), dtype=np.uint8) * 255
        n_lines = 0
        ordered_lines = np.zeros((self.N, 4), dtype=int)

        for c in filtered_contours:
            first = True
            approx = self.detect_shape(c)
            points = approx[:, 0, :]
            p = points.shape[0]
            S = np.zeros((p, 4))
            for i in range(p):
                x = points[i,0]
                y = points[i,1]
                if first:
                    S[i, 0] = x
                    S[i, 1] = y
                    S[i + p - 1, 2] = x
                    S[i + p - 1, 3] = y
                    first = False
                else:
                    S[i - 1, 2] = x
                    S[i - 1, 3] = y
                    S[i, 0] = x
                    S[i, 1] = y

            abscissas1 = S[:,0]
            min_idx = np.argmin(abscissas1)

            L = np.zeros((p,4))
            L[0,:] = S[min_idx,:]
            L[:p - min_idx, :] = S[min_idx:, :]
            L[p - min_idx:, :] = S[:min_idx, :]

            if n_lines + p < self.N:
                ordered_lines[n_lines:n_lines+p,:] = L
            else:
                save = False
                break #col break qui si ottengono solo curve chiuse
                ordered_lines[n_lines:, :] = L[:self.N - n_lines, :]
                n_lines = n_lines + p

            n_lines = n_lines + p

        up = min(n_lines, self.N)

        try:
            if save==True:  # save sketch with less than N segments
                norm_segments = normalization(ordered_lines[:up, :])
                norm_matrix[1:up + 1, :] = norm_segments  # starting point is [0,0,0,0]
                norm_matrix[up+1, :] = 1.0  # ending point is [1,1,1,1]

                for i in range(up):  # draw sketch with extracted segments
                     r, g, b = scale_color(i)
                     cv2.line(segm_img,
                              (int(round(norm_matrix[i+1, 0]*self.max_value)), int(round(norm_matrix[i+1, 1]*self.max_value))),
                              (int(round(norm_matrix[i+1, 2]*self.max_value)), int(round(norm_matrix[i+1, 3]*self.max_value))),
                              (r, g, b), 1)

        except:
            print('*** no lines')
            save = False

        return norm_matrix, up, segm_img, save


if __name__=="__main__":
    import os

    category = "pear"
    sem = SegmentExtractionModule()
    src = 'DATASETS/rendered_sketch/' + category

    dst_dir = 'segment_images/' + category
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for frame in os.listdir(src):
        name = frame.split('.')[0]
        img = cv2.imread(src + '/' + frame,0)
        img = cv2.resize(img, (sem.w, sem.h))
        z_matrix, up, obj, save = sem.build_segment_matrix(img)
        rgb = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)

        print(frame, save)

        if save == True:
            cv2.imwrite(dst_dir + '/' + name + '_' + str(up) + '.jpg', rgb)
