"""
The image is divided into a grid gxg
Segments endpoints are represented as integer numbers (alpha, beta)
which indicate the number of the cell of the grid gxg in which the endpoint lies.
"""

import numpy as np
import cv2

class Grid:

    def __init__(self, N=50, g=20):
        self.g = g  # number of squares in a side of the grid
        self.N = N  # maximum nuber of segments
        self.max_px = 199  # maximum value that can assume a segment coordinate
        self.l = (self.max_px + 1) / self.g  # number of pixels in a square of the grid
        self.w = 200  # image width
        self.h = 200  # image height


    def from4to2(self, coordinates):
        """
        Convert the segment coordinates into grid elements
        :param coordinates: segment matrix (N+2)x4 [x0,x1,y0,y1]
        :return: grid matrix (N+2)x2 [alpha, beta]
        """
        coord_int = (np.round(coordinates * self.max_px)).astype(int)
        grid_matrix = np.zeros((self.N+2, 2), dtype=int)
        end = False
        grid_matrix[0,:] = self.g*self.g  # start id=g*g
        for i in range(1, self.N+2):
            x1 = coord_int[i,0]
            y1 = coord_int[i,1]
            x2 = coord_int[i,2]
            y2 = coord_int[i,3]

            # if ending_point
            if x1 == self.max_px and y1 == self.max_px and x2 == self.max_px and y2 == self.max_px:
                grid_matrix[i,:] = self.g*self.g + 1  # end id = g*g+1
                end = True
            else:
                if end == False:
                    grid_matrix[i, 0] = int(y1/self.l) * self.g + int(x1 / self.l)
                    grid_matrix[i, 1] = int(y2/self.l) * self.g + int(x2 / self.l)
                else:
                    grid_matrix[i, :] = self.g * self.g  # put g*g where there are not segments

        return grid_matrix

    def from2to4(self, grid_data):
        """
        Convert segments of the grid [alpha, beta] into segments coordinates [x0,y0,x1,y1]
        :param grid_data: matrix (N+2,2)
        :return: matrix (N+2,4)
        """
        coord = np.zeros((self.N+2,4), dtype=int)
        for i in range(0, grid_data.shape[0]):
            alfa = grid_data[i,0]
            beta = grid_data[i,1]
            if alfa == self.g*self.g+1 and beta == self.g*self.g+1:
                coord[i,:] = self.max_px
                break
            else:
                if alfa == self.g*self.g: #startID
                    coord[i, 0] = 0
                    coord[i, 1] = 0
                elif alfa == self.g*self.g+1: #endID
                    coord[i,0]=self.max_px
                    coord[i,1]=self.max_px
                else:
                    coord[i,0]=(alfa -int(alfa/self.g)*self.g)*self.l + self.l/2
                    coord[i,1]=int(alfa/self.g)*self.l + self.l/2
                if beta == self.g*self.g: #startID
                    coord[i, 2] = 0
                    coord[i, 3] = 0
                elif beta == self.g*self.g+1: #endID
                    coord[i,2]=self.max_px
                    coord[i,3]=self.max_px
                else:
                    coord[i,2]=(beta -int(beta/self.g)*self.g)*self.l + self.l/2
                    coord[i,3]=int(beta/self.g)*self.l + self.l/2

        return coord


    def draw_lines(self, matrix, start):
        """
        Draw segments
        :param matrix: segment matrix [x1,y1,x2,y2]
        :param start: row index at which start drawing. If input start=1, if output start=0
        :return:
            white: image with drawn segments
            n_lines: number of drawn lines
        """

        n_lines = 0
        white = np.ones((self.w, self.h, 3)) * 255

        for i in range(start, matrix.shape[0]):
            x1 = matrix[i, 0]
            x2 = matrix[i, 2]
            y1 = matrix[i, 1]
            y2 = matrix[i, 3]

            if x1 == self.max_px and x2 == self.max_px and y1 == self.max_px and y2 == self.max_px:
                break

            cv2.line(white, (x1, y1), (x2, y2), (0, 0, 0), 1)
            n_lines = n_lines + 1

        return white, n_lines

    def from2to4_segment(self,alfa,beta):
        # convert (alpha,beta) endpoints into (x1,y1,x2,y2) endpoints
        x1 = int((alfa - int(alfa / self.g) * self.g) * self.l + self.l / 2)
        y1 = int(int(alfa / self.g) * self.l + self.l / 2)
        x2 = int((beta - int(beta / self.g) * self.g) * self.l + self.l / 2)
        y2 = int(int(beta / self.g) * self.l + self.l / 2)
        return x1,y1,x2,y2

    def draw_testOR(self, half, m):
        """
        Draw sketch from grid matrix and stops drawing if stopID alfa OR beta = g*g+1
        :param half: segments matrix (N+2,2)
        :param m: number of segments
        :return:
            sketch_img: image with drawn segments
            stop: boolean indicating if sketch is finished
        """
        stop = False
        sketch_img = np.ones((self.w, self.h, 3)) * 255

        for i in range(1, m+2):
            alfa = half[i, 0]
            beta = half[i, 1]

            if alfa == self.g*self.g + 1 or beta == self.g*self.g + 1:
                stop = True
                break

            else:
                x1,y1,x2,y2 = self.from2to4_segment(alfa,beta)

                if i == m + 1:  # draw last segment in red
                    cv2.line(sketch_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                else:
                    cv2.line(sketch_img, (x1, y1), (x2, y2), (0, 0, 0), 1)

        return sketch_img, stop
