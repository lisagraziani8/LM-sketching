"""
Build matrices of segments (N+2,4) from sketch images.
Segments are represented as [x0,y0,x1,y1], where (x0,y0) and (x1,y1) are segment endpoints
Split in train 80%, validation 10%, test 10%
"""

import os
import cv2
import numpy as np
from sem4 import SegmentExtractionModule

sem = SegmentExtractionModule()
dst_dir = 'rnn_matrices'  # folder where to save the segment matrices
train_size = 0.8  # percentage of the training set
val_size = 0.1  # percentage of the validation set

if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

def img2matrix(img):
    """
    Resize image and build matrix from it
    :param img: opencv image of the sketch
    :return: z_matrix: matrix of segments (N+2,4)
            n_lines: number of segments of the sketch
            save: (boolean) if saving the segments matrix
    """
    img = cv2.resize(img, (sem.w, sem.h))
    img_array = np.array(img)
    z_matrix, n_lines, _, save = sem.build_segment_matrix(img_array)
    return z_matrix, n_lines, save


def extract_matrix(src_dir):
    """
    Given a dataset of sketch images divided for category, build segment matrices, divide them in train, val, test,
    and save them
    :param src_dir: directory name of the dataset
    :return:
    """
    train_dir = dst_dir + '/train'
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    val_dir = dst_dir + '/validation'
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)

    test_dir = dst_dir + '/test'
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)


    for category in sorted(os.listdir(src_dir)):
        print(category)
        count = 0
        filename_list = os.listdir(src_dir + '/' + category)
        n = len(filename_list)
        n_train = int(round(n * train_size))
        n_val = int(round(n * val_size))

        for frame in filename_list:
            try:
                name = frame.split('.')[0]
                img = cv2.imread(src_dir + '/' + category + '/' + frame, 0)
                z_matrix, n_lines, save = img2matrix(img)

                if n_lines > 0 and save==True:
                    if count < n_train:
                        np.save(train_dir + '/' + category + '_' + name + '_' + str(n_lines) + '.npy', z_matrix)
                    elif count > n_train and count < n_val + n_train:
                        np.save(val_dir + '/' + category + '_' + name + '_' + str(n_lines) + '.npy', z_matrix)
                    else:
                        np.save(test_dir + '/' + category + '_' + name + '_' + str(n_lines) + '.npy', z_matrix)

                count = count + 1

            except:
                print('errore sketch ' + frame)



extract_matrix('DATASETS/rendered_sketch')



