"""
Convert segment matrices (N+2)x4 into grid matrices (N+2)x2 for train, val and test set.
Grid matrices are concatenated and saved in a numpy array for train, val and test.
"""

import os
import numpy as np
from gridding import Grid

grid = Grid()
src_dir = 'rnn_matrices'

train_data = []
for file in os.listdir(src_dir + '/train'):
    matrix = np.load(src_dir + '/train/' + file)
    grid_matrix = grid.from4to2(matrix)
    train_data.append(grid_matrix)

train_data = np.array(train_data)
print('train', train_data.shape)
np.save(src_dir + '/grid_train.npy', train_data)

val_data = []
for file in os.listdir(src_dir + '/validation'):
    matrix = np.load(src_dir + '/validation/' + file)
    grid_matrix = grid.from4to2(matrix)
    val_data.append(grid_matrix)

val_data = np.array(val_data)
print('val', val_data.shape)
np.save(src_dir + '/grid_val.npy', val_data)

test_data = []
for file in os.listdir(src_dir + '/test'):
    matrix = np.load(src_dir + '/test/' + file)
    grid_matrix = grid.from4to2(matrix)
    test_data.append(grid_matrix)

test_data = np.array(test_data)
print('test', test_data.shape)
np.save(src_dir + '/grid_test.npy', test_data)

test_data_samples = []
for file in os.listdir(src_dir + '/test_samples'):
    matrix = np.load(src_dir + '/test_samples/' + file)
    grid_matrix = grid.from4to2(matrix)
    test_data_samples.append(grid_matrix)

test_data_samples = np.array(test_data_samples)
print('test samples', test_data_samples.shape)
np.save(src_dir + '/grid_test_samples.npy', test_data_samples)

