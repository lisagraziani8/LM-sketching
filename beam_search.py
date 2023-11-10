import cv2
import numpy as np
from gridding import Grid
import os

grid = Grid()

def generate_beam_searchA(grid_data, sess, directory, rnn, conf, X_ph, beam_size):
    """
    For each input matrix generate beam_size different examples.
    The sketches are generated starting from the first half of the segments of the input sketch.
    :param grid_data: numpy array of input sketch (n_examples, N+2, 2)
    :param sess: Tensorflow session
    :param directory: directory in which to save generated sketch
    :param rnn: RNN class
    :param conf: Conf class
    :param X_ph: input matrix placeholder
    :param beam_size: beam size
    :return:
    """

    print('BEAM SEARCH')
    for n_ex in range(grid_data.shape[0]):  # iterate on the number of examples

        if not os.path.exists(directory + '/' + str(n_ex)):
            os.makedirs(directory + '/' + str(n_ex))

        grid_matrix = grid_data[n_ex, :, :]

        #draw target
        coord = grid.from2to4(grid_matrix)
        img, n_lines = grid.draw_lines(coord, 1)
        cv2.imwrite(directory + '/' + str(n_ex) + '/0input_' + str(n_lines) + '.jpg', img)

        # sketch with half segments (sketch to complete)
        n = grid_matrix[grid_matrix[:,0] != conf.g*conf.g].shape[0] - 1  # number of segments
        half = np.ones((conf.N + 2, 2),dtype=int)*conf.g*conf.g
        M = int(n / 2)
        half[:M + 1, :] = grid_matrix[:M + 1, :]
        half[M + 1, :] = [conf.g*conf.g+1,conf.g*conf.g+1]
        coord_half = grid.from2to4(half)
        img_half, n_lines_half = grid.draw_lines(coord_half, 1) # draw half sketch
        cv2.imwrite(directory + '/' + str(n_ex) + '/half_' + str(n_lines_half) + '.jpg', img_half)
        half1 = half.reshape((1, conf.N + 2, 2))

        probs = {}
        probs_temp = np.zeros((beam_size, beam_size))
        coords = {}
        alfas_temp = np.zeros((beam_size, beam_size))
        betas_temp = np.zeros((beam_size, beam_size))

        # predict next segment from half sketch
        p_alfa, p_beta = sess.run([rnn.p_soft_alfa, rnn.p_soft_beta], feed_dict={X_ph: half1})
        idxs_alfa = np.argsort(p_alfa[0, M])[-beam_size:]  # get the indices of the first beam_size alpha with the highest probability
        idxs_beta = np.argsort(p_beta[0, M])[-beam_size:]  # get the indices of the first beam_size beta with the highest probability

        for j in range(beam_size):
            coords[(M,j)] = (idxs_alfa[j], idxs_beta[j])
            probs[(M,j)] = 0.5*(p_alfa[0,M,idxs_alfa[j]] + p_beta[0,M,idxs_beta[j]])

        input_j = np.ones((1, conf.N + 2, 2), dtype=int) * conf.g * conf.g
        input_j[0, :M+ 1, :] = half1[0,:M+1,:]

        for m in range(M+1,n):
            list_prob = []
            for j in range(beam_size):
                for l in range(M,m):
                    input_j[0, l+1, 0] = coords[(l, j)][0]
                    input_j[0, l+1, 1] = coords[(l, j)][1]

                input_j[0, m+1, :] = [conf.g * conf.g + 1, conf.g * conf.g + 1]

                p_alfa, p_beta = sess.run([rnn.p_soft_alfa, rnn.p_soft_beta], feed_dict={X_ph: input_j})
                idxs_alfa_j = np.argsort(p_alfa[0, m])[-beam_size:]
                idxs_beta_j = np.argsort(p_beta[0, m])[-beam_size:]

                for k in range(beam_size):
                    alfas_temp[j,k] = idxs_alfa_j[k]
                    betas_temp[j,k] = idxs_beta_j[k]
                    pjk = probs[(m-1,j)]*0.5*(p_alfa[0,m,idxs_alfa_j[k]] + p_beta[0,m,idxs_beta_j[k]])
                    probs_temp[j,k] = pjk
                    list_prob.append(pjk)

            prob_array = np.array(list_prob)
            idxs = np.argsort(prob_array)[-beam_size:]
            prev_beams = (idxs / beam_size).astype(int)
            k_beams = idxs % beam_size

            coords_copy = coords.copy()

            for h in range(beam_size):
                coords[(m,h)] = (alfas_temp[prev_beams[h], k_beams[h]], betas_temp[prev_beams[h], k_beams[h]])
                probs[(m,h)] = prob_array[idxs[h]]
                coords[(m-1,h)] = coords_copy[(m-1,prev_beams[h])]

        # For each beam, save an image for each generated segment.
        # For each beam save the numpy array of the generated sketch.
        for k in range(beam_size):
            for m in range(M,n):
                input_fin = np.ones((conf.N + 2, 2), dtype=int) * conf.g * conf.g
                input_fin[:M + 1, :] = half1[0, :M + 1, :]
                for l in range(M,m+1):
                    input_fin[l + 1, 0] = coords[(l, k)][0]
                    input_fin[l + 1, 1] = coords[(l, k)][1]

                input_fin[m + 2, :] = [conf.g * conf.g + 1, conf.g * conf.g + 1]

                s, stop = grid.draw_testOR(input_fin, m)

                if stop == True:  # stop drawing
                    break
                else:
                    cv2.imwrite(directory + '/' + str(n_ex) + '/output_' + str(k) + '_' + str(m+1) + '.jpg', s)

            np.save(directory + '/' + str(n_ex) + '/zcoord_' + str(k) + '_' + str(m) + '.npy', input_fin)