import cv2
import numpy as np
from gridding import Grid
from utils import create_minibatches

grid = Grid()

def draw_samples_train(outputs, inputs, epoch, directory):
    """
    Given a train example create an image with drawn input segments, and another for output
    :param outputs: output matrix [N+2,2]
    :param inputs: input matrix [N+2,2]
    :param epoch: number of the current epoch
    :param directory: directory where to save the images
    :return:
    """
    for i in range(outputs.shape[0]):
        coord_in = grid.from2to4(inputs[i, :, :])
        s_in, n_lines_in = grid.draw_lines(coord_in, 1)
        cv2.imwrite(directory + '/' + str(epoch) + '_' + str(i) + '_input_' + str(n_lines_in) + '.jpg', s_in)
        coord_out = grid.from2to4(outputs[i,:,:])
        s_out, n_lines_out = grid.draw_lines(coord_out[:n_lines_in, :], 0)
        cv2.imwrite(directory + '/' + str(epoch) + '_' + str(i) + '_output_' + str(n_lines_out) + '.jpg', s_out)



def SMsoft_accuracy(grid_target, grid_output):
    """
    Soft accuracy: accuracy considering correct the predictions on the 8 cells surrounding the target cell
    :param grid_target: target matrix (n_examples,N+2,2)
    :param grid_output: output matrix (n_examples,N+2,2)
    :return: sum of the accuracy of all examples
    """
    g = grid.g
    accuracy = 0
    for i in range(grid_target.shape[0]):
        acc_ex = 0
        grid_in = grid_target[i, :, :]
        grid_out = grid_output[i, :, :]
        n_seg = grid_in[grid_in[:, 0] != g *g].shape[0] - 1  # number of segments

        start = 0 #1
        end = n_seg #n_seg+1

        for j in range(start,end):
            alfa_in = grid_in[j,0]
            beta_in = grid_in[j,1]
            alfa_out = grid_out[j, 0]
            beta_out = grid_out[j, 1]

            if alfa_in == g*g or alfa_out==g*g or alfa_in == g*g + 1 or alfa_out==g*g + 1: #alfa=startID or stopID
                if alfa_in == alfa_out:
                    acc_ex = acc_ex + 1
            else: #same cell or 8 surrounding cells
                if alfa_in == alfa_out or alfa_in == alfa_out + g or alfa_in == alfa_out + g + 1 or alfa_in == alfa_out + g - 1\
                    or alfa_in == alfa_out - g or  alfa_in == alfa_out - g + 1 or alfa_in == alfa_out - g - 1:
                    acc_ex = acc_ex + 1

            if beta_in == g * g or beta_out == g * g or beta_in == g * g + 1 or beta_out == g * g + 1:
                if beta_in == beta_out:
                    acc_ex = acc_ex + 1
            else:
                if beta_in == beta_out or beta_in == beta_out + g or beta_in == beta_out + g + 1 or beta_in == beta_out + g - 1 \
                    or beta_in == beta_out - g or beta_in == beta_out - g + 1 or beta_in == beta_out - g - 1:
                    acc_ex = acc_ex + 1

        acc_ex = acc_ex / (2*n_seg)  # alpha and beta --> divided by 2
        accuracy = accuracy + acc_ex

    return accuracy

def calculate_measuresA(grid_data, n, sess, conf, rnn, X_ph):
    """
    Calculate accuracy, soft accuracy, perplexity with mini-batches, to avoid memory problems
    :param grid_data: numpy array with grid data (n, N+2, 2)
    :param n: number of examples
    :param sess: tensorflow session
    :param conf: Conf class
    :param rnn: RNN class
    :param X_ph: input placeholder
    :return: accuracy, soft_accuracy, perplexity
    """
    sum_acc = 0
    sum_soft_acc = 0
    sum_loss_alfa = 0
    sum_loss_beta = 0
    idxs = create_minibatches(grid_data, conf.batch_size)
    for idx in idxs:
        X_batch = grid_data[idx]
        SMacc_batch, loss_alfa_batch, loss_beta_batch, output = sess.run(
            [rnn.accuracy_ph, rnn.loss_alfa, rnn.loss_beta, rnn.outputs_ph],
            feed_dict={X_ph: X_batch})

        SMsoft_acc_batch = SMsoft_accuracy(X_batch[:, 1:, :], output)

        sum_acc = sum_acc + SMacc_batch
        sum_soft_acc = sum_soft_acc + SMsoft_acc_batch
        sum_loss_alfa = sum_loss_alfa + np.sum(loss_alfa_batch)
        sum_loss_beta = sum_loss_beta + np.sum(loss_beta_batch)

    SMsoft_acc = sum_soft_acc / n
    SMacc = sum_acc / n
    px_alfa = np.exp(sum_loss_alfa / n)
    px_beta = np.exp(sum_loss_beta / n)
    px = 0.5 * (px_alfa + px_beta)

    return SMacc, SMsoft_acc, px