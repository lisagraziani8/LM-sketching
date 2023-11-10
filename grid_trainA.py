import os
import numpy as np
import argparse
import tensorflow as tf
from grid_rnnA import Config, RNN
from utils import create_minibatches, find_last_checkpoint, get_perturbed_matrix
from grid_eval import draw_samples_train, calculate_measuresA
from gridding import Grid
from beam_search import generate_beam_searchA

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

tf.compat.v1.disable_eager_execution()

def parse_args():
    desc = "model A"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--lambda_e', type=float, default=0.01, help='weight for input embedding constraint')
    parser.add_argument('--activation', type=str, default='None', help='activation function nel FL. None, tanh, relu, sigmoid')
    return parser.parse_args()

if __name__ == '__main__':
    beam_size = 5
    args = parse_args()
    src_dir = 'rnn_matrices'
    conf = Config(args)
    grid = Grid()

    # During training data augmentation on train set is performed
    train_data = []
    grid_train_data = []
    for file in os.listdir(src_dir + '/train'):
        matrix = np.load(src_dir + '/train/' + file)
        train_data.append(matrix)
        grigliato = grid.from4to2(matrix)
        grid_train_data.append(grigliato)

    train_data = np.array(train_data)
    grid_train_data = np.array(grid_train_data)
    print('train_data', train_data.shape)

    grid_val_data = np.load(src_dir + '/grid_val.npy')
    grid_test_data = np.load(src_dir + '/grid_test.npy')
    grid_test_samples = np.load(src_dir + '/grid_test_samples.npy')  # a subset of test set in which to do beam search

    if not os.path.exists(conf.main_dir):
        os.mkdir(conf.main_dir)
    if not os.path.exists(conf.checkpoint):
        os.mkdir(conf.checkpoint)
    if not os.path.exists(conf.train_samples_dir):
        os.mkdir(conf.train_samples_dir)

    n_train = train_data.shape[0]
    n_val = grid_val_data.shape[0]
    n_test = grid_test_data.shape[0]
    log_file_name = 'grid_' + conf.date
    f = open(conf.main_dir + '/' + log_file_name + '.txt', 'w')
    f.write("lambda_e: " + str(args.lambda_e) + '\n')
    f.write("N: " + str(conf.N) + '\n')
    f.write("g: " + str(conf.g) + '\n')
    f.write("units: " + str(conf.units) + '\n')
    f.write("emb size: " + str(conf.emb_size) + '\n')
    f.write("neurons: " + str(conf.neurons) + '\n')
    f.write("activation: " + args.activation + '\n')
    f.write("lr: " + str(conf.lr) + '\n')
    f.write("epochs: " + str(conf.epochs) + '\n')
    f.write("patience: " + str(conf.patience) + '\n')
    f.write("batch size: " + str(conf.batch_size) + '\n')
    f.write("train: " + str(n_train) + '\n')
    f.write("val: " + str(n_val) + '\n')
    f.write("test: " + str(n_test) + '\n \n')
    f.close()

    X_ph = tf.compat.v1.placeholder(tf.int32, [None, conf.N + 2, 2])  # input
    rnn = RNN(X_ph, conf)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=conf.lr)

    if args.lambda_e > 0:  # add spatial relationship constraint
        loss_emb_in_ph = rnn.constraint_emb_in(conf)
    else:
        loss_emb_in_ph = tf.constant(0, dtype=tf.float32)

    loss_ph = rnn.loss_ce_ph + args.lambda_e*loss_emb_in_ph
    train_op = optimizer.minimize(loss_ph)

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(init)
        max_acc = 0
        bad_epochs = 0
        total_batch_train = int(train_data.shape[0] / conf.batch_size)  # number of batches in training set
        total_batch_val = int(grid_val_data.shape[0] / conf.batch_size)
        total_batch_test = int(grid_test_data.shape[0] / conf.batch_size)
        print('total batch', total_batch_train)
        f = open(conf.main_dir + '/' + log_file_name + '.txt', 'a+')

        for epoch in range(conf.epochs):
            print('***' + str(epoch) + '***')
            sum_loss = 0  # total loss (cross entropy + constraint)
            sum_loss_ce = 0  # cross entropy loss. If lambda_e=0 --> sum_loss=sum_loss_ce
            sum_acc_train = 0
            idxs_train = create_minibatches(train_data, conf.batch_size)
            for idx in idxs_train:
                X_batch_train = train_data[idx]
                # DATA AUGMENTATION
                aug_train = []
                for i in range(X_batch_train.shape[0]):
                    matrix = X_batch_train[i,:,:]
                    grid_matrix = grid.from4to2(matrix)
                    norm_aug_matrix = get_perturbed_matrix(matrix, conf.N, 0.05)  # slightly move some endpoints
                    aug_grid_matrix = grid.from4to2(norm_aug_matrix)
                    aug_train.append(grid_matrix)  # add to train set the original (alpha,beta) matrix
                    aug_train.append(aug_grid_matrix)  # add to train set the augmented (alpha, beta) matrix
                aug_train = np.array(aug_train)

                _, loss_batch, loss_ce_batch, acc_batch = sess.run([train_op, loss_ph, rnn.loss_ce_ph, rnn.accuracy_ph],
                                                                   feed_dict={X_ph: aug_train})

                sum_loss = sum_loss + loss_batch
                sum_loss_ce = sum_loss_ce + loss_ce_batch
                sum_acc_train = sum_acc_train + acc_batch

            loss = sum_loss / total_batch_train
            loss_ce = sum_loss_ce / total_batch_train
            acc_train = sum_acc_train / (n_train * 2)  # with data augmentation the number of training examples is doubled

            sum_acc_val = 0
            idxs = create_minibatches(grid_val_data, conf.batch_size)
            for idx in idxs:
                X_batch_val = grid_val_data[idx]
                SMacc_batch_val = sess.run(rnn.accuracy_ph, feed_dict={X_ph: X_batch_val})
                sum_acc_val = sum_acc_val + SMacc_batch_val
            acc_val = sum_acc_val / n_val

            print('loss', loss)
            print('loss ce', loss_ce)
            print('acc. train', acc_train)
            print('acc. val', acc_val)
            f.write("epoch " + str(epoch) + '\n')
            f.write("loss: %.8f" % loss + '\n')
            f.write("loss ce: %.8f" % loss_ce + '\n')
            f.write("acc. train: %.6f" % acc_train + '\n')
            f.write("acc. val: %.6f" % acc_val + '\n')

            if epoch % conf.save_freq == 0:  # draw segments generated at this epoch
                outputs_train = sess.run(rnn.outputs_ph, feed_dict={X_ph: aug_train})
                draw_samples_train(outputs_train, aug_train, epoch, conf.train_samples_dir)

            if acc_val > max_acc:  # early stopping on validation accuracy
                max_acc = acc_val
                saver.save(sess, os.path.join(conf.checkpoint, conf.model_name), global_step=epoch)  # save weights
                bad_epochs = 0
            else:
                bad_epochs = bad_epochs + 1

            if bad_epochs > conf.patience or acc_train==1:
                last_checkpoint = find_last_checkpoint(conf.checkpoint, conf.model_name)
                print(last_checkpoint)
                saver.restore(sess, os.path.join(conf.checkpoint, last_checkpoint))

                ##### TRAIN ##########
                outputs_train = sess.run(rnn.outputs_ph, feed_dict={X_ph: aug_train})
                draw_samples_train(outputs_train, aug_train, epoch, conf.train_samples_dir)

                SMacc_train, SMsoft_acc_train, px_train = calculate_measuresA(grid_train_data, n_train, sess, conf, rnn, X_ph)

                print('SMacc train', SMacc_train)
                print('SMsoft_acc train', SMsoft_acc_train)
                print('px train', px_train)
                f.write('\n')
                f.write("SMacc train: %.6f" % SMacc_train + '\n')
                f.write("SMsoft_acc train: %.6f" % SMsoft_acc_train + '\n')
                f.write("px train: %.6f" % px_train + '\n')

                SMacc_val, SMsoft_acc_val, px_val = calculate_measuresA(grid_val_data, n_val, sess, conf, rnn, X_ph)
                print('SM acc. val', SMacc_val)
                print('SM soft acc val', SMsoft_acc_val)
                print('px val', px_val)
                f.write("SM_acc_val: %.6f" % SMacc_val + '\n')
                f.write("SM soft acc val: %.6f" % SMsoft_acc_val + '\n')
                f.write("px val: %.8f" % px_val + '\n')

                SMacc_test, SMsoft_acc_test, px_test = calculate_measuresA(grid_test_data, n_test, sess, conf, rnn, X_ph)
                print('SM acc. test', SMacc_test)
                print('SM soft acc test', SMsoft_acc_test)
                print('px test', px_test)
                f.write("SM_acc_test: %.6f" % SMacc_test + '\n')
                f.write("SM soft acc_test: %.6f" % SMsoft_acc_test + '\n')
                f.write("px test: %.8f" % px_test + '\n')

                if not os.path.exists(conf.test_gen_dir):
                    os.mkdir(conf.test_gen_dir)

                # beam search generation on a reduced test sample
                generate_beam_searchA(grid_test_samples, sess, conf.test_gen_dir, rnn, conf, X_ph, beam_size)

                f.close()
                exit()

        saver.save(sess, os.path.join(conf.checkpoint, conf.model_name), global_step=epoch)
        f.close()