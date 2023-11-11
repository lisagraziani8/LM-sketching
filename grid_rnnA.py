import tensorflow as tf
from gridding import Grid
from datetime import datetime

class Config:
    def __init__(self, args, N=50, g=20):
        self.units = 200  # number of units in RNN
        self.lr = 0.001  # learning rate
        self.epochs = 5000
        self.batch_size = 128
        self.N = N  # maximum number of segments
        self.save_freq = 20  # how often (epochs) to save generated sketch from training set
        self.patience = 10
        self.emb_size = 50  # embedding size
        self.g = g  # number of cell of the grid in a side
        self.neurons = 300  # number of neurons in the fully connected layers

        if args.activation == 'tanh':
            self.activation = tf.nn.tanh
        elif args.activation == 'sigmoid':
            self.activation = tf.nn.sigmoid
        elif args.activation == 'relu':
            self.activation = tf.nn.relu
        else:  # only a layer with g*g+2 neurons
            self.activation = None

        now = datetime.now()
        self.date = now.strftime("%d%b%Y-%H.%M")
        self.main_dir = 'gridA_' + self.date  # main directory where to save results and checkpoints of the current experiment
        self.model_name = 'gridA.ckpt'
        self.train_samples_dir = self.main_dir + '/TRAIN-SM-QUAL_FINAL'
        self.train_gen_dir = self.main_dir + '/TRAIN-GEN-QUAL_SEQ'
        self.test_gen_dir = self.main_dir + '/TEST-GEN-QUAL_SEQ'
        self.checkpoint = self.main_dir + '/checkpoint'
        self.grid = Grid()


class RNN():
    def __init__(self, X_ph, conf):
        alfa = X_ph[:,:,0]
        beta = X_ph[:,:,1]
        self.embeddings = tf.compat.v1.get_variable("InputEmbeddings", [conf.g*conf.g+2, conf.emb_size], dtype=tf.float32)
        wes_alfa = tf.nn.embedding_lookup(self.embeddings, alfa)  # batch_size x sentence_max_len x word_emb_size
        wes_beta = tf.nn.embedding_lookup(self.embeddings, beta)  # batch_size x sentence_max_len x word_emb_size
        concat = tf.concat([wes_alfa, wes_beta], axis=2)
        lstm = tf.keras.layers.LSTM(conf.units, return_sequences=True, time_major=False)
        lstm_output = lstm(concat)  # (None,N,units)

        if conf.activation != None:
            fl1_alfa = tf.compat.v1.layers.dense(lstm_output[:,:-1,:], conf.neurons, name='fl1_alfa', activation=conf.activation)
            fl1_beta = tf.compat.v1.layers.dense(lstm_output[:,:-1,:], conf.neurons, name='fl1_beta', activation=conf.activation)
            logits_alfa = tf.compat.v1.layers.dense(fl1_alfa, conf.g*conf.g+2, name='logits_alfa')
            logits_beta = tf.compat.v1.layers.dense(fl1_beta, conf.g*conf.g+2, name='logits_beta')
        else:
            logits_alfa = tf.compat.v1.layers.dense(lstm_output[:,:-1,:], conf.g * conf.g + 2, name='logits_alfa')
            logits_beta = tf.compat.v1.layers.dense(lstm_output[:,:-1,:], conf.g * conf.g + 2, name='logits_beta')

        target_alfa = alfa[:, 1:]
        target_beta = beta[:, 1:]
        mask = tf.cast(tf.not_equal(target_alfa, conf.g*conf.g), dtype=tf.float32) # to take only rows representing segments
        print('mask',mask)
        length = tf.reduce_sum(mask, axis=1)  # number of segments

        onehot_alfa = tf.one_hot(target_alfa, conf.g*conf.g +2, axis=2)
        onehot_beta = tf.one_hot(target_beta, conf.g*conf.g +2, axis=2)
        ce_alfa = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_alfa, logits=logits_alfa)
        ce_beta = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_beta, logits=logits_beta)
        # cross entropy only on segments
        ce_masked_alfa = ce_alfa * mask
        ce_masked_beta = ce_beta * mask
        self.loss_alfa = tf.compat.v1.div(tf.reduce_sum(ce_masked_alfa, axis=1), length)
        self.loss_beta = tf.compat.v1.div(tf.reduce_sum(ce_masked_beta, axis=1), length)
        self.loss_ce_ph = tf.reduce_mean(self.loss_alfa + self.loss_beta)  # cross entropy of alpha and beta

        self.perplexity = 0.5*(tf.exp(tf.reduce_mean(self.loss_alfa)) + tf.exp(tf.reduce_mean(self.loss_beta)))

        self.p_soft_alfa = tf.nn.softmax(logits_alfa, axis=2)
        self.p_soft_beta = tf.nn.softmax(logits_beta, axis=2)
        predictions_alfa = tf.argmax(self.p_soft_alfa, axis=2)
        predictions_beta = tf.argmax(self.p_soft_beta, axis=2)
        alfa_out = tf.expand_dims(predictions_alfa,axis=2)
        beta_out = tf.expand_dims(predictions_beta,axis=2)
        self.outputs_ph = tf.concat([alfa_out,beta_out], axis=2)

        correct_pred_alfa = tf.equal(tf.cast(predictions_alfa[:,1:], dtype=tf.int32), target_alfa[:,1:])  # accuracy is not calculated on the first segment
        correct_pred_beta = tf.equal(tf.cast(predictions_beta[:,1:], dtype=tf.int32), target_beta[:,1:])
        acc_alfa = tf.cast(correct_pred_alfa, tf.float32)*mask[:,1:]
        acc_beta = tf.cast(correct_pred_beta, tf.float32)*mask[:,1:]
        self.accuracy_ph = tf.reduce_sum(tf.compat.v1.div(tf.reduce_sum((acc_alfa + acc_beta)/2, axis=1), length - 1))


    def constraint_emb_in(self, conf):
        """
        Constraint on spatial relationship of grid cell.
        Cells that are nearby are enforced to be embedded into similar vector
        :param conf: Config class
        :return: loss for spatial relationship
        """

        #emb_gg = self.embeddings[1:conf.g*conf.g+1, :]
        emb_gg = self.embeddings[:-2, :]
        print('emb_gg', emb_gg)
        e = tf.reshape(emb_gg, [conf.g, conf.g, conf.emb_size])
        norm1 = tf.reduce_sum(tf.square(e[1:,:,:]-e[:-1,:,:]))
        norm2 = tf.reduce_sum(tf.square(e[:,1:,:]-e[:,:-1,:]))
        norm3 = tf.reduce_sum(tf.square(e[1:,1:,:]-e[:-1,:-1,:]))
        norm4 = tf.reduce_sum(tf.square(e[1:,:-1,:]-e[:-1,1:,:]))
        loss_emb_ph = norm1+norm2+norm3+norm4
        return loss_emb_ph
