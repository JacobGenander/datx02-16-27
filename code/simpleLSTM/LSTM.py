#! /usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import interfaceLSTM
import plot
import time
import sys
import os
from DataMan import DataMan
from hyperParams import *

# TensorFlow's API (if you want to know what arguments we pass to the different methods)
# https://www.tensorflow.org/versions/r0.7/api_docs/python/index.html

class LSTM_Network(object):

    def __init__(self, training, batch_size):
        self.batch_size = batch_size
        # 2-dimensional tensors for input data and targets
        self._input = tf.placeholder(tf.int32, [batch_size, MAX_SEQ])
        self._target = tf.placeholder(tf.int64, [batch_size, MAX_SEQ])
        # This is the length of each sentence
        self._seq_lens = tf.placeholder(tf.int32, [batch_size])
        # We need these to crop output and targets so we do not train on more time sequences than needed
        self._out_dim = tf.placeholder(tf.int32, [2])
        self._target_dim = tf.placeholder(tf.int32, [2])

        # Fetch word vectors
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding",
                    [DataMan.vocab_size, EMBEDDING_SIZE])
            inputs = tf.nn.embedding_lookup(embedding, self._input)

        if KEEP_PROB < 1 and training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)

        # Create the network
        cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_LAYER_SIZE)
        if KEEP_PROB < 1 and training:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=KEEP_PROB)
        stacked_cells = tf.nn.rnn_cell.MultiRNNCell([cell] * NUMBER_OF_LAYERS)

        self._initial_state = state = stacked_cells.zero_state(batch_size, tf.float32)

        # Give input the right shape
        inputs = [ tf.squeeze(input_, [1]) for input_ in tf.split(1, MAX_SEQ, inputs)]

        # Run through the whole batch and update state
        outputs, _ = tf.nn.rnn(stacked_cells, inputs, initial_state=self._initial_state, sequence_length=self._seq_lens)
        # Turn output into tensor instead of list
        outputs = tf.concat(1, outputs)

        # Crop output and targets to the length of the longest sentence
        outputs = tf.slice(outputs, [0, 0], self._out_dim)
        target = tf.slice(self._target, [0, 0], self._target_dim)

        # Flatten output into a tensor where each row is the output from one word
        output = tf.reshape(outputs, [-1, HIDDEN_LAYER_SIZE])

        w = tf.get_variable("out_w", [HIDDEN_LAYER_SIZE, DataMan.vocab_size])
        b = tf.get_variable("out_b", [DataMan.vocab_size])
        z = tf.matmul(output, w) + b # Add supports broadcasting over each row

        # Average negative log probability
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(z, tf.reshape(target, [-1]))
        self._cost = cost = tf.reduce_sum(loss) / batch_size

        if not training:
            self._train_op = tf.no_op()
            return

        self._learning_rate = tf.Variable(LEARNING_RATE, trainable=False)
        # Gradient descent training op
        optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
        self._train_op = optimizer.minimize(self._cost)

    def set_learning_rate(self, sess, value):
        sess.run(tf.assign(self._learning_rate, value))

    def lr_decay_and_set(self, sess, epoch):
        if epoch > DECAY_START:
            decay = LEARNING_DECAY ** (epoch - DECAY_START)
            self.set_learning_rate(sess, LEARNING_RATE * decay)

    def calc_output_dims(self, seq_lens):
        max_batch_seq = max(seq_lens)
        dim_output = np.array([self.batch_size, max_batch_seq * HIDDEN_LAYER_SIZE])
        dim_target = np.array([self.batch_size, max_batch_seq])
        return dim_output, dim_target

def run_epoch(sess, data_set, net, perplexity):
    total_cost = 0.0
    steps = 0
    for i, (x, y, z) in enumerate(data_set.batch_iterator(net.batch_size)):
        d_out, d_target = net.calc_output_dims(z)
        steps += max(z)
        # Input
        feed = { net._input : x, net._target : y, net._seq_lens : z, net._out_dim : d_out, net._target_dim : d_target}
        # Run the computational graph
        cost, _ = sess.run([net._cost, net._train_op], feed_dict=feed)
        total_cost += cost

    if perplexity:
        return np.exp(total_cost / steps)
    else:
        return total_cost / (i+1)

def save_state(sess, saver):
    print("Saving model.")
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in file: {}".format(save_path))

def create_data_sets(data_path):
    try:
        train_path = os.path.join(data_path, "train.txt")
        training_set = DataMan(train_path, MAX_SEQ)
        valid_path = os.path.join(data_path, "valid.txt")
        validation_set = DataMan(valid_path, MAX_SEQ, rebuild_vocab=False)
        test_path = os.path.join(data_path, "test.txt")
        test_set = DataMan(test_path, MAX_SEQ, rebuild_vocab=False)
    except IOError:
        print("File not found. Data path needs to contain three files: train.txt, valid.txt and test.txt")
        sys.exit(1)

    return training_set, validation_set, test_set

def main():
    start_time = time.time()

    args = interfaceLSTM.parser.parse_args()
    training_set, validation_set, test_set = create_data_sets(args.data_path)

    save_path = args.save_path
    if not os.path.isdir(save_path):
        print("Couldn't find save directory")
        sys.exit(1)

    initializer = tf.random_uniform_initializer(-INIT_RANGE, INIT_RANGE)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        train_net = LSTM_Network(True, BATCH_SIZE)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        val_net = LSTM_Network(False, BATCH_SIZE)
        test_net = LSTM_Network(False, 1)

    # We always need to run this operation before anything else
    init = tf.initialize_all_variables()
    # This operation will save our state at the end
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)

        print("Training.")
        cost_train, cost_valid = [], []
        cost_valid = []
        for i in range(MAX_EPOCH):
            print("\r{}% done".format(int(i/MAX_EPOCH * 100)))
            train_net.lr_decay_and_set(sess, i)
            cost = run_epoch(sess, training_set, train_net, False)
            cost_train.append(cost)
            cost = run_epoch(sess, validation_set, val_net, False)
            cost_valid.append(cost)
        print("100% done")

        print("Calculating perplexity.")
        perplexity = run_epoch(sess, test_set, test_net, True)
        print("Perplexity: {}".format(perplexity))

        print("Creating plot.")
        plot.create_plots(save_path, range(MAX_EPOCH), cost_train, cost_valid)

        save_state(sess, saver)
        print("--- {} seconds ---".format(round(time.time() - start_time, 2)))

if __name__ == "__main__":
    main()

