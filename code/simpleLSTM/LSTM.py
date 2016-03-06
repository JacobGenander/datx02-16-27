#! /usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from DataMan import DataMan
from hyperParams import *

# TensorFlow's API (if you want to know what arguments we pass to the different methods)
# https://www.tensorflow.org/versions/r0.7/api_docs/python/index.html

class LSTM_Network(object):

    def __init__(self, data_set, training):
        initializer = tf.random_uniform_initializer(-init_range, init_range)
        max_seq = data_set.max_seq

        # 2-dimensional tensors for input data and targets
        self._input = tf.placeholder(tf.int32, [batch_size, None], name="input_data")
        self._target = tf.placeholder(tf.int64, [batch_size, None], name="target_data")
        # This is the length of each sentence
        self._seq_lens = tf.placeholder(tf.int32, [batch_size], name="sequence_lengths")

        # Fetch word vectors
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding",
                    [DataMan.vocab_size, embedding_size],
                    initializer=initializer)
            inputs = tf.nn.embedding_lookup(embedding, self._input)

        if keep_prob < 1 and not training:
            inputs = tf.nn.dropout(inputs, keep_prob)

        # Create the network
        cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_size)
        if keep_prob < 1 and not training:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        stacked_cells = tf.nn.rnn_cell.MultiRNNCell([cell] * number_of_layers)

        self._initial_state = state = stacked_cells.zero_state(batch_size, tf.float32)

        # Give input the right shape
        inputs = [ tf.squeeze(input_, [1]) for input_ in tf.split(1, max_seq, inputs)]

        # Run through the whole batch and update state
        outputs, _ = tf.nn.rnn(stacked_cells, inputs, initial_state=self._initial_state, sequence_length=self._seq_lens)

        # The output also needs some massaging
        output = tf.reshape(tf.concat(1, outputs), [-1, hidden_layer_size])

        w = tf.get_variable("out_w", [hidden_layer_size, DataMan.vocab_size], initializer=initializer)
        b = tf.get_variable("out_b", [DataMan.vocab_size], initializer=initializer)
        z = tf.matmul(output, w) + b # Add supports broadcasting over each row

        # Average negative log probability
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(z, tf.reshape(self._target, [-1]))
        self._cost = cost = tf.reduce_sum(loss) / batch_size

        if not training:
            self._train_op = tf.no_op()
            return

        self._learning_rate = tf.Variable(learning_rate, trainable=False)
        # Gradient descent training op
        optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
        self._train_op = optimizer.minimize(self._cost)

    def set_learning_rate(self, sess, value):
        sess.run(tf.assign(self._learning_rate, value))

    def lr_decay_and_set(self, sess, epoch):
        if epoch > decay_start:
            decay = learning_decay ** (epoch - decay_start)
            self.set_learning_rate(sess, learning_rate * decay)

def run_epoch(sess, data_set, net):
    total_cost = 0
    i = 0
    for i, (x, y, z) in enumerate(data_set.batch_iterator(batch_size)):
        # Input
        feed = { net._input : x, net._target : y, net._seq_lens : z}
        # Run the computational graph
        cost, _ = sess.run([net._cost, net._train_op], feed_dict=feed)
        total_cost += cost

    return total_cost / (i+1)

def save_state(sess, saver):
    print("Saving model.")
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in file: {}".format(save_path))

def create_plots(xs, t_ys, v_ys):
    plt.plot(xs, t_ys)
    plt.plot(xs, v_ys)
    plt.ylabel('cost')
    plt.xlabel('epochs')
    plt.title('Cost of training and evaluation')
    plt.grid(True)
    plt.savefig("foo")

def main():
    start_time = time.time()

    training_set = DataMan("train.txt")
    with tf.variable_scope("model", reuse=None):
        training_net = LSTM_Network(training_set, True)

    validation_set = DataMan("valid.txt", rebuild_vocab=False)
    with tf.variable_scope("model", reuse=True):
        validation_net = LSTM_Network(validation_set, False)

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
        for i in range(max_epoch):
            print("\r{}% done".format(int(i/max_epoch * 100)))
            training_net.lr_decay_and_set(sess, i)
            cost = run_epoch(sess, training_set, training_net)
            cost_train.append(cost)
            cost = run_epoch(sess, validation_set, validation_net)
            cost_valid.append(cost)
        print("100% done")

        print("Creating plot.")
        create_plots(range(max_epoch), cost_train, cost_valid)

        save_state(sess, saver)
        print("--- {} seconds ---".format(round(time.time() - start_time, 2)))

if __name__ == "__main__":
    main()

