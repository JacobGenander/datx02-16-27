#! /usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cPickle as pickle
import hyperParams
import DataManager 
import plot
import time
import sys

# TensorFlow's API (if you want to know what arguments we pass to the different methods)
# https://www.tensorflow.org/versions/r0.7/api_docs/python/index.html

class LSTM_Network(object):
    def __init__(self, training, config):
        self.batch_size = batch_size = config["batch_size"] 
        self.num_steps = num_steps = config["num_steps"]
        self.size = size = config["hidden_layer_size"]
        keep_prob = config["keep_prob"]
        vocab_size = config["vocab_size"]

        # 2-dimensional tensors for input data and targets
        self._input = tf.placeholder(tf.float32, [batch_size, num_steps])
        self._target = tf.placeholder(tf.int64, [batch_size, num_steps])

        inputs = self._input 
        if keep_prob < 1 and training:
            inputs = tf.nn.dropout(inputs, keep_prob)

        # Create the network
        cell = tf.nn.rnn_cell.BasicLSTMCell(size)
        if keep_prob < 1 and training:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        stacked_cells = tf.nn.rnn_cell.MultiRNNCell([cell] * config["number_of_layers"])

        self.initial_state = stacked_cells.zero_state(batch_size, tf.float32)

        # Give input the right shape
        inputs = tf.split(1, self.num_steps, inputs)

        # Run through the whole batch and update state
        outputs, state = tf.nn.rnn(stacked_cells, inputs, initial_state=self.initial_state)
        self.final_state = state

        # Turn output into tensor instead of list
        outputs = tf.concat(1, outputs)

        # Flatten output into a tensor where each row is the output from one word
        output = tf.reshape(outputs, [-1, size])

        w = tf.get_variable("out_w", [size, vocab_size])
        b = tf.get_variable("out_b", [vocab_size])
        z = tf.matmul(output, w) + b # Add supports broadcasting over each row

        # Average negative log probability
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(z, tf.reshape(self._target, [-1]))
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        
        if not training:
            self._train_op = tf.no_op()
            return

        self._learning_rate = tf.Variable(config["learning_rate"], trainable=False)

        # Clip the gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config["gradient_clip"])
        # RMSProp training op
        optimizer = tf.train.RMSPropOptimizer(self._learning_rate)
        self._train_op = optimizer.apply_gradients(zip(grads,tvars))

    def set_learning_rate(self, sess, value):
        sess.run(tf.assign(self._learning_rate, value))

def run_epoch(sess, net, data_man, data_set):
    total_cost = 0.0
    state = sess.run(net.initial_state)
    for i, (x, y) in enumerate(data_man.batch_iterator(net.batch_size, net.num_steps, data_set)):
        # Input
        feed = { net._input : x, net._target : y , net.initial_state : state}
        # Calculate cost and train the network 
        cost, state, _ = sess.run([net._cost, net.final_state, net._train_op], feed_dict=feed)
        total_cost += cost
    return total_cost / (i+1)

def save_state(sess, saver, config):
    print("Saving model.")
    saver.save(sess, "model.ckpt")
    pickle.dump(config, open("config.p", "wb"))

def main():
    start_time = time.time()

    config = hyperParams.config
    data_man = DataManager.DataMan("data.txt", 0.1)
    # Update config with information from DataMan
    data_params = { "word_to_id" : data_man.char_to_id, "id_to_word" : data_man.id_to_char,
            "vocab_size" : data_man.vocab_size}
    config.update(data_params)

    # Create networks for training and evaluation
    initializer = tf.random_uniform_initializer(-config["init_range"], config["init_range"])
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        train_net = LSTM_Network(True, config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        val_net = LSTM_Network(False, config)

    # We always need to run this operation if not loading an old state
    init = tf.initialize_all_variables()
    # This operation will save our state at the end
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)

        max_epoch = config["max_epoch"]
        cost_train, cost_valid = [], []
        print("Training.")
        for i in range(0, max_epoch):
            print("\r{}% done".format(int(i/max_epoch * 100)), end="")
            sys.stdout.flush()

            if i > config["decay_start"]:
                decay = config["learning_decay"] ** (i - config["decay_start"])
                train_net.set_learning_rate(sess, config["learning_rate"] * decay)

            cost_t = run_epoch(sess, train_net, data_man, DataManager.TRAIN_SET)
            cost_train.append(cost_t)
            cost_v = run_epoch(sess, val_net, data_man, DataManager.VALID_SET)
            cost_valid.append(cost_v)

            plot.create_plots(range(i+1), cost_train, cost_valid)
        print("\r100% done")

        save_state(sess, saver, config)
        sess.close()
        print("--- {} seconds ---".format(round(time.time() - start_time, 2)))

if __name__ == "__main__":
    main()

