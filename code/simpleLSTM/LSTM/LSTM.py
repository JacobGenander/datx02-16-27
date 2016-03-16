#! /usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import interfaceLSTM
import hyperParams
import cPickle as pickle
import zipfile
import plot
import time
import sys
import os
from DataMan import DataMan

# TensorFlow's API (if you want to know what arguments we pass to the different methods)
# https://www.tensorflow.org/versions/r0.7/api_docs/python/index.html

class LSTM_Network(object):
    def __init__(self, training, batch_size, config, emb_init):
        self.batch_size = batch_size
        self.size = size = config["hidden_layer_size"]
        max_seq = config["max_seq"]
        keep_prob = config["keep_prob"]
        vocab_size = config["vocab_size"]

        # 2-dimensional tensors for input data and targets
        self._input = tf.placeholder(tf.int32, [batch_size, max_seq])
        self._target = tf.placeholder(tf.int64, [batch_size, max_seq])
        # This is the length of each sentence
        self._seq_lens = tf.placeholder(tf.int32, [batch_size])
        # We need these to crop output and targets so we do not train on more time sequences than needed
        self._out_dim = tf.placeholder(tf.int32, [2])
        self._target_dim = tf.placeholder(tf.int32, [2])

        # Fetch word vectors
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", emb_init[1], initializer=emb_init[0])
            inputs = tf.nn.embedding_lookup(embedding, self._input)

        if keep_prob < 1 and training:
            inputs = tf.nn.dropout(inputs, keep_prob)

        # Create the network
        cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=config["forget_bias"])
        if keep_prob < 1 and training:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        stacked_cells = tf.nn.rnn_cell.MultiRNNCell([cell] * config["number_of_layers"])

        self._initial_state = state = stacked_cells.zero_state(batch_size, tf.float32)

        # Give input the right shape
        inputs = [ tf.squeeze(input_, [1]) for input_ in tf.split(1, max_seq, inputs)]

        # Run through the whole batch and update state
        outputs, _ = tf.nn.rnn(stacked_cells, inputs, initial_state=self._initial_state, sequence_length=self._seq_lens)
        # Turn output into tensor instead of list
        outputs = tf.concat(1, outputs)

        # Crop output and targets to the length of the longest sentence
        outputs = tf.slice(outputs, [0, 0], self._out_dim)
        target = tf.slice(self._target, [0, 0], self._target_dim)

        # Flatten output into a tensor where each row is the output from one word
        output = tf.reshape(outputs, [-1, size])

        w = tf.get_variable("out_w", [size, vocab_size])
        b = tf.get_variable("out_b", [vocab_size])
        z = tf.matmul(output, w) + b # Add supports broadcasting over each row

        # Average negative log probability
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(z, tf.reshape(target, [-1]))
        self._cost = cost = tf.reduce_sum(loss) / batch_size

        if not training:
            self._train_op = tf.no_op()
            return

        self._learning_rate = tf.Variable(config["learning_rate"], trainable=False)
        # Gradient descent training op
        optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
        self._train_op = optimizer.minimize(self._cost)

    def set_learning_rate(self, sess, value):
        sess.run(tf.assign(self._learning_rate, value))

    def calc_output_dims(self, seq_lens):
        max_batch_seq = max(seq_lens)
        dim_output = np.array([self.batch_size, max_batch_seq * self.size])
        dim_target = np.array([self.batch_size, max_batch_seq])
        return dim_output, dim_target

def run_epoch(sess, data_set, net):
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

    perplexity = np.exp(total_cost / steps)
    average_cost = total_cost / (i+1)
    return average_cost, perplexity

def save_state(sess, name, saver, config, save_path):
    print("\nSaving model.")
    model_path = os.path.join(save_path, "model.ckpt")
    config_path = os.path.join(save_path, "config.p")
    zip_path = os.path.join(save_path, name)

    saver.save(sess, model_path)
    pickle.dump(config, open(config_path, "wb"))

    zf = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)
    zf.write(model_path, os.path.relpath("model.ckpt", save_path))
    zf.write(config_path, os.path.relpath("config.p", save_path))
    zf.close()

def extract_state(root_path, restore_path):
    zf = zipfile.ZipFile(restore_path, "r")
    extract_path = os.path.join(root_path, "tmp")
    zf.extractall(extract_path)
    zf.close()

def create_data_sets(data_path, max_seq, max_vocab_size):
    try:
        train_path = os.path.join(data_path, "train.txt")
        training_set = DataMan(train_path, max_seq, max_vocab_size=max_vocab_size)
        valid_path = os.path.join(data_path, "valid.txt")
        validation_set = DataMan(valid_path, max_seq, rebuild_vocab=False)
        test_path = os.path.join(data_path, "test.txt")
        test_set = DataMan(test_path, max_seq, rebuild_vocab=False)
    except IOError:
        print("File not found. Data path needs to contain three files: train.txt, valid.txt and test.txt")
        sys.exit(1)

    return training_set, validation_set, test_set

def main():
    start_time = time.time()
    root_path = os.path.dirname(os.path.realpath(__file__))
    args = interfaceLSTM.parser.parse_args()

    save_path = args.save_path
    if not os.path.isdir(save_path):
        print("Couldn't find save directory")
        sys.exit(1)

    # Fetch parameters
    if args.load_state:
        extract_state(root_path, args.load_state)
        file_path = os.path.join(root_path, "tmp", "config.p")
        with open(file_path, "rb") as f:
            config = pickle.load(f)
    else:
        config = hyperParams.config
        config["start_epoch"] = 0
        config["cost_train"], config["cost_valid"] = [], []

    # Load data
    training_set, validation_set, test_set = create_data_sets(
            args.data_path, config["max_seq"], config["max_vocab_size"])

        # Update config with information from DataMan
    data_params = { "word_to_id" : DataMan.word_to_id, "id_to_word" : DataMan.id_to_word,
            "vocab_size" : DataMan.vocab_size, "unk_id" : DataMan.unk_id}
    config.update(data_params)

    initializer = tf.random_uniform_initializer(-config["init_range"], config["init_range"])
    # Choose initialization of embedding matrix
    if args.embedding:
        config["embedding_size"] = 300 # 300 is the dimensionality of word2vec
        with  open(args.embedding, "rb") as f:
            emb_matrix = pickle.load(f)
        emb_initzer = tf.constant(emb_matrix, dtype=tf.float32)
        init_shape = None
    else:
        emb_initzer = initializer
        init_shape = [config["vocab_size"], config["embedding_size"]]
    emb_init = (emb_initzer, init_shape)

    # Create networks for training and evaluation
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        train_net = LSTM_Network(True, config["batch_size"], config, emb_init)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        val_net = LSTM_Network(False, config["batch_size"], config, emb_init)
        test_net = LSTM_Network(False, 1, config, emb_init)

    # We always need to run this operation if not loading an old state
    init = tf.initialize_all_variables()

    # This operation will save our state
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize variables
        if args.load_state:
            file_path = os.path.join(root_path, "tmp", "model.ckpt")
            saver.restore(sess, file_path)
        else:
            sess.run(init)

        # Some further initialization before the training loop
        max_epoch = config["max_epoch"]
        start_epoch = config["start_epoch"]

        print("Training.")
        for i in range(start_epoch, max_epoch, 1):
            print("\r{}% done".format(int(i/max_epoch * 100)), end="")
            sys.stdout.flush()

            if i > config["decay_start"]:
                decay = config["learning_decay"] ** (i - config["decay_start"])
                train_net.set_learning_rate(sess, config["learning_rate"] * decay)

            cost_t, _ = run_epoch(sess, training_set, train_net)
            config["cost_train"].append(cost_t)
            cost_v, _ = run_epoch(sess, validation_set, val_net)
            config["cost_valid"].append(cost_v)

            if (i != 0 and i % config["save_epoch"] == 0) or (i == max_epoch - 1):
                config["start_epoch"] = i+1
                name = "epoch{0}cost{1}.zip".format(i, round(cost_t, 4))
                save_state(sess, name, saver, config, save_path)
        print("\r100% done")

        print("Calculating perplexity.")
        _, perplexity = run_epoch(sess, test_set, test_net)
        print("Perplexity: {}".format(perplexity))

        print("Creating plot.")
        plot.create_plots(save_path, range(max_epoch), config["cost_train"], config["cost_valid"])

        sess.close()
        print("--- {} seconds ---".format(round(time.time() - start_time, 2)))

if __name__ == "__main__":
    main()

