#! /usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import interfaceGenHead
import cPickle as pickle
import random
import os

class LSTM_Network(object):

    def __init__(self, vocab_size, config):
        batch_size = config["batch_size"]
        size = config["hidden_layer_size"]
        self._inputs = tf.placeholder(tf.int32, [batch_size, 1])

        cell = tf.nn.rnn_cell.BasicLSTMCell(size)
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * config["number_of_layers"])
        self._initial_state = state = stacked_cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, config["embedding_size"]])
            inputs = tf.nn.embedding_lookup(embedding, self._inputs)
        inputs = [tf.squeeze(inputs, [1])]

        outputs, state = tf.nn.rnn(stacked_cell, inputs, initial_state=self._initial_state)

        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        w = tf.get_variable("out_w", [size, vocab_size])
        b = tf.get_variable("out_b", [vocab_size])
        z = tf.matmul(output, w) + b
        softmax = tf.nn.softmax(z)

        self._word_predictions = softmax
        self._final_state = state

def generate_input(vocab_size, batch_size):
    return np.random.randint(0, vocab_size, [batch_size, 1])

def choose_words(word_probs, batch_size):
    res = np.zeros([batch_size, 1], dtype=np.int32)
    most_prob = interfaceGenHead.parser.parse_args().most_prob
    if most_prob:
        for i, probs in enumerate(word_probs):
            # Choose the most probable words
            res[i,0] = np.argmax(probs, axis=0)
    # This case weights its prediction based on all probabilities
    else:
        rand = random.uniform(0,1)
        for i, probs in enumerate(word_probs):
            s = 0
            index_picked = False
            for j in range(len(probs)):
                s += probs[j] # The added probabilty that makes us exceed the rand value will be the word we choose
                if s >= rand:
                    index_picked = True
                    res[i,0] = j
                    break
            if not index_picked:
                res[i,0] = j
    return res

def gen_sentences(net, sess, vocab_size, config):
    current_state = net._initial_state.eval()
    inputs = generate_input(vocab_size, config["batch_size"])
    sentences = [inputs]
    for i in range(config["max_seq"]):
        feed = {net._inputs : inputs, net._initial_state : current_state}
        output, current_state = sess.run([net._word_predictions, net._final_state], feed_dict=feed)

        next_words = choose_words(output, config["batch_size"])

        sentences.append(next_words)
        inputs = next_words

    return np.concatenate(sentences, 1)

def format_sentence(s):
    return s.split("<eos>", 1)[0].capitalize()

def main():
    args = interfaceGenHead.parser.parse_args()
    model_folder = args.model_folder

    config_path = os.path.join(model_folder, "params.p")
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    dict_path = os.path.join(model_folder, "word_dict.p")
    with open(dict_path, "rb") as f:
        id_to_word = pickle.load(f)

    init = tf.initialize_all_variables()

    vocab_size = len(id_to_word)
    with tf.variable_scope("model", reuse=False):
        net = LSTM_Network(vocab_size, config)

    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()

        model_path = os.path.join(model_folder, "model.ckpt")
        saver.restore(sess, model_path)

        sentences = gen_sentences(net, sess, vocab_size, config)

        for i, s in enumerate(sentences):
            if i >= args.n: # Decides the number of displayed headlines
                break
            s = [ id_to_word[w] for w in s]
            s = " ".join(s)
            print("Sentence {0}:\t{1}".format(i+1, format_sentence(s)))

if __name__ == "__main__":
    main()
