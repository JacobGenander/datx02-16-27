#! /usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import interfaceGenHead
import random
from DataMan import DataMan
from hyperParams import *

class LSTM_Network(object):

    def __init__(self, vocab_size):
        self._inputs = tf.placeholder(tf.int32, [BATCH_SIZE, 1])

        cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_LAYER_SIZE)
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * NUMBER_OF_LAYERS)
        self._initial_state = state = stacked_cell.zero_state(BATCH_SIZE, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, EMBEDDING_SIZE])
            inputs = tf.nn.embedding_lookup(embedding, self._inputs)

        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, 1, inputs)] # Probably unnecessary split
        outputs, state = tf.nn.rnn(stacked_cell, inputs, initial_state=self._initial_state)

        output = tf.reshape(tf.concat(1, outputs), [-1, HIDDEN_LAYER_SIZE])
        w = tf.get_variable("out_w", [HIDDEN_LAYER_SIZE, vocab_size])
        b = tf.get_variable("out_b", [vocab_size])
        z = tf.matmul(output, w) + b
        softmax = tf.nn.softmax(z)

        self._word_predictions = softmax
        self._final_state = state

def generate_input(vocab_size):
    return np.random.randint(0, vocab_size, [BATCH_SIZE, 1])

def choose_words(word_probs, most_prob):
    res = np.zeros([BATCH_SIZE, 1], dtype=np.int32)
    if most_prob:
        for i, probs in enumerate(word_probs):
            res[i,0] = np.argmax(probs, axis=0)
    else:
        rand = random.uniform(0,1)
        for i, probs in enumerate(word_probs):
            s = 0
            index_picked = False
            for j in range(len(probs)):
                s += probs[j]
                if s >= rand:
                    index_picked = True
                    res[i,0] = j
                    break
            if not index_picked:
                res[i,0] = j
    return res

def gen_sentences(net, sess, vocab_size, max_word_seq):
    inputs = generate_input(vocab_size)
    current_state = net._initial_state.eval()

    sentences = [inputs]
    for i in range(max_word_seq):
        feed = {net._inputs : inputs, net._initial_state : current_state}
        output, current_state = sess.run([net._word_predictions, net._final_state], feed_dict=feed)
        next_words = choose_words(output, False)
        sentences.append(next_words)
        inputs = next_words

    return np.concatenate(sentences, 1)

def format_sentence(s):
    return s.split("<eos>", 1)[0].capitalize()

def main():
    args = interfaceGenHead.parser.parse_args()
    model_path = args.model_path

    data_set = DataMan("train.txt", MAX_SEQ)
    with tf.variable_scope("model", reuse=False):
        net = LSTM_Network(data_set.vocab_size)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        sentences = gen_sentences(net, sess, DataMan.vocab_size, MAX_SEQ)

        for i, s in enumerate(sentences):
            if i >= 20: # Decides how many titles we should display
                break
            print("Sentence {}:".format(i+1))
            s = [ data_set.id_to_word[w] for w in s]
            s = " ".join(s)
            print(format_sentence(s))

if __name__ == "__main__":
    main()

