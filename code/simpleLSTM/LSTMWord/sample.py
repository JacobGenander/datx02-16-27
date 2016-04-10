#! /usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cPickle as pickle
import DataManager
import argparse
import random
import sys
import os

parser = argparse.ArgumentParser(description=
        'Generates sentences from a pretrained LSTM-model')
parser.add_argument('--model_dir', metavar='PATH', type=str, default='results',
        help='directory where the model is saved')
parser.add_argument('-n', metavar='N', type=int, default=1,
        help='number of sequences to generate (might be capped by batch size)')
parser.add_argument('--length', metavar='L', type=int, default=200,
        help='length of each sequence')
parser.add_argument('--init_seq', type=str, default='',
        help='condition model on some initial sequence')
parser.add_argument('--skip_unk', action='store_true', default=False,
        help='skip _UNK when generating sequences')

class LSTM_Network(object):

    def __init__(self, size, layers, vocab_size, batch_size, embedding_size):
        self._inputs = tf.placeholder(tf.int32, [batch_size, 1])

        cell = tf.nn.rnn_cell.BasicLSTMCell(size)
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layers)
        self._initial_state = state = stacked_cell.zero_state(batch_size, tf.float32)

        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [vocab_size, embedding_size])
            inputs = tf.nn.embedding_lookup(embedding, self._inputs)
        inputs = [tf.squeeze(inputs, [1])]

        outputs, state = tf.nn.rnn(stacked_cell, inputs, initial_state=self._initial_state)

        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        w = tf.get_variable('out_w', [size, vocab_size])
        b = tf.get_variable('out_b', [vocab_size])
        z = tf.matmul(output, w) + b

        self._word_predictions = tf.nn.softmax(z)
        self._final_state = state

def choose_words(word_probs, batch_size, vocab_size, skip_unk):
    res = np.zeros([batch_size, 1], dtype=np.int32)
    for i, probs in enumerate(word_probs):
        if skip_unk:
            inc = probs[DataManager.UNK_ID] / (len(probs) - 1)
            probs = [ p+inc for p in probs]
            probs[DataManager.UNK_ID] = 0.0
            probs /= sum(probs) # Renormalize
        j = np.random.choice(range(vocab_size), p=probs)
        res[i,0] = j
    return res

def gen_init_batch(seq, batch_size, word_to_id):
    try:
        ids = [[word_to_id[w] for w in list(seq)]]
    except KeyError:
        print('Could not find word in vocabulary')
        sys.exit(1)
    init_batch =  np.repeat(ids, batch_size, 0)
    return init_batch

def gen_sentences(net, sess, word_to_id, conf):
    batch_size = conf.batch_size
    args = parser.parse_args()

    init_seq = args.init_seq
    if init_seq:
        init_batch = gen_init_batch(init_seq, batch_size, word_to_id)
        inputs = np.reshape(init_batch[:, 0], [batch_size, 1])
    else:
        inputs = np.random.randint(0, len(word_to_id), [batch_size, 1])

    sentences = [inputs]
    num_init_words = len(list(init_seq))
    current_state = net._initial_state.eval()
    for i in range(args.length):
        feed = {net._inputs : inputs, net._initial_state : current_state}
        output, current_state = sess.run([net._word_predictions, net._final_state], feed_dict=feed)

        if i >= num_init_words - 1:
            next_words = choose_words(output, batch_size, conf.vocab_size, args.skip_unk)
        else:
            next_words = np.reshape(init_batch[:, i+1], [batch_size, 1])

        sentences.append(next_words)
        inputs = next_words

    return np.concatenate(sentences, 1)

def format_sentences(s):
    s = ' '.join(s)
    return s.replace('_EOS', '\n')

def main():
    args = parser.parse_args()

    config_path = os.path.join(args.model_dir, 'config.p')
    with open(config_path, 'rb') as f:
        conf = pickle.load(f)

    with tf.variable_scope('model', reuse=False):
        net = LSTM_Network( conf.layer_size,
                            conf.num_layers,
                            conf.vocab_size,
                            conf.batch_size,
                            conf.embedding_size)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        model_path = os.path.join(args.model_dir, 'model.ckpt')
        saver.restore(sess, model_path)

        sentences = gen_sentences(net, sess, conf.word_to_id, conf)
        for i, s in enumerate(sentences):
            if i >= args.n: # Decides the number of displayed headlines
                break
            s = [ conf.id_to_word[w] for w in s]
            if i != 0:
                print('-' * 50)
            print(' ' + format_sentences(s))

if __name__ == '__main__':
    main()

