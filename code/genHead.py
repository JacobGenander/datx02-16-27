#! /usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from DataMan import DataMan

# Hardcoded for the moment
batch_size = 20
hidden_layer_size = embedding_size = 200 
number_of_layers = 2
learning_rate = 1.0 
init_range = 0.1
max_epoch = 13

class LSTM_Network(object):

    def __init__(self, vocab_size):
        self._inputs = tf.placeholder(tf.int32, [batch_size, 1])

        cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_size)
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * number_of_layers)
        self._initial_state = state = stacked_cell.zero_state(batch_size, tf.float32)
 
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, embedding_size])
            inputs = tf.nn.embedding_lookup(embedding, self._inputs)

        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, 1, inputs)] # Probably unnecessary split
        outputs, state = tf.nn.rnn(stacked_cell, inputs, initial_state=self._initial_state)

        output = tf.reshape(tf.concat(1, outputs), [-1, hidden_layer_size])
        w = tf.get_variable("out_w", [hidden_layer_size, vocab_size])
        b = tf.get_variable("out_b", [vocab_size])
        z = tf.matmul(output, w) + b
        softmax = tf.nn.softmax(z)
        
        words = tf.argmax(softmax, 1)
        
        self._next_words = tf.reshape(words, [batch_size, 1]) 
        self._final_state = state

def generate_input(vocab_size):
    return np.random.randint(0, vocab_size, [batch_size, 1])

def gen_sentences(net, sess, vocab_size, max_word_seq):
    inputs = generate_input(vocab_size)
    current_state = net._initial_state.eval()
    
    sentences = [inputs]
    for i in range(max_word_seq):
        feed = {net._inputs : inputs, net._initial_state : current_state}
        output, current_state = sess.run([net._next_words, net._final_state], feed_dict=feed)
        sentences.append(output)
        inputs = output 
        
    return np.concatenate(sentences, 1) # Check if done on right dimension

def format_sentence(s):
    return s.split("<eos>", 1)[0].capitalize()

def main():
    reader = DataMan("titles.txt")
    net = LSTM_Network(reader.vocab_size)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        saver = tf.train.Saver() # Is this correct? Will it overwrite eariler inits?
        saver.restore(sess, "/tmp/model.ckpt") # Should cell state be restored from training?

        sentences = gen_sentences(net, sess, reader.vocab_size, reader.max_seq)
        
        for i, s in enumerate(sentences):
            if i >= 20: # Decides how many titles we should display
                break
            print("Sentence {}:".format(i+1))
            s = [ reader.id_to_word[w] for w in s]
            s = " ".join(s)
            print(format_sentence(s))

if __name__ == "__main__":
    main()

