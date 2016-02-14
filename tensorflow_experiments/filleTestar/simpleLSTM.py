#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell
import logging
from gensim.models import Word2Vec

# Enable debug information to be written to the console
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load google's pretrained vectors
#model = Word2Vec.load_word2vec_format('/home/filip/Hämtningar/GoogleNews-vectors-negative300.bin.gz', binary=True)

# TensorFlow's API (if you want to know what arguments we pass to the different methods)
# https://www.tensorflow.org/versions/0.6.0/api_docs/python/index.html

# Hyper parameters
max_word_seq = 50 # Size of the longest headline that can occur
word_vec_size = 100 # Each word is represented by a vector of this size
batch_size = 100
hidden_layer_size = 100 # Number of neurons in each LSTM layer
number_of_layers = 3
learning_rate = 0.01

data_size = 1000
vocab_size = 10000 # Amount of words in our dictionary

class LSTM_Network(object):

    def __init__(self):
        # This 3-dimensional tensor will hold all data for a specific batch
        self._input = tf.placeholder(tf.float32, [batch_size, max_word_seq, word_vec_size])
        self._target = tf.placeholder(tf.float32, [batch_size, max_word_seq, vocab_size])

        # Create the network 
        cell = rnn_cell.LSTMCell(hidden_layer_size, word_vec_size) # Possibility to set forget_bias

        # Define the initial state 
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        # Run through the batch and update state 
        state = self._initial_state
        self._loss = tf.Variable(tf.zeros([1]))
        with tf.variable_scope("RNN"):
            for i in range(max_word_seq):
                if i > 0: tf.get_variable_scope().reuse_variables()
                output, state = cell(self._input[:, i, :], state)
                y_target = self._target[:, i, :]

                w = tf.get_variable("out_w", [hidden_layer_size, vocab_size])
                b = tf.get_variable("out_b", [vocab_size])
                z = tf.matmul(output, w) + b # Hur dom olika batcherna slogs ihop har jag ingen aning om...
                
                self._loss += tf.nn.softmax_cross_entropy_with_logits(z, y_target)

        self._final_state = state # This is the operator we will use when training
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self._train_op = optimizer.minimize(self._loss)

def dummyInput():
    return np.random.rand(batch_size, max_word_seq, word_vec_size)

def dummyTarget():
    return np.random.rand(batch_size, max_word_seq, vocab_size)

def main():
    # Create the network
    net = LSTM_Network()

    # We always need to run this operation before anything else
    init = tf.initialize_all_variables()

    # Create a session and run the initialization
    with tf.Session() as sess:
        sess.run(init)
        current_state = net._initial_state.eval()

        # This is where we iterate through all the data 
        for i in range(0, data_size, batch_size):
            # Our input  will just be a batch with random values
            feed = { net._input : dummyInput(), net._target : dummyTarget(), net._initial_state : current_state} # Need to ask if this really overwrites the initial state in the constructor
            current_state, test = sess.run([net._final_state, net._train_op], feed_dict=feed) 

        print(current_state)
   
if __name__ == "__main__":
    main()

