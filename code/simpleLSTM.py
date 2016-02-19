#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell
#import logging
#from gensim.models import Word2Vec

# Enable debug information to be written to the console
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load google's pretrained vectors
#model = Word2Vec.load_word2vec_format('/home/filip/Hämtningar/GoogleNews-vectors-negative300.bin.gz', binary=True)

# TensorFlow's API (if you want to know what arguments we pass to the different methods)
# https://www.tensorflow.org/versions/0.6.0/api_docs/python/index.html

# Hyper parameters (adjust these if you want the code to run faster)
max_word_seq = 50 # Size of the longest headline that can occur
word_vec_size = 25 # Each word is represented by a vector of this size
batch_size = 10
hidden_layer_size = 20 # Number of neurons in each LSTM layer
number_of_layers = 2
learning_rate = 0.01

data_size = 100
vocab_size = 350 # Amount of words in our dictionary

class LSTM_Network(object):

    def __init__(self):
        # This 3-dimensional tensor will hold all data for a specific batch
        self._input = tf.placeholder(tf.float32, [batch_size, max_word_seq, word_vec_size])
        # And this one will hold our target data
        self._target = tf.placeholder(tf.float32, [batch_size, max_word_seq, vocab_size])

        # Create a single lstm layer
        cell = rnn_cell.BasicLSTMCell(hidden_layer_size, word_vec_size) # Possibility to set forget_bias
        # Stack some layers together 
        stacked_cells = rnn_cell.MultiRNNCell([cell] * number_of_layers)

        # Define the initial state 
        self._initial_state = stacked_cells.zero_state(batch_size, tf.float32)

        # Run through the batch and update state 
        state = self._initial_state
        self._loss = tf.Variable(tf.zeros([1]))
        with tf.variable_scope("RNN"): # I'm not sure about variable scopes, but the code won't run unless I add this...
            for i in range(max_word_seq):
                if i > 0: tf.get_variable_scope().reuse_variables() # ... and this
                # Pick out the specific batch
                output, state = stacked_cells(self._input[:, i, :], state)
                y_target = self._target[:, i, :]

                # Theses calculations shoud be familiar to you
                w = tf.get_variable("out_w", [hidden_layer_size, vocab_size])
                b = tf.get_variable("out_b", [vocab_size])
                z = tf.add(tf.matmul(output, w), b) # OBS! Add supports broadcasting over each column
                
                # TensorFlow's built in costfunctions can take whole batches as input 
                self._loss += tf.nn.softmax_cross_entropy_with_logits(z, y_target)

        self._final_state = state # This is one of the operators we will run when training
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self._train_op = optimizer.minimize(self._loss) # This is the other training operator (for the weights and biases)

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
            feed = { net._input : dummyInput(), net._target : dummyTarget(), net._initial_state : current_state} # _initial_state is no placeholder, but we can still give it as an argument (???)
            current_state, _  = sess.run([net._final_state, net._train_op], feed_dict=feed) 

        # To get some kind of output we print the last state in the list 
        print(current_state[-1])
   
if __name__ == "__main__":
    main()

