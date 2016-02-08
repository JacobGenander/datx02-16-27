#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell

# TensorFlow's API (if you want to know what arguments we pass to the different methods)
# https://www.tensorflow.org/versions/0.6.0/api_docs/python/index.html

# Hyper parameters
max_word_seq = 50 # Size of the longest headline that can occur
word_vec_size = 100 # Each word is represented by a vector of this size
batch_size = 100
hidden_layer_size = 100 # Number of neurons in each LSTM layer
number_of_layers = 3

data_size = 1000

class LSTM_Network(object):
    
    def __init__(self):
    
        # This 3-dimensional tensor will hold all data for a specific batch
        self._data = tf.placeholder(tf.float32, [max_word_seq, batch_size, word_vec_size])

        # Split the data into words
        word_columns = tf.split(1, max_word_seq, self._data) 

        # After the split we get tensors with the shape: 1 x batch size x vector size
        # This means we have to reshape them
        inputs = [ tf.reshape(w, (batch_size, word_vec_size)) for w in word_columns]

        # Create the network 
        cell = rnn_cell.LSTMCell(hidden_layer_size, word_vec_size)
        stacked_lstm = rnn_cell.MultiRNNCell([cell] * number_of_layers)

        # Define the initial state 
        self._initial_state = stacked_lstm.zero_state(batch_size, tf.float32)
        # Run through the batch and update state 
        outputs, states = rnn.rnn(stacked_lstm, inputs, initial_state=self._initial_state) # This helper function saves us form having to unroll the LSTM
        
        ##########################
        # TODO: Implement traing # 
        ##########################

        self._final_state = states # This is the operator we will use when training 

    @property
    def input(self):
        return self._data

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

def main():

    net = LSTM_Network()

    # We always need to run this operation before anything else
    init = tf.initialize_all_variables()

    # Create a session and run the initialization
    with tf.Session() as sess:
        sess.run(init)
        current_state = net.initial_state.eval()

        # This is where we iterate through all the data 
        for i in range(0, data_size, batch_size):
            # Our input  will just be a batch with random values
            feed = { net.input : np.random.rand(max_word_seq, batch_size, word_vec_size), net.initial_state : current_state} # Need to ask if this really overwrites the initial state in the constructor
            current_state = sess.run(net.final_state, feed_dict=feed)[-1] # We only want the latest state (gets a list)

        print(current_state)
   
if __name__ == "__main__":
    main()

