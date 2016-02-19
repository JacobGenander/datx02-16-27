#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import time
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
        # 3-dimensional tensors for input data and targets 
        self._input = tf.placeholder(tf.float32, [batch_size, max_word_seq, word_vec_size], name="input_data")
        self._target = tf.placeholder(tf.float32, [batch_size, max_word_seq, vocab_size], name= "target_data")

        # Create the network 
        cell = rnn_cell.BasicLSTMCell(hidden_layer_size, word_vec_size) 
        stacked_cells = rnn_cell.MultiRNNCell([cell] * number_of_layers)
    
        with tf.name_scope("initial_state"):
            self._initial_state = stacked_cells.zero_state(batch_size, tf.float32)
            state = self._initial_state
        
        with tf.name_scope("input"):
            # Give our input the right shape
            inputs = [ tf.squeeze(input_, [1]) for input_ in tf.split(1, max_word_seq, self._input)]

        # Run through the whole batch and update state
        outputs, state = rnn.rnn(stacked_cells, inputs, initial_state=self._initial_state)
        
        with tf.name_scope("output"):
            # The output also needs some massaging
            output = tf.reshape(tf.concat(1, outputs), [-1, hidden_layer_size])

        y = None
        with tf.name_scope("softmax"):
            w = tf.get_variable("out_w", [hidden_layer_size, vocab_size])
            b = tf.get_variable("out_b", [vocab_size])
            z = tf.matmul(output, w) + b # Add supports broadcasting over each row 
            y = tf.nn.softmax(z)

            # This is just to enable information for TensorBoard 
            w_hist = tf.histogram_summary("weights", w)
            b_hist = tf.histogram_summary("biases", b)
                
        # TensorFlow's built in costfunctions can take whole batches as input 
        with tf.name_scope("cost"):
            y_ = tf.reshape(self._target, [-1, vocab_size])
            with tf.name_scope("cross_entropy"):
                xent = -tf.reduce_sum(y_*tf.log(y))

            with tf.name_scope("batch_mean"):
                self._cost = cost = tf.reduce_sum(xent) / tf.constant(batch_size, name="batch_size", dtype=tf.float32)
                ce_summ = tf.scalar_summary("cost", cost) 

        self._final_state = state[-1] # We only want the latest state 
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self._train_op = optimizer.minimize(self._cost) 

def dummyInput():
    return np.random.rand(batch_size, max_word_seq, word_vec_size)

def dummyTarget():
    return np.random.rand(batch_size, max_word_seq, vocab_size)

def main():
    start_time = time.time()
    
    net = LSTM_Network()

    # We always need to run this operation before anything else
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        # Some initial stuff for TensorBoard
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("/tmp/tensorFlow_logs", sess.graph_def)

        # Initialize everything else
        sess.run(init)
        current_state = net._initial_state.eval()

        # This is where we iterate through all the data 
        for i in range(100):
            # Our input  will just be a batch with random values
            feed = { net._input : dummyInput(), net._target : dummyTarget(), net._initial_state : current_state} # _initial_state is no placeholder, but we can still give it as an argument (???)
            current_state, _  = sess.run([net._final_state, net._train_op], feed_dict=feed) 
            
            if i % 10 == 0:
                print("{0} % done".format(i))
                res = sess.run(merged, feed_dict=feed)
                summary_str = res
                writer.add_summary(summary_str, i)

        # To get some kind of output we print the last state in the list 
        print(current_state[-1])
        print("--- {:.5} seconds ---".format(time.time() - start_time))
   
if __name__ == "__main__":
    main()

