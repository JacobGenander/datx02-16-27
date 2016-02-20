#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import time
import reader
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
max_word_seq = 40 # Size of the longest headline that can occur
word_vec_size = 25 # Each word is represented by a vector of this size
batch_size = 10
hidden_layer_size = 20 # Number of neurons in each LSTM layer
number_of_layers = 2
learning_rate = 0.01
max_epoch = 5

vocab_size = 80645 # Amount of words in our dictionary

class LSTM_Network(object):

    def __init__(self):
        # 3-dimensional tensors for input data and targets 
        self._input = tf.placeholder(tf.int32, [batch_size, max_word_seq], name="input_data")
        self._target = tf.placeholder(tf.int32, [batch_size, max_word_seq], name="target_data")

        embedding = tf.get_variable("embedding", [vocab_size, hidden_layer_size ])
        ins = tf.nn.embedding_lookup(embedding, self._input)

        # Create the network 
        cell = rnn_cell.BasicLSTMCell(hidden_layer_size, word_vec_size) 
        stacked_cells = rnn_cell.MultiRNNCell([cell] * number_of_layers)
    
        with tf.name_scope("initial_state"):
            self._initial_state = stacked_cells.zero_state(batch_size, tf.float32)
            state = self._initial_state
        
        with tf.name_scope("input"):
            # Give our input the right shape
            inputs = [ tf.squeeze(input_, [1]) for input_ in tf.split(1, max_word_seq, ins)]

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
            y_ = tf.cast(tf.reshape(self._target, [-1]), dtype=tf.float32)
            y = tf.reduce_max(y,1)
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

def run_epoch(sess, data, net, info_op, writer):

    current_state = net._initial_state.eval()

    for i, (x, y) in enumerate(reader.batch_iterator(data, batch_size, max_word_seq)): 
        feed = { net._input : x, net._target : y, net._initial_state : current_state}
        current_state, _  = sess.run([net._final_state, net._train_op], feed_dict=feed)

        if i % 10 == 0:
            info = sess.run(info_op, feed_dict=feed)
            summary_str = info 
            writer.add_summary(summary_str, i) 

def main():
    start_time = time.time()
    
    net = LSTM_Network()
    _, data = reader.prepare_data("./titles2.txt", max_word_seq)

    # We always need to run this operation before anything else
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        # Some initial stuff for TensorBoard
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("/tmp/tensorFlow_logs", sess.graph_def)

        # Initialize everything else
        sess.run(init)
        
        for i in range(max_epoch):
            print("{0} % done".format(i / max_epoch * 100))
            run_epoch(sess, data, net, merged, writer)
            

        print("100 % done!")
        print("--- {:.5} seconds ---".format(time.time() - start_time))
   
if __name__ == "__main__":
    main()


