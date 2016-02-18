#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell
import os
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

save_steps = 10 # the number of steps taken between each saved state

# Counter for the number of training steps taken, used for saving the state
global_step = tf.Variable(0, name='global_step', trainable=False)

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
                z = tf.add(tf.matmul(output, w), b) # OBS! Add supports broadcasting over each row 
                
                # TensorFlow's built in costfunctions can take whole batches as input 
                self._loss += tf.nn.softmax_cross_entropy_with_logits(z, y_target)
            
        self._final_state = state # This is one of the operators we will run when training
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # This is the other training operator (for the weights and biases)
        # global_step is incremented every time the variables are optimized
        self._train_op = optimizer.minimize(self._loss, global_step = global_step)

def dummyInput():
    return np.random.rand(batch_size, max_word_seq, word_vec_size)

def dummyTarget():
    return np.random.rand(batch_size, max_word_seq, vocab_size)

def main():
    # Create the network
    net = LSTM_Network()
    
    # Flag indicating whether the network is training or not
    is_training = False

    # We always need to run this operation before anything else
    init = tf.initialize_all_variables()
    
    # Create a saver object that can store the state of the trained network
    saver = tf.train.Saver()
    
    # Specify where the variables should be saved (for now, the directory of the script)
    checkpoint_dir = os.path.dirname(os.path.realpath(__file__))
    # This is the full path to the checkpoint file
    checkpoint_path = checkpoint_dir + '/model-at-global-step'

    # Create a session and run the initialization
    with tf.Session() as sess:
        sess.run(init)
        
        current_state = net._initial_state.eval()
        
        # Get the last saved checkpoint, if any.
        latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir, latest_filename=None)
        
        # Demonstrate that the current state may be erased
        print('Weights when initialized:')
        with tf.variable_scope("RNN", reuse=True):
            print(tf.get_variable("out_w").eval())
        
        # Restore saved state if any existed
        if latest_ckpt:
            saver.restore(sess, latest_ckpt) # restore the saved state
            print("Successfully restored " + os.path.basename(latest_ckpt))
        
        # demonstrate that it actually was restored
        print('Weights after restoring:')
        with tf.variable_scope("RNN", reuse=True):
            print(tf.get_variable("out_w").eval())

        # This is where we iterate through all the data 
        for i in range(0, data_size, batch_size):
            # Our input  will just be a batch with random values
            feed = { net._input : dummyInput(), net._target : dummyTarget(), net._initial_state : current_state} # _initial_state is no placeholder, but we can still give it as an argument (???)
            current_state, _  = sess.run([net._final_state, net._train_op], feed_dict=feed) 

            # Save the state of the network trained in the current session every save_steps step 
            gs = sess.run(global_step)
            if gs % save_steps == 0:
                save_path = saver.save(sess, checkpoint_path, global_step = gs)
                print("Saved at global step " + str(gs))
                
        # To get some kind of output we print the last state in the list 
        print(current_state[-1])
        print('Weights before ending:')
        with tf.variable_scope("RNN", reuse=True):
            print(tf.get_variable("out_w").eval())
        print(sess.run(global_step))
   
if __name__ == "__main__":
    main()

