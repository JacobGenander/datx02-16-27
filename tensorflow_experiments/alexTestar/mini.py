'''
Simplified model. 
'''

import tf_embed
import tensorflow as tf
import numpy as np
import os 

# Parameters
learning_rate = 0.06
training_epochs = 100
batch_size = 150
data_size = 3000

display_step = 1
embedding = tf_embed.get_embeding()
dictionary, reverse_dictionary = tf_embed.get_dictionaries() ## dictionary['word'] gets index of 'word' in embedding.
vocab_size = len(dictionary)


#Read data
with open('headlines.train.txt') as f:
    words = f.read().replace('\n',' EOS ').split()[:data_size]
    data = tf.nn.embedding_lookup(embedding, [dictionary.get(w, 0) for w in words]) ## Get word or the UNK word at index 0

# tf Graph Input
x = tf.placeholder("float", [None, 100]) # Vector representing a word
y = tf.placeholder("float", [None, vocab_size]) # multinomial of how likly each word in vocab is as next word.

# Create model

# Set model weights
W = tf.Variable(tf.zeros([100, vocab_size]))
b = tf.Variable(tf.zeros([vocab_size]))

# Construct model
activation = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indices=1)) # Cross entropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # Gradient Descent

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
  #  # Set up saver
  #  saver = tf.train.Saver()
  #  checkpoint_dir = os.path.dirname(os.path.realpath(__file__))
  #  checkpoint_path = checkpoint_dir + '/mini'
  #  
  #  # Restore eventual checkpoint
  #  latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir, latest_filename=None)
  #  if latest_ckpt:
  #      saver.restore(sess, latest_ckpt) # restore the saved state
  #      print("Successfully restored " + os.path.basename(latest_ckpt))

    
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = len(words)/batch_size
        # Loop over all batches
        for i in range(total_batch):
            # Slice out a batch of words
            word_slice = tf.slice(data, [i*batch_size,0],[batch_size,-1])
            
            # Unpack first dimension to a list, i.e go from [batch_size, vocab_length] to [[vocab_length]] of length batch_size.
            # We also need to eval the tf nodes before they can be fed as a dict.
            batch_xs = [w.eval() for w in tf.unpack(word_slice)]
            #Set tartgets to 1-hot multinomials, ravel flattens the ndarray to 1d.
            batch_ys = [np.eye(vocab_size,1,dictionary.get(words[j],0)).ravel() for j in range(i*batch_size,(i+1)*batch_size)]
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
            print "Done with batch number:"
            print i+1
        # Display logs per epoch step and save checkpoint
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
  #          saver.save(sess, checkpoint_path, global_step = epoch)

    print "Optimization Finished!"

    # Test model by generating a headline.
    next_word = tf.argmax(activation, 1)
    headline = [dictionary["a"]]
    pred_word = sess.run(next_word, feed_dict={x:[embedding[dictionary['a']]]})[0]
    headline.append(pred_word)
    for i in range(15):
        pred_word = sess.run(next_word, feed_dict={x:[embedding[pred_word]]})[0]
        headline.append(pred_word)
    print [reverse_dictionary[i] for i in headline]
           
             
