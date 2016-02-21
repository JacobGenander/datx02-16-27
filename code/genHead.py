#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import reader

# Hardcoded for the moment
hidden_layer_size = 200
vocab_size = 20984

def main():
    w = tf.Variable(tf.zeros([hidden_layer_size, vocab_size]), name="out_w")
    b = tf.Variable(tf.zeros([vocab_size]), name="out_b")

    sess = tf.Session()
    saver = tf.train.Saver()
   
    _ , id_to_word = reader.build_vocab("./titles2.txt")
    saver.restore(sess, "/tmp/model.ckpt")

if __name__ == "__main__":
    main()


