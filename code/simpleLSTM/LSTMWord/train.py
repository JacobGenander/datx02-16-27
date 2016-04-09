#! /usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cPickle as pickle
import DataManager
import argparse
import time
import sys
import os
from shutil import copy

# TensorFlow's API (if you want to know what arguments we pass to the different methods)
# https://www.tensorflow.org/versions/r0.7/api_docs/python/index.html

parser = argparse.ArgumentParser()

# -------- Data paths and saving --------
parser.add_argument('--data_path', type=str, default='data/data.txt',
        help='path to data set')
parser.add_argument('--save_dir', type=str, default='results',
        help='directory to store model and graphs')
parser.add_argument('--checkpoint_dir', type=str, default='',
        help='resume training from checkpoint found at given directory') 
parser.add_argument('--save_epoch', type=int, default=1,
        help='decides how often we will save our progress')
parser.add_argument('--time_out', type=int, default=3600,
        help='stop training after this amount of seconds') 
parser.add_argument('--overwrite', action='store_true', default=False,
        help='overwrite existing model in save directory')
# -------- Parameters for data processing
parser.add_argument('--threshold', type=int, default=1,
        help='words occuring fewer times than this will not be included')
parser.add_argument('--eval_ratio', type=float, default=0.05,
        help='amount of training data used for evaluation')
# -------- Model parameters --------
parser.add_argument('--batch_size', type=int, default=50,
        help='number of sequences to train on in prallell')
parser.add_argument('--max_epoch', type=int, default=50,
        help='number of passes through the data set')
parser.add_argument('--num_steps', type=int, default=50,
        help='number of timesteps to unroll for')
parser.add_argument('--num_layers', type=int, default=2,
        help='number of neuron layers')
parser.add_argument('--layer_size', type=int, default=100,
        help='number of neurons in each layer')
parser.add_argument('--decay_start', type=int, default=10,
        help='decay learning rate after this epoch')
parser.add_argument('--learning_rate', type=float, default=0.002,
        help='starter learning rate')
parser.add_argument('--learning_decay', type=float, default=1.0,
        help='learning rate decay')
parser.add_argument('--gradient_clip', type=int, default=5,
        help='clip gradients at this value')
parser.add_argument('--keep_prob', type=float, default=1.0,
        help='probability that input/output is kept. 1 = No dropout')
parser.add_argument('--init_range', type=float, default=0.08,
        help='initiate parameters withing this range. -/+ init_range')


class LSTM_Network(object):
    def __init__(self, training, conf):
        self.batch_size = batch_size = conf.batch_size
        self.num_steps = num_steps = conf.num_steps
        self.size = size = conf.layer_size
        keep_prob = conf.keep_prob
        vocab_size = conf.vocab_size

        # 2-dimensional tensors for input data and targets
        self._input = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._target = tf.placeholder(tf.int64, [batch_size, num_steps])

        # Fetch embeddings
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [vocab_size, size])
            inputs = tf.nn.embedding_lookup(embedding, self._input)

        if keep_prob < 1 and training:
            inputs = tf.nn.dropout(inputs, keep_prob)

        # Create the network
        cell = tf.nn.rnn_cell.BasicLSTMCell(size)
        if keep_prob < 1 and training:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        stacked_cells = tf.nn.rnn_cell.MultiRNNCell([cell] * conf.num_layers)

        self.initial_state = stacked_cells.zero_state(batch_size, tf.float32)

        # Give input the right shape
        inputs =[tf.squeeze(input_, [1]) for input_ in tf.split(1, num_steps, inputs)]

        # Run through the whole batch and update state
        outputs, state = tf.nn.rnn(stacked_cells, inputs, initial_state=self.initial_state)
        self.final_state = state

        # Turn output into tensor instead of list
        outputs = tf.concat(1, outputs)

        # Flatten output into a tensor where each row is the output from one word
        output = tf.reshape(outputs, [-1, size])

        w = tf.get_variable('out_w', [size, vocab_size])
        b = tf.get_variable('out_b', [vocab_size])
        z = tf.matmul(output, w) + b # Add supports broadcasting over each row

        # Average negative log probability
        targets = tf.reshape(self._target, [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(z, targets)
        self.cost = cost = tf.reduce_sum(loss) / batch_size
        # Accuracy is calculated by looking for each target in the top 5 most predicted
        correct_preds = tf.nn.in_top_k(tf.nn.softmax(z), targets, 5)
        self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

        if not training:
            self.train_op = tf.no_op()
            return

        self._learning_rate = tf.Variable(0.0, trainable=False)

        # Clip the gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), conf.gradient_clip)
        # Training op
        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads,tvars))

    def set_learning_rate(self, sess, value):
        sess.run(tf.assign(self._learning_rate, value))

def run_epoch(sess, net, data_man, data_set):
    total_cost, total_acc = 0.0, 0.0
    iters = 0
    state = sess.run(net.initial_state)
    for i, (x, y) in enumerate(data_man.batch_iterator(net.batch_size, net.num_steps, data_set)):
        # Input to network
        feed = { net._input : x, net._target : y , net.initial_state : state}
        # Calculate cost and train the network
        cost, state, acc,  _ = sess.run([net.cost, net.final_state, net.accuracy, net.train_op], feed_dict=feed)
        total_acc += acc
        total_cost += cost
        iters += net.num_steps
    return total_cost / iters, total_acc / (i+1)

def save_state(sess, saver, conf, epoch):
    print('Saving model...')
    save_path_model = os.path.join(conf.save_dir, 'model.ckpt')
    path = saver.save(sess, save_path_model)
    save_path_config = os.path.join(conf.save_dir, 'config.p')
    pickle.dump(conf, open(save_path_config, 'wb'))
    # Also save the results in a separate folder
    dir_path = os.path.join(conf.save_dir, 'epoch' + str(epoch))
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    copy(save_path_model, dir_path)
    copy(save_path_config, dir_path)
    print('Saved in ' + path)

def init_config(parser, data_man):
    parser.add_argument('--start_epoch', default=0)
    parser.add_argument('--vocab_size', default=data_man.vocab_size)
    parser.add_argument('--word_to_id', default=data_man.word_to_id)
    parser.add_argument('--id_to_word', default=data_man.id_to_word)
    parser.add_argument('--file_hash', default=data_man.file_hash)
    parser.add_argument('--cost_train', default=[])
    parser.add_argument('--cost_valid', default=[])
    parser.add_argument('--accuracy', default=[])
    return parser.parse_args()

def main():
    start_time = time.time()
    conf = parser.parse_args() 

    # We don not want to run anything without knowing that we can save our results
    save_dir = conf.save_dir
    if not os.path.isdir(save_dir):
        print('Could not find save directory')
        sys.exit(1)

    model_save_path = os.path.join(save_dir, 'model.ckpt')
    if os.path.isfile(model_save_path) and not conf.overwrite:
        print('Save path already conatins a model file.')
        ans = raw_input('Overwrite and continue? (y/n) ')
        if ans == 'y':
            pass
        else:
            print('Exiting.')
            sys.exit(0)

    # Load previous model if a checkpoint path is provided
    checkpoint_dir = conf.checkpoint_dir
    if not conf.checkpoint_dir:
        data_man = DataManager.DataMan(conf.data_path, conf.eval_ratio, conf.threshold)
        conf = init_config(parser, data_man)
    else:
        config_path = os.path.join(conf.checkpoint_dir, 'config.p')
        with open(config_path, 'rb') as f:
            conf = pickle.load(f)
            conf.save_dir = save_dir
            conf.checkpoint_dir = checkpoint_dir
            data_man = DataManager.DataMan(conf.data_path, conf.eval_ratio, conf.threshold)
            if data_man.file_hash != conf.file_hash:
                print('File does not match checksum found in checkpoint.')
                sys.exit(1)

    # Create networks for training and evaluation
    initializer = tf.random_uniform_initializer(-conf.init_range, conf.init_range)
    with tf.variable_scope('model', reuse=None, initializer=initializer):
        train_net = LSTM_Network(True, conf)
    with tf.variable_scope('model', reuse=True, initializer=initializer):
        val_net = LSTM_Network(False, conf)
        # We need to make som adjustments for the test net
        t_batch_size, t_num_steps = conf.batch_size, conf.num_steps
        conf.batch_size, conf.num_steps = 1, 1
        test_net = LSTM_Network(False, conf)
        # Since we are saving the configuration we have to restore the old values
        conf.batch_size, conf.num_steps = t_batch_size, t_num_steps

    # We always need to run this operation if not loading an old state
    init = tf.initialize_all_variables()
    # This object will save our state at certain intervals 
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize variables
        if not conf.checkpoint_dir:
            sess.run(init)
        else:
            model_path = os.path.join(conf.checkpoint_dir, 'model.ckpt')
            saver.restore(sess, model_path)

        max_epoch = conf.max_epoch
        quit_training = False
        print('Training.')
        for i in range(conf.start_epoch, max_epoch):
            # Code needed for learning rate decay
            decay = conf.learning_decay ** max(i - conf.decay_start, 0.0)
            train_net.set_learning_rate(sess, conf.learning_rate * decay)

            # Train the network and evaluate it
            cost_t, acc_t = run_epoch(sess, train_net, data_man, DataManager.TRAIN_SET)
            conf.cost_train.append(cost_t)
            cost_v, acc_v = run_epoch(sess, val_net, data_man, DataManager.VALID_SET)
            conf.cost_valid.append(cost_v)
            conf.accuracy.append(acc_v)

            time_stamp = time.time() - start_time
            if time_stamp >= conf.time_out:
                quit_training = True

            # Print some results
            print('----- Running time {0:.2f}s (epoch {1}/{2}) -----'.format(time_stamp, i+1, max_epoch))
            print('\t\ttraining\tvalidation')
            print('COST:\t\t{0:.4f}\t\t{1:.4f}'.format(cost_t, cost_v))
            print('ACCURACY:\t{0:.4f}\t\t{1:.4f}'.format(acc_t, acc_v))
            print('PERPLEXITY:\t{0:.2f}\t\t{1:.2f}'.format(np.exp(cost_t), np.exp(cost_v)))
            
            # See if it is time to save
            if (i % conf.save_epoch == 0) or (i == max_epoch - 1) or quit_training:
                conf.start_epoch = i+1
                save_state(sess, saver, conf, i)
                if quit_training:
                    print('Time out.')
                    break

        # Run evaluation on test set if all trainig is done
        if not quit_training:
            print('Evaluating on test set.')
            test_cost, test_acc = run_epoch(sess, test_net, data_man, DataManager.TEST_SET)
            print('PERPLEXITY:\t{0:.2f}'.format(np.exp(test_cost)))
            print('ACCURACY:\t{0:.4f}'.format(test_acc))

        sess.close()
        print('Training finished.')
        print('----- Total running time {0:.2f}s -----'.format(time.time() - start_time))

if __name__ == '__main__':
    main()

