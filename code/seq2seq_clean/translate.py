# vim: set sw=2 ts=2 expandtab:

# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import pdb

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

#from tensorflow.models.rnn.translate import data_utils
#from tensorflow.models.rnn.translate import seq2seq_model

import data_utils
import seq2seq_model


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("fr_vocab_size", 40000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("adam_optimizer", True, "Set True to use Adam optimizer instead of SGD")
tf.app.flags.DEFINE_integer("max_runtime", 0, "if (max_runtime != 0), stops execution after max_runtime minutes")
tf.app.flags.DEFINE_string("perplexity_log", None, "Filename for logging perplexity")
tf.app.flags.DEFINE_integer("last_bucket_enc_len", 200, "The longest allowed input length for the encoder")
tf.app.flags.DEFINE_integer("last_bucket_dec_len", 50, "The longest allowed input length for the decoder")
tf.app.flags.DEFINE_boolean("evaluation_file", False, "Use the files \"evaluation_a.txt\" and \"evaluation_t.txt\" for evaluation data")
tf.app.flags.DEFINE_boolean("use_roulette_search", False, "Set to true to use roulette search in decoder (much slower)")
tf.app.flags.DEFINE_integer("use_specific_checkpoint", 0, "Integer specifying which checkpoint to use when resoring the model")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
#_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
_buckets = []#, (10, 15), (20, 25), (40, 50)]

_buckets.append((FLAGS.last_bucket_enc_len, FLAGS.last_bucket_dec_len))
print("Using buckets defined as", _buckets)


def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 25000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        # Truncate ids to largest bucket size
        source_ids = ([int(x) for x in source.split()])[:(_buckets[-1][0]-1)]
        target_ids = ([int(x) for x in target.split()])[:(_buckets[-1][1]-2)]
        #pdb.set_trace()
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
      for bucket_id, (source_size, target_size) in enumerate(_buckets):
        print("Length of bucket %d (%d, %d): %d" % 
                (bucket_id, source_size, target_size, len(data_set[bucket_id]))
        )
  return data_set


def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.en_vocab_size, FLAGS.fr_vocab_size, _buckets,
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,
      use_adam_optimizer=FLAGS.adam_optimizer)
  if FLAGS.use_specific_checkpoint:
    print("Using specific checkpoint: %d" % FLAGS.use_specific_checkpoint)
    checkpoint_path = os.path.join(FLAGS.train_dir, ("translate.ckpt-%d" % FLAGS.use_specific_checkpoint))
    ckpt = False
  else:
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    checkpoint_path = ckpt.model_checkpoint_path
  if (ckpt or FLAGS.use_specific_checkpoint) and tf.gfile.Exists(checkpoint_path):
    print("Reading model parameters from %s" % checkpoint_path)
    model.saver.restore(session, checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def train():
  """Train a en->fr translation model using WMT data."""
  # Prepare WMT data.
  print("Preparing WMT data in %s" % FLAGS.data_dir)
  en_train, fr_train, en_dev, fr_dev, _, _ = data_utils.prepare_wmt_data(
      FLAGS.data_dir, FLAGS.en_vocab_size, FLAGS.fr_vocab_size)

  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    #dev_set = read_data(en_dev, fr_dev, FLAGS._max_train_data_size)
    train_set = read_data(en_train, fr_train, FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    time_train_start = time.time()
    while True:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        if FLAGS.perplexity_log:
            with tf.gfile.Open(os.path.join(FLAGS.train_dir, FLAGS.perplexity_log), "a") as logfile:
                logfile.write("%d;%.4f;%.4f;%.4f\n" % 
                        (model.global_step.eval(), model.learning_rate.eval(),
                            step_time, perplexity)
                        )
                logfile.close()
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        #for bucket_id in xrange(len(_buckets)):
        #  if len(dev_set[bucket_id]) == 0:
        #    print("  eval: empty bucket %d" % (bucket_id))
        #    continue
        #  encoder_inputs, decoder_inputs, target_weights = model.get_batch(
        #      dev_set, bucket_id)
        #  _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
        #                               target_weights, bucket_id, True)
        #  eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
        #  print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()
        if FLAGS.max_runtime:
          max_time = int(FLAGS.max_runtime) * 60
          elapsed_time = (time.time() - time_train_start)
          if elapsed_time >= max_time:
            print("Terminated after %d minutes. . . (limit set to %d)" %
                (elapsed_time/60, max_time/60))
            break;
          else:
            sys.stdout.write("%3d minutes left | " % ((max_time - elapsed_time)/60))


def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    en_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.a" % FLAGS.en_vocab_size)
    fr_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.t" % FLAGS.fr_vocab_size)
    en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
    _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # Get token-ids for the input sentence.
      token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)[:(_buckets[-1][0]-1)]
      # Which bucket does it belong to?
      bucket_id = min([b for b in xrange(len(_buckets))
                       if _buckets[b][0] > len(token_ids)])
      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()

def decode_many_slow_and_greedy():
  eval_a = "evaluation_a.txt"
  eval_t = "evaluation_t.txt"
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    en_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.a" % FLAGS.en_vocab_size)
    fr_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.t" % FLAGS.fr_vocab_size)
    en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
    _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    
    articles = []
    titles = []

    with tf.gfile.Open(os.path.join(FLAGS.data_dir, eval_a), "r") as evaluation_file_a:
      for line in evaluation_file_a:
        articles.append(line)
    with tf.gfile.Open(os.path.join(FLAGS.data_dir, eval_t), "r") as evaluation_file_t:
      for line in evaluation_file_t:
        titles.append(line)

    article_title_pairs = zip(articles, titles)

    for (idx, (article, title)) in enumerate(article_title_pairs):
      # Get token-ids for the input sentence.
      token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(article), en_vocab)[:(_buckets[-1][0]-1)]
      # Which bucket does it belong to?
      bucket_id = min([b for b in xrange(len(_buckets))
                       if _buckets[b][0] > len(token_ids)])
      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      print("{:-^80}".format("Article %d" % idx))
      print(article)
      print("{:-^80}".format("Real Title %d" % idx))
      print(title)
      print("{:-^80}".format("Generated Title %d" % idx))
      print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
      print("\n{:#^80}".format(""))
      print("{:#^80}\n".format(""))
      sys.stdout.flush()

def decode_many(use_roulette_search=False):
  eval_a = "evaluation_a.txt"
  eval_t = "evaluation_t.txt"
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    en_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.a" % FLAGS.en_vocab_size)
    fr_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.t" % FLAGS.fr_vocab_size)
    en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
    _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

    # Decode from standard input.
    sys.stdout.flush()
    
    articles = []
    titles = []

    print("Reading evaluation data")
    with tf.gfile.Open(os.path.join(FLAGS.data_dir, eval_a), "r") as evaluation_file_a:
      for line in evaluation_file_a:
        articles.append(line)
    with tf.gfile.Open(os.path.join(FLAGS.data_dir, eval_t), "r") as evaluation_file_t:
      for line in evaluation_file_t:
        titles.append(line)
    model.batch_size = len(articles)
    article_title_pairs = zip(articles, titles)
    id_pairs = []

    print("Tokenizing evaluation data")
    for (idx, (article, title)) in enumerate(article_title_pairs):
      # Get token-ids for the input sentence.
      token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(article), en_vocab)[:(_buckets[-1][0]-1)]
      id_pairs.append((token_ids, []))
    

    # Which bucket does it belong to?
    bucket_id = min([b for b in xrange(len(_buckets))
                     if _buckets[b][0] > len(token_ids)])
    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
        {bucket_id: id_pairs}, bucket_id)
    # Get output logits for the sentence.

    if use_roulette_search:
      target_weights = np.ones_like(target_weights)
      #print("|decoder_inputs|=%d" % len(decoder_inputs))
      #print("|target_weights|=%d" % len(target_weights))
      #print("|encoder_inputs|=%d" % len(encoder_inputs))
      print("Stepping through evaluation data (pairs: %d), word:\n" % model.batch_size)
      for word_idx in xrange(_buckets[-1][1] - 1):
        sys.stdout.write("%d, " % word_idx)
        sys.stdout.flush()
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
            target_weights, bucket_id, True)
        current_word_logit = output_logits[word_idx]
        softmax_op = tf.nn.softmax(current_word_logit)
        # softmaxes[word_idx]<batch><candidates>
        softmaxes = sess.run(softmax_op)
        # chosen_words<batch>=id
        chosen_words = [np.random.choice(list(xrange(len(softmax))), p=softmax) for softmax in softmaxes]
        #pdb.set_trace()
        #decoder_inputs[word_idx]<batch>=id
        decoder_inputs[word_idx + 1] = chosen_words
    else:
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
          target_weights, bucket_id, True)

    
    for (idx, (article, title)) in enumerate(article_title_pairs):
      if use_roulette_search:
        outputs = [word_position[idx] for word_position in decoder_inputs[1:]] 
      else:
        #print(output_logits)
        #print(len(output_logits))
        #print((output_logits[0].shape))
        pdb.set_trace()
        outputs = [int(np.argmax(word_position[idx])) for word_position in output_logits]
        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        #pdb.set_trace()
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      print("{:-^80}".format("Article %d" % idx))
      print(article)
      print("{:-^80}".format("Real Title %d" % idx))
      print(title)
      print("{:-^80}".format("Generated Title %d" % idx))
      print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
      print("\n{:#^80}".format(""))
      print("{:#^80}\n".format(""))
      sys.stdout.flush()



def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)


def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    if FLAGS.evaluation_file:
      use_slow_and_greedy_method = True
      if use_slow_and_greedy_method:
        decode_many_slow_and_greedy()
      else:
        decode_many(FLAGS.use_roulette_search)
    else:
      decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()

