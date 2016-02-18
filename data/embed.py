"""
Create get a list of vector batches and a word to vector dict.
Hard coded paths call get_data(batch_size) to get batches of size 
batch_size and a vector dict of the words in the vocabulary.
"""

import os

import numpy as np
import tensorflow as tf
import collections

#Path to glove file
glovename = "glove.6B.100d.txt"

#Path to headlines
headlinename = "headlines.train.txt"

#Size of vocab
vocab_size = 50000


def _read_glove(glovename):
  lookup = {}
  lookup["<eos>"]= np.random.rand(100)
  lookup["<unk>"]= np.random.rand(100)
  with open(glovename, "r") as f:
    for line in f:
       data = line.split()
       key = data[0]
       value =np.array(map(float, data[1:]))
       lookup[key] = value
  return lookup 

def _read_words(filename):
  with open(filename, "r") as f:
    return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
  lookup = _read_glove(glovename)
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  words = list(words)[:vocab_size-2]
  words.append("<unk>")
  words.append("<eos>")
  word_to_vec = dict(zip(words, map(lambda w : lookup.get(w, "<unk>"), words)))

  return word_to_vec


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id.get(word,"<unk>") for word in data]


def get_data(batch_size):
  word_to_id = _build_vocab(headlinename)
  data = _file_to_word_ids(headlinename, word_to_id)
  batches = [data[x:x+batch_size] for x in xrange(0,len(data), batch_size)]
  return batches, word_to_id

