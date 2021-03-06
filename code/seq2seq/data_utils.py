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

"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
from six.moves import urllib

import numpy as np

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="r") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 25000 == 0:
          print("  processing line %d" % counter)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: a string, the sentence to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """
  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="r") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 25000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def string_converter(s):
    return s.decode("utf-8")


# Reads a glove file and returns an numpy array where the first column
# contains the words and columns > 0 contains the values in each dimension
def read_glove(input_file, dimensions):

    # The format for the words: Unicode, up to 128 characters
    columns_datatype = [("word", np.dtype("U128"))]

    # Create the datatypes for the dimensional data: 8 byte floats
    for i in range(dimensions):
        columns_datatype.append(("dim{}".format(i), np.dtype("f8")))

    # print("Using the following datatypes for the columns\n")
    # print(columns_datatype)

    glove_data = np.genfromtxt(
        fname=input_file,
        dtype=columns_datatype,
        converters={0: string_converter},
        comments=None,
        delimiter=' ',
        loose=False
    )
    return glove_data


def glove_data_to_dict(glove_data, dimensions):
    index_words = "word"
    index_vectors = []
    for dim in range(dimensions):
        index_vectors.append("dim{}".format(dim))

    words = glove_data[index_words]
    vectors = glove_data[index_vectors]
    glove_dict = dict()

    for i in range(glove_data.size):
        glove_dict[words[i]] = vectors[i]

    return glove_dict


def read_glove_to_dict_of_strings(input_file):
    with gfile.GFile(input_file, mode="r") as source_file:
        glove_dict = dict()
        line = source_file.readline()
        counter = 0
        while line:
            counter += 1
            if counter % 100000 == 0:
              print("  Reading glove word %d" % counter)
            word, _, vector = line.partition(" ")
            glove_dict[word] = vector.rstrip()
            line = source_file.readline()
    return glove_dict


def glove_vector_vocab_to_array(source_glove_vocab):
    vectors = np.loadtxt(source_glove_vocab, dtype=np.float32, delimiter=" ")
    return vectors


def glove_vector_vocab_from_vocabulary(source_vocab, source_glove, target_glove_vocab, dimensions, default_initializer, glove_dict=None):
    if gfile.Exists(target_glove_vocab):
        print("Glove vocab \"%s\" already exists" % target_glove_vocab)
        return None

    print("Creating glove vocab at \"%s\" from vocab \"%s\" using glove vectors \"%s\"" %
          (target_glove_vocab, source_vocab, source_glove))
    print("Reading glove vectors. . .")
    #glove_data = read_glove(source_glove, dimensions)
    if glove_dict is None:
        print("Constructing dict")
        #glove_dict = glove_data_to_dict(glove_data, dimensions)
        glove_dict = read_glove_to_dict_of_strings(source_glove)
    else:
        print("Reusing dict")
    print("Translating words from \"%s\" with vectors from \"%s\"" % 
            (source_vocab, source_glove))
    with gfile.GFile(source_vocab, mode="r") as source_file:
      with gfile.GFile(target_glove_vocab, mode="w") as target_file:
        counter = 0
        failed_words = []
        source_word = source_file.readline()
        while source_word:
          source_word = source_word.rstrip()
          #source_word = source_word.replace("\n", "")
          counter += 1
          if counter % 1000 == 0:
            print("  Glovifying word %d (%s)" % (counter, source_word))
          vector = glove_dict.get(source_word)
          if vector is None:
            failed_words.append(source_word)
            vector = default_initializer()
          target_file.write(vector + "\n")
          source_word = source_file.readline()
    print("Failed to match %d words: " % len(failed_words))
    print(failed_words)
    return glove_dict



def prepare_news_data(data_dir, article_file, title_file, article_vocabulary_size, title_vocabulary_size):
  """Get new data into data_dir, create vocabularies and tokenize data.

  Args:
    data_dir: directory in which the data sets will be stored.
    article_vocabulary_size: size of the articles vocabulary to create and use.
    titles_vocabulary_size: size of the titles vocabulary to create and use.

  Returns:
    A tuple of 4 elements:
      (1) path to the token-ids for article training data-set,
      (2) path to the token-ids for titles training data-set,
      (3) path to the articles vocabulary file,
      (4) path to the titles vocabulary file.
  """

  # Create vocabularies of the appropriate sizes.
  title_vocab_path = os.path.join(data_dir, "vocab%d.title" % title_vocabulary_size)
  title_src_path = os.path.join(data_dir, title_file)
  article_vocab_path = os.path.join(data_dir, "vocab%d.article" % article_vocabulary_size)
  article_src_path = os.path.join(data_dir, article_file)
  create_vocabulary(title_vocab_path, title_src_path, title_vocabulary_size)
  create_vocabulary(article_vocab_path, article_src_path, article_vocabulary_size)

  # Create token ids for the training data.
  title_train_ids_path = ("train_ids.ids%d.title" % title_vocabulary_size)
  article_train_ids_path =  ("train_ids.ids%d.article" % article_vocabulary_size)
  data_to_token_ids(title_src_path, title_train_ids_path, title_vocab_path)
  data_to_token_ids(article_src_path, article_train_ids_path, article_vocab_path)

  return (article_train_ids_path, title_train_ids_path,
          article_vocab_path, title_vocab_path)
