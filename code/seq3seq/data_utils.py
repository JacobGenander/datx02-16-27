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
import nltk.data

import multiprocessing

from six.moves import urllib

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


nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size):
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
    #pool = multiprocessing.Pool(8)
    with gfile.GFile(data_path, mode="r") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 10000 == 0:
          print("  processing line %d" % counter)

        text = tokenizer.tokenize(line)
        text = [nltk.word_tokenize(s) for s in text]
        #text = pool.map(nltk.word_tokenize, text)
        tokens = [tok for sent in text for tok in sent]
        for word in tokens:
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

def text_to_token_ids(text, vocab, preserve_sent=False):
  #pool = multiprocessing.Pool(8)
  tok_text = tokenizer.tokenize(text)
  tok_text = [nltk.word_tokenize(s) for s in tok_text]
  #tok_text = map(nltk.word_tokenize, tok_text)
  id_text = []
  if preserve_sent:
    for sent in tok_text:
      id_sent = []
      id_sent.extend([vocab.get(w, UNK_ID) for w in sent])
      id_text.append(id_sent)
  else:
    for sent in tok_text:
      id_text.extend([vocab.get(w, UNK_ID) for w in sent])
      id_text.append(EOS_ID)
  return id_text

def data_to_token_ids(data_path, target_path, vocabulary_path):
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
          if counter % 10000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = text_to_token_ids(line, vocab)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


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
  title_vocab_path = os.path.join(data_dir, "vocab%d.titles3" % title_vocabulary_size)
  title_src_path = os.path.join(data_dir, title_file)
  article_vocab_path = os.path.join(data_dir, "vocab%d.article3" % article_vocabulary_size)
  article_src_path = os.path.join(data_dir, article_file)
  create_vocabulary(title_vocab_path, title_src_path, title_vocabulary_size)
  create_vocabulary(article_vocab_path, article_src_path, article_vocabulary_size)

  # Create token ids for the training data.
  title_train_ids_path = ("train_ids.ids%d.title3" % title_vocabulary_size)
  article_train_ids_path =  ("train_ids.ids%d.article3" % article_vocabulary_size)
  data_to_token_ids(title_src_path, title_train_ids_path, title_vocab_path)
  data_to_token_ids(article_src_path, article_train_ids_path, article_vocab_path)

  return (article_train_ids_path, title_train_ids_path,
          article_vocab_path, title_vocab_path)
