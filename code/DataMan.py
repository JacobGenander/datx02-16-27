#! /usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf

class DataMan(object):
    def __init__(self, filename, dicts=None):
        raw_data = None
        with open(filename, 'r') as f:
            raw_data = f.read()
        if dicts == None:
            self._build_vocab(raw_data)
        else:
            self._word_to_id = dicts[0]
            self._id_to_words = dicts[1]
            self._unk_id = self._word_to_id['<unk>']
            self._vocab_size = len(self._word_to_id)
        self._prepare_data(raw_data)

    def _tokenize(self, text):
        return text.replace('\n', ' <eos> ').split()

    def _build_vocab(self, raw_data):
        self._data = self._tokenize(raw_data) + ['<unk>']

        # Give the words ids based on the number of occurrences in the data set
        counter = collections.Counter(self._data)
        count_pairs = counter.most_common()
        self._id_to_word = sort_words = [ word for word, _ in count_pairs ]
        self._word_to_id = dict(zip(sort_words, range(len(sort_words))))
        self._unk_id = self._word_to_id['<unk>']
        self._vocab_size = len(self._word_to_id)

    def _sentence_to_ids(self, sentence):
        split_sent = self._tokenize(sentence)
        return [ self._word_to_id.get(word, self._unk_id) for word in split_sent]

    def _prepare_data(self, raw_data):
        sentences = raw_data.splitlines(True)
        self._data_len = data_len = len(sentences)

        s_split = [ self._sentence_to_ids(s) for s in sentences]
        self._max_seq = max_seq = max([ len(s) for s in s_split])

        self._data = np.zeros([data_len, max_seq], dtype=np.int)
        self._seq_lens = np.zeros([data_len], dtype=np.int)
        for i, s  in enumerate(s_split):
            self._seq_lens[i] = len(s)
            fill = [0]*(max_seq - len(s))
            self._data[i] = s + fill

    def _shuffle_data(self):
        seed = np.random.randint(10000)
        np.random.seed(seed)
        np.random.shuffle(self._data)
        np.random.seed(seed)
        np.random.shuffle(self._seq_lens)

    def batch_iterator(self, batch_size):
        epoch_size = self._data_len // batch_size
        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

        # Shuffle the data to generate different batches each time
        self._shuffle_data()

        # Generate batches (not foolproof but should work if the data is sane)
        fill = np.zeros([batch_size,1], dtype=np.int)
        for i in range(0, epoch_size):
            start = i*batch_size
            x = self._data[start : start+batch_size]
            # Shift by one and pad with zeros to get targets
            y = np.column_stack((self._data[start : start+batch_size, 1:], fill))
            z = self._seq_lens[start : start+batch_size]
            yield (x, y, z)

    def get_dicts(self):
        return (self._word_to_id, self._id_to_word)

    @property
    def word_to_id(self):
        return self._word_to_id

    @property
    def id_to_word(self):
        return self._id_to_word

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def max_seq(self):
        return self._max_seq

    @property
    # This is the number of sentences in the data set
    def data_len(self):
        return self._data_len

