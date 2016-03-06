#! /usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf

class DataMan(object):
    # Static variables
    max_seq = 30
    vocab_size = 0
    word_to_id = None
    id_to_word = None
    unk_id = None

    def __init__(self, filename, rebuild_vocab=True):
        raw_data = None
        with open(filename, 'r') as f:
            raw_data = f.read()
        if rebuild_vocab:
            self._build_vocab(raw_data)
        self._prepare_data(raw_data)

    def _tokenize(self, text):
        return text.replace('\n', ' <eos> ').split()

    def _build_vocab(self, raw_data):
        self._data = self._tokenize(raw_data) + ['<unk>']

        # Give the words ids based on the number of occurrences in the data set
        counter = collections.Counter(self._data)
        count_pairs = counter.most_common()
        DataMan.id_to_word = sort_words = [ word for word, _ in count_pairs ]
        DataMan.word_to_id = dict(zip(sort_words, range(len(sort_words))))
        DataMan.unk_id = DataMan.word_to_id['<unk>']
        DataMan.vocab_size = len(DataMan.word_to_id)

    def _sentence_to_ids(self, sentence):
        split_sent = self._tokenize(sentence)
        return [ DataMan.word_to_id.get(word, DataMan.unk_id) for word in split_sent]

    def _prepare_data(self, raw_data):
        sentences = raw_data.splitlines(True)
        self._data_len = data_len = len(sentences)
        max_seq = DataMan.max_seq

        s_split = [ self._sentence_to_ids(s) for s in sentences]

        self._data = np.zeros([data_len, max_seq], dtype=np.int)
        self._seq_lens = np.zeros([data_len], dtype=np.int)
        for i, s  in enumerate(s_split):
            if len(s) > max_seq:
                s = s[:max_seq]
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

    @property
    # This is the number of sentences in the data set
    def data_len(self):
        return self._data_len

