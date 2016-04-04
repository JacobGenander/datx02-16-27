#! /usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import nltk
import re

TRAIN_SET = 0
VALID_SET = 1
TEST_SET = 2

EOS_ID = 0
UNK_ID = 1

nltk.download('punkt')

class DataMan(object):

    def __init__(self, filename, eval_ratio):
        with open(filename, 'r') as f:
            text = f.read()
        self._build_vocab(text)
        self._create_sets(text, eval_ratio)

    def _tokenize(self, text):
        text = re.sub(br'\d', '0', text) # Replace all digits with zeros
        text = text.replace('\n', ' _EOS ')
        return nltk.word_tokenize(text)

    def _build_vocab(self, text):
        self._data = self._tokenize(text)
        counter = collections.Counter(self._data)
        count_pairs = counter.most_common()
        self.id_to_word = ['_EOS', '_UNK']
        self.id_to_word.extend([ word for word, c in count_pairs if word != '_EOS' and c > 1])
        self.word_to_id = { w:i for i,w in enumerate(self.id_to_word) }
        self.vocab_size = len(self.word_to_id)

    def _create_sets(self, text, eval_ratio):
        data = [ self.word_to_id.get(w, UNK_ID) for w in self._data ]
        self.data_len = len(data)
        eval_len = int(self.data_len * eval_ratio)

        test = data[:eval_len]
        valid = data[eval_len:eval_len*2]
        train = data[eval_len*2:]
        self.data_sets = [train, valid, test]

    def batch_iterator(self, batch_size, num_steps, data_set):
        raw_data = np.array(self.data_sets[data_set], dtype=np.int32)
        data_len = len(raw_data)
        batch_len = data_len // batch_size
        data = np.zeros([batch_size, batch_len], dtype=np.int32)
        for i in range(batch_size):
            data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

        epoch_size = (batch_len - 1) // num_steps

        if epoch_size == 0:
            raise ValueError('epoch_size == 0, decrease batch_size or num_steps')

        for i in range(epoch_size):
            x = data[:, i*num_steps:(i+1)*num_steps]
            y = data[:, i*num_steps+1:(i+1)*num_steps+1]
            yield (x, y)

