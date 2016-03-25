#! /usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

TRAIN_SET = 0
VALID_SET = 1
TEST_SET = 2

class DataMan(object):

    def __init__(self, filename, eval_ratio):
        with open(filename, 'r') as f:
            text = f.read().lower()
        self._build_vocab(text)
        self._create_sets(text, eval_ratio)

    def _build_vocab(self, text):
        chars = list(set(text))
        self.char_to_id = { c:i for i,c in enumerate(chars) }
        self.id_to_char = { i:c for i,c in enumerate(chars) }
        self.vocab_size = len(chars)

    def _create_sets(self, text, eval_ratio):
        data = [ self.char_to_id[c] for c in text ] 
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
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

        for i in range(epoch_size):
            x = data[:, i*num_steps:(i+1)*num_steps]
            y = data[:, i*num_steps+1:(i+1)*num_steps+1]
            yield (x, y)
        
