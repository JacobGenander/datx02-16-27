#! /usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf

class DataMan(object):
    def __init__(self, filename):
        raw_data = None
        with open(filename, 'r') as f:
            raw_data = f.read() 
        self._build_vocab(raw_data)
        self._prepare_data(raw_data)

    def _tokenize(self, text):
        return text.replace('\n', ' <eos> ').split()

    def _build_vocab(self, raw_data):
        self._data = self._tokenize(raw_data)
        
        # Give the words ids based on the number of occurrences in the data set 
        counter = collections.Counter(self._data)
        count_pairs = counter.most_common() 
        self._id_to_word = sort_words = [ word for word, _ in count_pairs ]
        self._word_to_id = dict(zip(sort_words, range(len(sort_words))))
        self._vocab_size = len(self._word_to_id)

    def _sentence_to_ids(self, sentence):
        split_sent = self._tokenize(sentence)
        return [ self._word_to_id[word] for word in split_sent]

    def _prepare_data(self, raw_data):
        sentences = raw_data.splitlines(True) 
        self._data_len = data_len = len(sentences)
    
        s_split = [ self._sentence_to_ids(s) for s in sentences]
        self._max_seq = max_seq = max([ len(s) for s in s_split])

        self._data = np.zeros([data_len, max_seq], dtype=np.int)
        for i, s  in enumerate(s_split):
            fill = [0]*(max_seq - len(s))
            self._data[i] = s + fill

    def batch_iterator(self, batch_size):
        epoch_size = self._data_len // batch_size 
        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    
        # Shuffle the data to generate different batches each time
        np.random.shuffle(self._data)
    
        # Generate batches (not foolproof but should work if the data is sane) 
        fill = np.zeros([batch_size,1], dtype=np.int)
        for i in range(0, epoch_size, batch_size):
            x = self._data[i : i+batch_size]
            # Shift by one and pad with zeros to get targets
            y = np.column_stack((self._data[i : i+batch_size, 1:], fill))
            yield (x, y)

    @property
    def id_to_word(self):
        return self._id_to_word

    @property
    def word_to_id(self):
        return self._word_to_id

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

def main():
    reader = DataMan('titles.txt')
    batch_size = 10
    
    i = 0
    for x, y in reader.batch_iterator(batch_size):
        i += 1
        if i == 100:
            print('X:')
            print(x[0])
            print('Y:')
            print(y[0])
                
    print("Data length: {0}".format(reader.data_len))
    print("Batches: {0}".format(i))
    print("Vocab size: {0}".format(reader.vocab_size))
    print("Longest sentence: {0}".format(reader.max_seq))
            
if __name__ == "__main__":
    main()
            
