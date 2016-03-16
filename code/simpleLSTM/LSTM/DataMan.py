#! /usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import nltk
import re
import numpy as np

nltk.download('punkt')

class DataMan(object):
    # Static variables
    vocab_size = 0
    word_to_id = None
    id_to_word = ["_EOS", "_PAD", "_UNK"]

    eos_id = 0
    pad_id = 1
    unk_id = 2

    def __init__(self, filename, max_seq, rebuild_vocab=True, max_vocab_size=10000):
        self._max_seq = max_seq
        raw_data = None
        with open(filename, 'r') as f:
            raw_data = f.read()
        if rebuild_vocab:
            self._build_vocab(raw_data, max_vocab_size)
        self._prepare_data(raw_data)

    def _tokenize(self, text):
        text = re.sub('\d', '0', text) # Replace all digits with 0
        text = text.replace("\n", " _EOS ")
        return nltk.word_tokenize(text)

    def _build_vocab(self, raw_data, max_vocab_size):
        self._data = self._tokenize(raw_data)

        # Give the words ids based on the number of occurrences in the data set
        counter = collections.Counter(self._data)
        count_pairs = counter.most_common()
        DataMan.id_to_word.extend([ word for word, _ in count_pairs if word != "_EOS" ])
        DataMan.id_to_word = sort_words = DataMan.id_to_word[:max_vocab_size]
        DataMan.word_to_id = dict(zip(sort_words, range(len(sort_words))))
        DataMan.vocab_size = len(DataMan.word_to_id)

    def _sentence_to_ids(self, sentence):
        split_sent = self._tokenize(sentence)
        return [ DataMan.word_to_id.get(word, DataMan.unk_id) for word in split_sent]

    def _prepare_data(self, raw_data):
        sentences = raw_data.splitlines(True)
        self._data_len = data_len = len(sentences)

        s_split = [ self._sentence_to_ids(s) for s in sentences]
        s_split.sort(key=len)

        max_seq = self._max_seq
        self._data = np.zeros([data_len, max_seq], dtype=np.int)
        self._seq_lens = np.ones([data_len], dtype=np.int) * DataMan.pad_id
        for i, s  in enumerate(s_split):
            if len(s) > max_seq:
                s = s[:max_seq]
            self._seq_lens[i] = len(s)
            fill = [DataMan.pad_id]*(max_seq - len(s))
            self._data[i] = s + fill

    def batch_iterator(self, batch_size):
        epoch_size = self._data_len // batch_size
        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

        # Generate batches (not foolproof but should work if the data is sane)
        fill = np.ones([batch_size,1], dtype=np.int) * DataMan.pad_id
        for i in range(0, epoch_size):
            start = i*batch_size
            x = self._data[start : start+batch_size]
            # Shift by one and pad with zeros to get targets
            y = np.column_stack((self._data[start : start+batch_size, 1:], fill))
            z = self._seq_lens[start : start+batch_size]
            yield (x, y, z)

