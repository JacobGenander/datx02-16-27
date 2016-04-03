#! /usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gensim
import argparse
import cPickle as pickle
import hyperParams
import numpy as np
from DataMan import DataMan

EMBEDDING_SIZE = 300

parser = argparse.ArgumentParser(description=
    'Create an embedding matrix with word2vec')

parser.add_argument('data_path',
        help='path to data file')
parser.add_argument('vectors_path',
        help='path to pretrained word vectors')

def main():
    args = parser.parse_args()
    config = hyperParams.config

    data_set = DataMan(args.data_path, config["max_seq"], max_vocab_size=config["max_vocab_size"])
    id_to_word = DataMan.id_to_word
    vocab_size = len(id_to_word)

    model = gensim.models.Word2Vec.load_word2vec_format(args.vectors_path, binary=True)

    init_range = config["init_range"]
    emb_matrix = np.zeros([vocab_size, EMBEDDING_SIZE])
    for i in range(vocab_size):
        word = id_to_word[i]
        try:
            vector = model[word]
        except:
            vector = np.random.uniform(-init_range, init_range, [EMBEDDING_SIZE]) # Init range must be same as in LSTM
        emb_matrix[i] = vector

    pickle.dump(emb_matrix, open("embedding.p", "wb"))

if __name__ == "__main__":
    main()
