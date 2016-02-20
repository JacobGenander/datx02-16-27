#! /usr/bin/env python3

import collections
import os

import numpy as np
import tensorflow as tf

def _tokenize(text):
    return text.replace('\n', ' <eos> ').replace('/',' ').split()

def _read_words(filename):
    with open(filename) as f:
        return _tokenize(f.read()) 

def _build_vocab(filename):
    data = _read_words(filename) 

    counter = collections.Counter(data)
    count_pairs = counter.most_common() 
    sort_words = [ word for word, _ in count_pairs ]
    word_to_id = dict(zip(sort_words, range(len(sort_words))))
    id_to_word = sort_words

    return (word_to_id, id_to_word)

def _sentence_to_ids(sentence, word_to_id):
    split_sent = _tokenize(sentence)
    return [ word_to_id[word] for word in split_sent]

def prepare_data(filename, max_word_seq):
    word_to_id, _ = _build_vocab(filename)

    f = open(filename)
    sentences = f.readlines()
    f.close()
    
    data_len = len(sentences)

    data = np.zeros([data_len, max_word_seq], dtype=np.int)
    for i in range(data_len):
        s = sentences[i]
        s_split = _sentence_to_ids(s, word_to_id)
        fill = [0]*(max_word_seq - len(s_split))
        data[i] = s_split + fill

    return len(word_to_id), data

def batch_iterator(data, sents_per_batch, max_word_seq):
    epoch_size = len(data) // sents_per_batch

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    
    fill = np.zeros([sents_per_batch,1], dtype=np.int)
    for i in range(0, epoch_size, sents_per_batch):
        x = data[i:i+sents_per_batch]
        y = np.column_stack((data[i:i+sents_per_batch, 1:], fill))
        yield (x, y)

# The main function is just here for testing
def main():
    vocab_size, data = prepare_data('./titles.txt', 40)

    for i, (x, y) in enumerate(batch_iterator(data, 10, 40)):
        print("Iteration {0}".format(i))
        #print(x[-1])
        #print(y[-1])

    print("Vocab size: {0}".format(vocab_size))

if __name__ == "__main__":
    main()

