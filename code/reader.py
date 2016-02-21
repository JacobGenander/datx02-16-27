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

    # Give the words ids based on the number of occurrences in the data set 
    counter = collections.Counter(data)
    count_pairs = counter.most_common() 
    sort_words = [ word for word, _ in count_pairs ]
    word_to_id = dict(zip(sort_words, range(len(sort_words))))
    id_to_word = sort_words

    return (word_to_id, id_to_word)

def _sentence_to_ids(sentence, word_to_id):
    split_sent = _tokenize(sentence)
    return [ word_to_id[word] for word in split_sent]

def prepare_data(filename):
    word_to_id, _ = _build_vocab(filename)

    f = open(filename)
    sentences = f.readlines()
    f.close()
    
    data_len = len(sentences)
    
    s_split = [ _sentence_to_ids(s, word_to_id) for s in sentences]
    max_word_seq = max([ len(s) for s in s_split])

    data = np.zeros([data_len, max_word_seq], dtype=np.int)
    for i in range(data_len):
        s = sentences[i]
        s_split = _sentence_to_ids(s, word_to_id)
        fill = [0]*(max_word_seq - len(s_split))
        data[i] = s_split + fill

    return data, data_len, len(word_to_id), max_word_seq

def batch_iterator(data, sents_per_batch, max_word_seq):
    epoch_size = len(data) // sents_per_batch
    
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    
    # Shuffle the data to generate different batches each time
    np.random.shuffle(data)
    
    # Generate batches (not foolproof but should work if the data is sane) 
    fill = np.zeros([sents_per_batch,1], dtype=np.int)
    for i in range(0, epoch_size, sents_per_batch):
        x = data[i:i+sents_per_batch]
        # Output is just the input shifted to the right by one
        # We also pad with a column of zeros to keep dimensions
        y = np.column_stack((data[i:i+sents_per_batch, 1:], fill))
        yield (x, y)

# The main function is just here for testing
def main():
    data, data_len, vocab_size, max_word_seq = prepare_data('./titles.txt')
    batch_size = 10

    i = 0
    for x, y in batch_iterator(data, batch_size, max_word_seq):
        i += 1
        if i == 100:
            print('X:')
            print(x[-1])
            print('Y:')
            print(y[-1])
    
    print("Data length: {0}".format(data_len))
    print("Batches: {0}".format(i))
    print("Vocab size: {0}".format(vocab_size))
    print("Longest sentence: {0}".format(max_word_seq))

if __name__ == "__main__":
    main()

