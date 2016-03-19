#! /usr/bin/python2

import argparse
from os import system
from random import shuffle

parser = argparse.ArgumentParser(description=
    'Generate sets from data file\n')

parser.add_argument('data_file', nargs='?',
        help='file to generate sets from')
parser.add_argument('--min_length', nargs='?', default=4,
        help='remove sentences with fewer words than this')
parser.add_argument('--eval_portion', nargs='?', default=0.1,
        help='validation and test are each going to be this fraction of the whole data set')

def create_file(name, data_set):
    with open(name, "w") as f:
        for s in data_set:
            f.write(s)

def main():
    args = parser.parse_args()
    with open(args.data_file, "r") as f:
        headlines = f.read()

    # Replace som punctuation charaters with new lines
    headlines = headlines.replace("!", "\n").replace("?", "\n")
    headlines = headlines.replace(",", "\n").replace(";", "\n").replace(":", "\n")
    headlines = headlines.splitlines(True)

    # Remove sentences with fewer words than min_length
    proc_heads = []
    for h in headlines:
        if len(h.split()) >= args.min_length:
            proc_heads.append(h)

    data_len = len(proc_heads)
    eval_len = int(data_len * args.eval_portion)

    shuffle(proc_heads)

    # Pick sets from data array
    val_set = proc_heads[:eval_len]
    test_set = proc_heads[eval_len:eval_len*2]
    train_set = proc_heads[eval_len*2:]

    # Save sets to files
    create_file("valid.txt", val_set)
    create_file("test.txt", test_set)
    create_file("train.txt", train_set)

    # Reomve non-ascii characters
    system("perl -i -pe 's/[^[:ascii:]]//g' valid.txt")
    system("perl -i -pe 's/[^[:ascii:]]//g' test.txt")
    system("perl -i -pe 's/[^[:ascii:]]//g' train.txt")

if __name__ == "__main__":
    main()
