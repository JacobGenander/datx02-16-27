#! /usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import argparse
import cPickle as pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='results/config.p',
        help='path to config.p')

def plot_costs(xs, *args):
    plt.clf()

    for ys in args:
        plt.plot(xs, ys)

    plt.ylabel('cost')
    plt.xlabel('epochs')
    plt.title('Cost of training and evaluation')
    plt.grid(True)

    plt.savefig('costs')
    
def plot_accuracy(xs, ys):
    plt.clf()

    plt.plot(xs, ys)
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.title('Accuracy on validation set')
    plt.grid(True)

    plt.savefig('accuracy')
    
def main():
    args = parser.parse_args()
    
    with open(args.data_path, 'rb') as f:
        conf = pickle.load(f)

    xs = range(conf.start_epoch)
    plot_costs(xs, conf.cost_train, conf.cost_valid)
    plot_accuracy(xs, conf.accuracy)

if __name__ == '__main__':
    main()
