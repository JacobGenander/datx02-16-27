from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import os

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

