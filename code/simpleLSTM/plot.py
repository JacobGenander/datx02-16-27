from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt

def create_plots(xs, *args):
    for ys in args:
        plt.plot(xs, ys)

    plt.ylabel('cost')
    plt.xlabel('epochs')
    plt.title('Cost of training and evaluation')
    plt.grid(True)
    plt.savefig("foo")


