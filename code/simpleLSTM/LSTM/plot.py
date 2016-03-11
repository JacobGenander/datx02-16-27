from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import os

def create_plots(path, xs, *args):
    for ys in args:
        plt.plot(xs, ys)

    plt.ylabel('cost')
    plt.xlabel('epochs')
    plt.title('Cost of training and evaluation')
    plt.grid(True)
    save_path = os.path.join(path, "plot")
    plt.savefig(save_path)


