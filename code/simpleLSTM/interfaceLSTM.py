from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

parser = argparse.ArgumentParser(description=
    'Trains an LSTM-network with parameters in hyperParams.py\n')

default_dir = os.path.dirname(os.path.abspath(__file__))
parser.add_argument('--data_path', nargs='?', default=default_dir,
        help='path where train.txt, valid.txt and test.txt can be found')
parser.add_argument('--save_path', nargs='?', default=default_dir,
        help='directory where the results are saved')
