from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

parser = argparse.ArgumentParser(description=
    'Trains an LSTM-network with parameters in hyperParams.py\n')

program_dir = os.path.dirname(os.path.abspath(__file__))
default_data_dir = os.path.join(program_dir, "data")
default_save_dir = os.path.join(program_dir, "results")

parser.add_argument('--data_path', nargs='?', default=default_data_dir,
        help='path where train.txt, valid.txt and test.txt can be found')
parser.add_argument('--save_path', nargs='?', default=default_save_dir,
        help='directory where the results are saved')
