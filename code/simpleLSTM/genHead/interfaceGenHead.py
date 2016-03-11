from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

parser = argparse.ArgumentParser(description=
        'Generates sentences from a pretrained LSTM-model')

parser.add_argument('model_folder',
        help='folder containg the results from LSTM.py')
parser.add_argument('-n', metavar='N', type=int, default=10,
        help='number of sentences to generate (might be capped by batch size)')
parser.add_argument('--most_prob', action='store_true', default=False,
        help='generate headlines with the most probable words')

