from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

parser = argparse.ArgumentParser(description=
        'Generates sentences from a pretrained LSTM-model')

parser.add_argument('model_path',
        help='full pathname to the model')
parser.add_argument('-n', metavar='N', type=int,
        help='number of sentences to generate (might be capped by batch size)')

