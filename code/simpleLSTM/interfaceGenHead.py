from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

parser = argparse.ArgumentParser(description=
        'Generates sentences from a pretrained LSTM-model')

requiredArgs = parser.add_argument_group('required arguments')
requiredArgs.add_argument('--model_path', nargs=1, required=True,
        help='full path name to the model')


