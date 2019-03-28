from __future__ import division

import onmt
import onmt.Markdown
import torch
import argparse
import math
import numpy
import sys

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src',   required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-tgt',
                    help='True target sequence')
parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size')




