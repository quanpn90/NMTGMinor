import math
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

from nmtg.models.transformer.transformer import Transformer
from nmtg.modules.positional_encoding import SinusoidalPositionalEncoding
from onmt.modules.Transformer.Layers import PositionalEncoding

def make_mask_random(size, fill):
    total = int(np.prod(size))
    ones = int(fill * total)
    mask = torch.cat([torch.ones(ones), torch.zeros(total - ones)]).byte()
    return mask[torch.randperm(total)].view(*size)


def make_mask_seq(size, fill):
    maxlen = size[1]
    avg_len = int(fill * maxlen)
    lens = torch.randint(avg_len - 1, avg_len + 2, (size[0],))
    return sequence_mask(lens, maxlen)


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()
cuda = args.cuda

quan_encoding = PositionalEncoding(512, len_max=512)
felix_encoding = SinusoidalPositionalEncoding(512, True, 512)

inputs = torch.zeros(50, 60, 512)
mask = make_mask_seq((50, 60), .9).eq_(0)

if cuda:
    quan_encoding.cuda()
    felix_encoding.cuda()
    inputs = inputs.cuda()
    mask = mask.cuda()

# correctness
quan_output = quan_encoding(inputs)
felix_output = felix_encoding(inputs)

if not torch.allclose(felix_output, quan_output):
    print(felix_output[0, :5, :10])
    print(quan_output[0, :5, :10])
else:
    print("Tensors match")

# masked_indices = torch.nonzero(mask.view(-1)).squeeze(1)
# felix_output2 = felix_encoding(inputs, mask).view(-1).index_select(0, masked_indices)
# quan_output2 = quan_output.view(-1).index_select(0, masked_indices)
#
# if not torch.allclose(felix_output2, quan_output2):
#     print(felix_output2[0, :5, :10])
#     print(quan_output2[0, :5, :10])
# else:
#     print("Tensors match")

# speed
repeats = (5, 10)
quan_command = 'quan_encoding(inputs)'
felix_command = 'felix_encoding(inputs)'
if cuda:
    repeats = (10, 100)
    torch.cuda.synchronize()
    quan_command += '; torch.cuda.synchronize()'
    felix_command += '; torch.cuda.synchronize()'

import timeit
time = min(timeit.Timer(quan_command, globals=globals()).repeat(*repeats))
print("Quan: {:.3f}ms".format(time * 1000 / repeats[1]))
time = min(timeit.Timer(felix_command, globals=globals()).repeat(*repeats))
print("Felix: {:.3f}ms".format(time * 1000 / repeats[1]))
