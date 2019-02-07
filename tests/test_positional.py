import math
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

from nmtg.modules.positional_encoding import SinusoidalPositionalEncoding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, p=0, len_max=512):
        # save a fixed positional embedding matrix up to len_max,
        # so that no need to recreate it everytime
        super(PositionalEncoding, self).__init__()
        self.len_max = len_max
        self.d_model = d_model
        self.data_type = None

        self.renew(len_max)

        self.p = p

    def renew(self, new_max_len):
        if hasattr(self, 'pos_emb'):
            del self.pos_emb
        position = torch.arange(0, new_max_len).float()

        num_timescales = self.d_model // 2
        log_timescale_increment = math.log(10000) / (num_timescales - 1)
        inv_timescales = torch.exp(torch.arange(0, num_timescales).float() * -log_timescale_increment)
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        pos_emb = torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), 1)

        if self.data_type is not None:
            pos_emb.type(self.data_type)
        # wrap in a buffer so that model can be moved to GPU
        self.register_buffer('pos_emb', pos_emb)
        self.data_type = self.pos_emb.type()
        self.len_max = new_max_len

    def forward(self, word_emb, t=None):

        len_seq = t if t else word_emb.size(1)

        if len_seq > self.len_max:
            self.renew(len_seq)

        if word_emb.size(1) == len_seq:
            out = word_emb + Variable(self.pos_emb[:len_seq, :], requires_grad=False)
        else:
            # out = word_emb + Variable(self.pos_emb[:len_seq, :][-1, :], requires_grad=False)
            time_emb = Variable(self.pos_emb[len_seq - 1, :], requires_grad=False)  # 1 x dim
            # out should have size bs x 1 x dim
            out = word_emb + time_emb.unsqueeze(0).repeat(word_emb.size(0), 1, 1)
        return out


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

masked_indices = torch.nonzero(mask.view(-1)).squeeze(1)
felix_output2 = felix_encoding(inputs, mask).view(-1).index_select(0, masked_indices)
quan_output2 = quan_output.view(-1).index_select(0, masked_indices)

if not torch.allclose(felix_output2, quan_output2):
    print(felix_output2[0, :5, :10])
    print(quan_output2[0, :5, :10])
else:
    print("Tensors match")

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
