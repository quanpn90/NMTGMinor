
import timeit
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from onmt.modules.Bottle import Bottle
from nmtg.modules.masking import MaskedFunction

import sys
cuda = len(sys.argv) > 1 and sys.argv[1] == '--cuda'
cuda_str = '.cuda()' if cuda else ''

nonmasked_init = "lin = nn.Linear(512, 512){cuda}; ten=torch.rand(50, 60, 512){cuda}"
nonmasked_code = "lin(ten)"
if cuda:
    nonmasked_code += "; torch.cuda.synchronize()"


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
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

# Test correctness



masked_init = nonmasked_init + "; mf=MaskedFunction(lin){cuda}; mask=make_mask_seq([50, 60], {0}){cuda}"
masked_code = "mf(ten, mask)"
if cuda:
    masked_code += "; torch.cuda.synchronize()"

print("Testing non-masked...")
out = sum(timeit.Timer(nonmasked_code, nonmasked_init.format(cuda=cuda_str), globals=globals()).repeat(3, 500))
print("{:.2f}ms".format(out * 1000 / 1500))

print("Testing masked...")
for prob in np.linspace(0.0, 1.0, 21):
    print("Testing probability {:.0%}".format(prob))
    out = sum(timeit.Timer(masked_code, masked_init.format(prob, cuda=cuda_str), globals=globals()).repeat(30, 100))
    print("{:.2f}ms".format(out * 1000 / 3000))

