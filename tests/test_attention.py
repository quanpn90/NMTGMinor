import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import nmtg.models
from nmtg.modules.attention import MultiHeadAttention
from nmtg.modules.linear import XavierLinear, group_linear
from nmtg.modules.masking import MaskedFunction

import onmt.Constants
from onmt.modules.Transformer.Layers import MultiHeadAttention as QuanAttention


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
parser.add_argument('--bf', action='store_true')  # batch first
parser.add_argument('--sa', action='store_true')  # self-attention
parser.add_argument('--nm', action='store_true')  # no mask
args = parser.parse_args()
cuda = args.cuda
bf = args.bf
sa = args.sa
nm = args.nm

in_tensor_A = torch.zeros((60, 50, 512))
in_tensor_B = torch.zeros((55, 50, 512))
nn.init.xavier_uniform_(in_tensor_A)
nn.init.xavier_uniform_(in_tensor_B)
in_tensor_A_T = in_tensor_A.transpose(0, 1).contiguous()
in_tensor_B_T = in_tensor_B.transpose(0, 1).contiguous()

if cuda:
    in_tensor_A = in_tensor_A.cuda()
    in_tensor_B = in_tensor_B.cuda()
    in_tensor_A_T = in_tensor_A_T.cuda()
    in_tensor_B_T = in_tensor_B_T.cuda()

future_mask_quan = torch.ByteTensor(np.triu(np.ones((60, 60)), k=1).astype('uint8'))
src_mask_quan = make_mask_seq((50, 55), .9).eq_(0)
tgt_mask_quan = make_mask_seq((50, 60), .9).eq_(0)

if nm:
    mask_quan = torch.zeros(1, 60, 60).byte()
elif sa:
    mask_quan = (tgt_mask_quan.unsqueeze(1) + future_mask_quan).gt_(0)
else:
    mask_quan = src_mask_quan.unsqueeze(1)

onmt.Constants.weight_norm = False
onmt.Constants.attention_out = 'default'

quan_attention = QuanAttention(8, 512, 0.0, False, 2)

if cuda:
    quan_attention.cuda()
    mask_quan = mask_quan.cuda()

in_a_quan = in_tensor_A
in_b_quan = in_tensor_A if sa else in_tensor_B

out_tensor_quan, _ = quan_attention(in_a_quan, in_b_quan, in_b_quan, mask_quan)
out_tensor_quan.sum().backward()
grads_quan = quan_attention.fc_query.function.linear.weight.grad.clone().detach().cpu()
grads_quan2 = quan_attention.fc_concat.function.linear.weight.grad.clone().detach().cpu()
quan_attention.zero_grad()


future_mask_felix = torch.ByteTensor(np.tril(np.ones((60, 60)), k=0).astype('uint8'))
src_mask_felix = src_mask_quan.eq(0)
tgt_mask_felix = tgt_mask_quan.eq(0)

if cuda:
    future_mask_felix = future_mask_felix.cuda()
    src_mask_felix = src_mask_felix.cuda()
    tgt_mask_felix = tgt_mask_felix.cuda()

in_tensor_A_Felix = in_tensor_A_T if bf else in_tensor_A
in_tensor_B_Felix = in_tensor_B_T if bf else in_tensor_B

in_a_felix = in_tensor_A_Felix
in_b_felix = in_tensor_A_Felix if sa else in_tensor_B_Felix

if nm:
    bias_felix = None
elif sa:
    mask_felix = (tgt_mask_felix.unsqueeze(1) + future_mask_felix).gt_(1)
    bias_felix = in_b_felix.new_full(mask_felix.size(), float('-inf')).masked_fill(mask_felix, 0)
else:
    bias_felix = in_a_felix.new_full(src_mask_felix.size(), float('-inf')).masked_fill(src_mask_felix, 0).unsqueeze(1)

felix_attention = MultiHeadAttention(512, 8, 0.0, bf, False)
felix_attention.query_projection.function.weight = quan_attention.fc_query.function.linear.weight
felix_attention.key_projection.function.weight = quan_attention.fc_key.function.linear.weight
felix_attention.value_projection.function.weight = quan_attention.fc_value.function.linear.weight
felix_attention.out_projection.function.weight = quan_attention.fc_concat.function.linear.weight

out_tensor_felix, _ = felix_attention(in_a_felix, in_b_felix, in_b_felix, bias_felix, tgt_mask_felix, src_mask_felix)
if bf:
    out_tensor_felix = out_tensor_felix.transpose(0, 1).contiguous()
out_tensor_felix.sum().backward()
grads_felix = felix_attention.query_projection.function.weight.grad.clone().detach().cpu()
grads_felix2 = felix_attention.out_projection.function.weight.grad.clone().detach().cpu()
felix_attention.zero_grad()

if not torch.allclose(out_tensor_felix, out_tensor_quan):
    print(out_tensor_felix[0, :5, :10])
    print(out_tensor_quan[0, :5, :10])
else:
    print("Tensors match")
if not torch.allclose(grads_quan, grads_felix):
    print(grads_quan)
    print(grads_felix)
else:
    print("Gradients match")
if not torch.allclose(grads_quan2, grads_felix2):
    print(grads_quan2)
    print(grads_felix2)
else:
    print("Gradients match")

repeats = (5, 10)
quan_command = 'quan_attention(in_a_quan, in_b_quan, in_b_quan, mask_quan)'
felix_command = 'felix_attention(in_a_felix, in_b_felix, in_b_felix, bias_felix, tgt_mask_felix, src_mask_felix)'
if cuda:
    repeats = (10, 100)
    torch.cuda.synchronize()
    quan_command += '; torch.cuda.synchronize()'
    felix_command += '; torch.cuda.synchronize()'

# Then test speed
import timeit
time = min(timeit.Timer(quan_command, globals=globals()).repeat(*repeats))
print("Quan: {:.3f}ms".format(time * 1000 / repeats[1]))
time = min(timeit.Timer(felix_command, globals=globals()).repeat(*repeats))
print("Felix: {:.3f}ms".format(time * 1000 / repeats[1]))
