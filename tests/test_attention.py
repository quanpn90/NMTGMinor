import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from nmtg.modules.dropout import StaticDropout
from nmtg.modules.linear import XavierLinear, group_linear
from nmtg.modules.masking import MaskedFunction


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, attn_p=0.1, static=True, share=3):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.d = d_model
        self.share = share

        assert d_model % h == 0

        self.d_head = d_model // h
        self.fc_query = MaskedFunction(XavierLinear(d_model, h * self.d_head, bias=False))
        self.fc_key = MaskedFunction(XavierLinear(d_model, h * self.d_head, bias=False))
        self.fc_value = MaskedFunction(XavierLinear(d_model, h * self.d_head, bias=False))

        self.fc_concat = MaskedFunction(XavierLinear(h * self.d_head, d_model, bias=False))

        self.sm = nn.Softmax(dim=-1)

        if static:
            self.attn_dropout = StaticDropout(attn_p)
        else:
            self.attn_dropout = nn.Dropout(attn_p)

    def forward(self, query, key, value, mask, query_mask=None, value_mask=None):

        len_query, b = query.size(0), query.size(1)
        len_key, b_ = key.size(0), key.size(1)

        key_mask = value_mask

        # batch_size*num_heads x len_query x head_dim
        # project inputs to multi-heads
        if self.share == 1:
            shared_qkv = group_linear(
                [self.fc_query.function, self.fc_key.function, self.fc_value.function], query)
            proj_query, proj_key, proj_value = shared_qkv.chunk(3, dim=-1)
        elif self.share == 2:
            proj_query = self.fc_query(query)  # batch_size x len_query x num_heads*head_dim
            shared_kv = group_linear([self.fc_key.function, self.fc_value.function], key)
            proj_key, proj_value = shared_kv.chunk(2, dim=-1)
        else:
            proj_query = self.fc_query(query, mask=query_mask)
            proj_key = self.fc_key(key, mask=key_mask)  # batch_size x len_key x num_heads*head_dim
            proj_value = self.fc_value(value, mask=value_mask)  # batch_size x len_key x num_heads*head_dim

        q, k, v = proj_query, proj_key, proj_value
        # prepare the shape for applying softmax
        q = q.contiguous().view(len_query, b * self.h, self.d_head).transpose(0, 1)
        k = k.contiguous().view(len_key, b * self.h, self.d_head).transpose(0, 1)
        v = v.contiguous().view(len_key, b * self.h, self.d_head).transpose(0, 1)

        q = q * (self.d_head ** -0.5)

        # get dotproduct softmax attns for each head
        attns = torch.bmm(q, k.transpose(1, 2))  # batch_size*num_heads x len_query x len_key

        attns = attns.view(b, self.h, len_query, len_key)
        mask_ = mask.unsqueeze(-3)
        # FP16 support: cast to float and back
        attns = attns.float().masked_fill_(mask_, -float('inf')).type_as(attns)
        attns = F.softmax(attns.float(), dim=-1).type_as(attns)
        # return mean attention from all heads as coverage
        coverage = torch.mean(attns, dim=1)
        attns = self.attn_dropout(attns)
        attns = attns.view(b * self.h, len_query, len_key)

        # apply attns on value
        out = torch.bmm(attns, v)  # batch_size*num_heads x len_query x head_dim
        out = out.transpose(0, 1).contiguous().view(len_query, b, self.d)

        out = self.fc_concat(out)

        return out, coverage


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


import nmtg.modules.attention
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

quan_attention = MultiHeadAttention(8, 512, 0.0, False, 2)

if cuda:
    quan_attention.cuda()
    mask_quan = mask_quan.cuda()

in_a_quan = in_tensor_A
in_b_quan = in_tensor_A if sa else in_tensor_B

out_tensor_quan, _ = quan_attention(in_a_quan, in_b_quan, in_b_quan, mask_quan)
out_tensor_quan.sum().backward()
grads_quan = quan_attention.fc_query.function.weight.grad.clone().detach().cpu()
grads_quan2 = quan_attention.fc_concat.function.weight.grad.clone().detach().cpu()
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

felix_attention = nmtg.modules.attention.MultiHeadAttention(512, 8, 0.0, bf, False)
felix_attention.query_projection.function.weight = quan_attention.fc_query.function.weight
felix_attention.key_projection.function.weight = quan_attention.fc_key.function.weight
felix_attention.value_projection.function.weight = quan_attention.fc_value.function.weight
felix_attention.out_projection.function.weight = quan_attention.fc_concat.function.weight

out_tensor_felix = felix_attention(in_a_felix, in_b_felix, in_b_felix, bias_felix, tgt_mask_felix, src_mask_felix)
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
