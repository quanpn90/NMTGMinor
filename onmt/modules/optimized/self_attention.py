import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from .self_attention_func import self_attn_func
from onmt.constants import double_precision


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1) # dim=-1 triggers a bug in torch < 1.8.0


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class SelfMultiheadAttn(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., rotary_pos_enc=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.bias = True
        self.scaling = self.head_dim ** -0.5
        self.rotary_pos_enc = rotary_pos_enc

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.out_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj_bias = Parameter(torch.Tensor(embed_dim))
        self.reset_parameters()

        self.attn_func = self_attn_func

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.in_proj_weight, gain=math.sqrt(2))
        # nn.init.xavier_uniform_(self.out_proj_weight)
        std_ = math.sqrt(2.0 / (self.embed_dim + self.embed_dim))
        nn.init.normal_(self.in_proj_weight, 0.0, std_)
        nn.init.normal_(self.out_proj_weight, 0.0, std_)

        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj_bias, 0.)

    def forward(self, inputs, pos, key_padding_mask=None, attn_mask=None,
                incremental=False, incremental_cache=None, **kwargs):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        is_training = self.training
        input_weights = self.in_proj_weight
        input_bias = self.in_proj_bias
        bsz, len_q = inputs.size(1), inputs.size(0)
        heads = self.num_heads
        head_dim = self.head_dim
        scale_t = torch.tensor([head_dim ** -0.5])

        # input_lin_results = F.linear(inputs, input_weights, input_bias)
        # input_lin_results = input_lin_results.view(inputs.size(0), inputs.size(1), input_weights.size(0))
        # input_lin_results = input_lin_results.view(inputs.size(0), inputs.size(1) * heads, 3, head_dim)
        #
        # queries = input_lin_results[:, :, 0, :]
        # keys = input_lin_results[:, :, 1, :]
        # values = input_lin_results[:, :, 2, :]
        #
        # if incremental:
        #     keys = keys.contiguous().view(len_q, bsz, heads * head_dim)
        #     values = values.contiguous().view(len_q, bsz, heads * head_dim)
        #     if 'k' in incremental_cache and 'v' in incremental_cache:
        #         keys = torch.cat([incremental_cache['k'], keys], dim=0)  # time first
        #         incremental_cache['k'] = keys
        #         values = torch.cat([incremental_cache['v'], values], dim=0)  # time first
        #         incremental_cache['v'] = values
        #     else:
        #         incremental_cache['k'] = keys
        #         incremental_cache['v'] = values
        #     keys = keys.view(-1, bsz * heads, head_dim)
        #     values = values.view(-1, bsz * heads, head_dim)
        #
        # len_k = keys.size(0)
        #
        # # apply rotary position encodings
        # if self.rotary_pos_enc:
        #     cos, sin = pos
        #     queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)
        #
        # matmul1_results = torch.bmm(queries.transpose(0, 1), keys.transpose(0, 1).transpose(1, 2)).mul(scale_t[0])
        #
        if key_padding_mask is not None:
            assert (attn_mask is None), "ERROR attn_mask and key_padding_mask should not be both defined!"
            mask = key_padding_mask
            if len(mask.shape) == 3:
                mask = mask.squeeze(1)
        #
        #     batches, seql_q, seql_k = matmul1_results.size()
        #     seqs = int(batches / heads)
        #     matmul1_results = matmul1_results.view(seqs, heads, seql_q, seql_k)
        #     mask = mask.to(torch.bool)
        #     matmul1_results = matmul1_results.masked_fill_(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        #     matmul1_results = matmul1_results.view(seqs * heads, seql_q, seql_k)
        #
        elif attn_mask is not None:
            mask = attn_mask
            if len(mask.shape) == 3:
                mask = mask.squeeze(0)
            mask = mask.to(torch.bool)
        #     matmul1_results.masked_fill_(mask, float('-inf'))
        #
        # softmax_results = F.softmax(matmul1_results, dim=-1)
        # dropout_results = F.dropout(softmax_results, self.dropout, training=self.training)
        #
        # matmul2_results = torch.bmm(dropout_results, values.transpose(0, 1))
        #
        # matmul2_results = matmul2_results.transpose(0, 1).contiguous().view(inputs.size(0), inputs.size(1),
        #                                                                     inputs.size(2))
        #
        # outputs = F.linear(matmul2_results, self.out_proj_weight, self.out_proj_bias)
        #
        # coverage = dropout_results

        outputs, coverage = self.attn_func(attn_mask is not None, is_training, self.num_heads, inputs,
                                           input_weights, self.out_proj_weight,
                                           input_bias, self.out_proj_bias,
                                           mask, self.dropout,
                                           self.rotary_pos_enc, pos,
                                           incremental, incremental_cache, True)

        return outputs, coverage

