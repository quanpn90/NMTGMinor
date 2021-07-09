import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from .self_attention_func import self_attn_func
from onmt.constants import double_precision


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

    def forward(self, input, pos, key_padding_mask=None, attn_mask=None,
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

        if key_padding_mask is not None:
            assert (attn_mask is None), "ERROR attn_mask and key_padding_mask should not be both defined!"
            mask = key_padding_mask
            if len(mask.shape) == 3:
                mask = mask.squeeze(1)
        elif attn_mask is not None:
            mask = attn_mask
            if len(mask.shape) == 3:
                mask = mask.squeeze(0)
        else:
            mask = None

        outputs, coverage = self.attn_func(attn_mask is not None, is_training, self.num_heads, input,
                                           input_weights, self.out_proj_weight,
                                           input_bias, self.out_proj_bias,
                                           mask, self.dropout,
                                           self.rotary_pos_enc, pos,
                                           incremental, incremental_cache, True)

        return outputs, coverage

