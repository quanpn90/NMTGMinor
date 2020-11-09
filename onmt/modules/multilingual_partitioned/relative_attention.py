import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
import math

from ..optimized.relative_self_attention_func import relative_self_attn_func


class MPRelativeSelfMultiheadAttn(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., factor_size=8, rank_size=-1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.factor_size = factor_size
        if rank_size == -1:
            rank_size = factor_size

        self.rank_size = rank_size
        self.factor_to_rank = nn.Linear(self.factor_size, self.rank_size)

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.bias = True

        self.in_proj_weight = Parameter(torch.Tensor(embed_dim * 3 * embed_dim, factor_size))
        self.out_proj_weight = Parameter(torch.Tensor(embed_dim * embed_dim, factor_size))
        self.pos_proj_weight = Parameter(torch.Tensor(embed_dim * embed_dim, factor_size))

        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim, factor_size))
        self.out_proj_bias = Parameter(torch.Tensor(embed_dim, factor_size))
        self.pos_proj_bias = Parameter(torch.Tensor(embed_dim, factor_size))

        self.r_w_bias = nn.Parameter(torch.Tensor(self.num_heads * self.head_dim, factor_size))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.num_heads * self.head_dim, factor_size))

        self.reset_parameters()
        self.attn_func = relative_self_attn_func

    def reset_parameters(self, init='normal'):

        if init == 'normal':  # xavier normal
            std_ = math.sqrt(2.0 / (self.embed_dim + self.embed_dim))
            nn.init.normal_(self.in_proj_weight, 0.0, std_)
            nn.init.normal_(self.out_proj_weight, 0.0, std_)
            nn.init.normal_(self.pos_proj_weight, 0.0, std_)
        else:  # xavier uniform
            std_ = math.sqrt(6.0 / (self.embed_dim + self.embed_dim))
            nn.init.uniform_(self.in_proj_weight, -std_, std_)
            nn.init.uniform_(self.out_proj_weight, -std_, std_)
            nn.init.uniform_(self.pos_proj_weight, -std_, std_)

        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj_bias, 0.)
        nn.init.constant_(self.pos_proj_bias, 0.)

        nn.init.normal_(self.r_w_bias, 0.0, 0.02)
        nn.init.normal_(self.r_r_bias, 0.0, 0.02)

    def forward(self, input, pos, factor=None, key_padding_mask=None, attn_mask=None, mems=None,
                incremental=False, incremental_cache=None):

        # factor = self.factor_to_rank(factor)

        embed_dim = self.embed_dim
        in_proj_weight = torch.mv(self.in_proj_weight, factor).view(embed_dim * 3, embed_dim)
        pos_proj_weight = torch.mv(self.pos_proj_weight, factor).view(embed_dim, embed_dim)
        out_proj_weight = torch.mv(self.out_proj_weight, factor).view(embed_dim, embed_dim)

        in_proj_bias = torch.mv(self.in_proj_bias, factor)
        pos_proj_bias = torch.mv(self.pos_proj_bias, factor)
        out_proj_bias = torch.mv(self.out_proj_bias, factor)

        r_w_bias = torch.mv(self.r_w_bias, factor).view(self.num_heads, self.head_dim)
        r_r_bias = torch.mv(self.r_r_bias, factor).view(self.num_heads, self.head_dim)

        if key_padding_mask is not None:
            assert (attn_mask is None), "ERROR attn_mask and key_padding_mask should not be both defined!"
            mask = key_padding_mask
            if len(mask.shape) == 3:
                mask = mask.squeeze(0).transpose(0, 1)
        elif attn_mask is not None:
            mask = attn_mask
            if len(mask.shape) == 3:
                mask = mask.squeeze(-1)
        else:
            mask = None

        is_training = self.training

        outputs, coverage = self.attn_func(input, pos, attn_mask is not None, is_training, self.num_heads,
                                           in_proj_weight, out_proj_weight, pos_proj_weight,
                                           in_proj_bias, out_proj_bias, pos_proj_bias,
                                           r_w_bias, r_r_bias,
                                           mask, self.dropout,
                                           incremental, incremental_cache, False, False)
        # last False is double precision

        return outputs, coverage

