import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from .relative_self_attention_func import relative_self_attn_func


class RelativeSelfMultiheadAttn(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., learnable_pos=False, max_pos=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.bias = True
        self.learnable_pos = learnable_pos

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.out_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        if self.learnable_pos:
            # If using learnable position embeddings, then assign embeddings for 2N - 1 max positions
            # (embeddings are shared across heads)
            assert max_pos >= 1
            self.pos_emb = nn.Embedding(2 * max_pos + 1, self.embed_dim)
            self.pos_proj_weight, self.pos_proj_bias = None, None
        else:
            # Using sin/cos position encodings which are linearly projected to head_dim (seperately per head)
            self.pos_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.pos_proj_bias = Parameter(torch.Tensor(embed_dim))

        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj_bias = Parameter(torch.Tensor(embed_dim))

        self.r_w_bias = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))

        self.reset_parameters()
        self.attn_func = relative_self_attn_func

    def reset_parameters(self, init='normal'):
        if init == 'normal':  # xavier normal
            std_ = math.sqrt(2.0 / (self.embed_dim + self.embed_dim))
            nn.init.normal_(self.in_proj_weight, 0.0, std_)
            nn.init.normal_(self.out_proj_weight, 0.0, std_)
            if self.pos_proj_weight is not None:
                nn.init.normal_(self.pos_proj_weight, 0.0, std_)

        else:
            std_ = math.sqrt(6.0 / (self.embed_dim + self.embed_dim))
            nn.init.uniform_(self.in_proj_weight, -std_, std_)
            nn.init.uniform_(self.out_proj_weight, -std_, std_)
            if self.pos_proj_weight is not None:
                nn.init.uniform_(self.pos_proj_weight, -std_, std_)

        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj_bias, 0.)
        if self.pos_proj_bias is not None:
            nn.init.constant_(self.pos_proj_bias, 0.)

        nn.init.normal_(self.r_w_bias, 0.0, 0.02)
        nn.init.normal_(self.r_r_bias, 0.0, 0.02)

    def forward(self, input, pos, key_padding_mask=None, attn_mask=None, mems=None,
                incremental=False, incremental_cache=None):
        """
        :param input: [T x B x H]
        :param pos: [T x 1 x H] or [T x T x H]
        :param key_padding_mask: [1 x T x B]
        :param attn_mask: [T x T]
        :param mems:
        :param incremental:
        :param incremental_cache:
        :return:
        """

        if key_padding_mask is not None:
            assert (attn_mask is None), "ERROR attn_mask and key_padding_mask should not be both defined!"
            mask = key_padding_mask
            if len(mask.shape) == 3:
                # [1 x T x B] -> [B x T]
                mask = mask.squeeze(0).transpose(0, 1)
        elif attn_mask is not None:
            mask = attn_mask
            if len(mask.shape) == 3:
                mask = mask.squeeze(-1)
        else:
            mask = None

        is_training = self.training

        if self.learnable_pos:
            # [len_q x len_k] -> [len_q x len_k x head_dim]
            # [len_r x 1] -> [len_r x 1 x model_size]
            pos = self.pos_emb(pos)

        outputs, coverage = self.attn_func(input, pos, attn_mask is not None, is_training, self.num_heads,
                                           self.in_proj_weight, self.out_proj_weight, self.pos_proj_weight,
                                           self.in_proj_bias, self.out_proj_bias, self.pos_proj_bias,
                                           self.r_w_bias, self.r_r_bias,
                                           mask, self.dropout,
                                           incremental, incremental_cache, False,
                                           self.learnable_pos, True)
        # last Falses are double precision, learnable_embedding and return coverage

        return outputs, coverage
