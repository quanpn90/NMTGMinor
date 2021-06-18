import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
import math

from ..optimized.encdec_attention_func import encdec_attn_func


class MFWEncdecMultiheadAttn(nn.Module):
    """Multi-headed encoder-decoder attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, num_heads, embed_dim, attn_drop=0.,
                 n_languages=1, rank=1,
                 use_multiplicative=False, no_bias=False,
                 weight_drop=0.0, mfw_activation="none"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = attn_drop
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.bias = False
        self.scaling = self.head_dim ** -0.5  # this value is hardcoded in the "fast" implementation
        self.use_multiplicative = use_multiplicative
        self.weight_drop = weight_drop
        self.no_bias = no_bias

        assert (not self.no_bias) or self.use_multiplicative

        self.in_proj_weight_q = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.in_proj_weight_kv = Parameter(torch.Tensor(2 * embed_dim, embed_dim))
        self.out_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        if not self.no_bias:
            self.r_q = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.s_q = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.r_kv = torch.nn.Parameter(torch.Tensor(n_languages, rank, 2 * embed_dim))
            self.s_kv = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.r_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.s_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))

        if use_multiplicative:
            self.ml_rank = 1
            self.rm_q = torch.nn.Parameter(torch.Tensor(n_languages, self.ml_rank, embed_dim))
            self.sm_q = torch.nn.Parameter(torch.Tensor(n_languages, self.ml_rank, embed_dim))
            self.rm_kv = torch.nn.Parameter(torch.Tensor(n_languages, self.ml_rank, 2 * embed_dim))
            self.sm_kv = torch.nn.Parameter(torch.Tensor(n_languages, self.ml_rank, embed_dim))
            self.rm_o = torch.nn.Parameter(torch.Tensor(n_languages, self.ml_rank, embed_dim))
            self.sm_o = torch.nn.Parameter(torch.Tensor(n_languages, self.ml_rank, embed_dim))

        self.in_proj_bias_q = None
        self.in_proj_bias_kv = None
        self.out_proj_bias = None

        self.attn_func = encdec_attn_func
        self.mfw_activation = mfw_activation.lower()

        self.reset_parameters()

    def reset_parameters(self, init='normal'):
        if init == 'normal':  # xavier normal
            std_ = math.sqrt(2.0 / (self.embed_dim + self.embed_dim))
            nn.init.normal_(self.in_proj_weight_q, 0.0, std_)
            nn.init.normal_(self.in_proj_weight_kv, 0.0, std_)
            nn.init.normal_(self.out_proj_weight, 0.0, std_)
        else:  # xavier uniform
            std_ = math.sqrt(6.0 / (self.embed_dim + self.embed_dim))
            nn.init.uniform_(self.in_proj_weight_q, -std_, std_)
            nn.init.uniform_(self.in_proj_weight_kv, -std_, std_)
            nn.init.uniform_(self.out_proj_weight, -std_, std_)

        if not self.no_bias:
            nn.init.normal_(self.r_q, 0.0, 0.02)
            nn.init.normal_(self.s_q, 0.0, 0.02)
            nn.init.normal_(self.r_kv, 0.0, 0.02)
            nn.init.normal_(self.s_kv, 0.0, 0.02)
            nn.init.normal_(self.r_o, 0.0, 0.02)
            nn.init.normal_(self.s_o, 0.0, 0.02)

        if self.use_multiplicative:
            nn.init.constant_(self.rm_q, 1.0)
            nn.init.constant_(self.sm_q, 1.0)
            nn.init.constant_(self.rm_kv, 1.0)
            nn.init.constant_(self.sm_kv, 1.0)
            nn.init.constant_(self.rm_o, 1.0)
            nn.init.constant_(self.sm_o, 1.0)

    def freeze(self):

        if not self.no_bias:
            self.r_q.requires_grad = False
            self.s_q.requires_grad = False
            self.r_kv.requires_grad = False
            self.s_kv.requires_grad = False
            self.r_o.requires_grad = False
            self.s_o.requires_grad = False

        if self.use_multiplicative:
            self.rm_q.requires_grad = False
            self.sm_q.requires_grad = False
            self.rm_kv.requires_grad = False
            self.sm_kv.requires_grad = False
            self.rm_o.requires_grad = False
            self.sm_o.requires_grad = False

    def unfreeze(self):

        if not self.no_bias:
            self.r_q.requires_grad = True
            self.s_q.requires_grad = True
            self.r_kv.requires_grad = True
            self.s_kv.requires_grad = True
            self.r_o.requires_grad = True
            self.s_o.requires_grad = True

        if self.use_multiplicative:
            self.rm_q.requires_grad = True
            self.sm_q.requires_grad = True
            self.rm_kv.requires_grad = True
            self.sm_kv.requires_grad = True
            self.rm_o.requires_grad = True
            self.sm_o.requires_grad = True

    def forward(self, query, key, value, src_indices=None, tgt_indices=None, attn_mask=None,
                incremental=False, incremental_cache=None, factorize=True, **kwargs):

        indices = tgt_indices
        bsz = query.size(1)

        assert value is key, "ERROR: Keys and values must be the same."

        is_training = self.training
        time_masking = False
        recompute = False

        # dropping the main weights during training
        in_proj_weight_q = self.in_proj_weight_q
        in_proj_weight_kv = self.in_proj_weight_kv
        out_proj_weight = self.out_proj_weight

        if factorize:
            in_proj_weight_q = F.dropout(self.in_proj_weight_q, p=self.weight_drop, training=self.training)
            in_proj_weight_kv = F.dropout(self.in_proj_weight_kv, p=self.weight_drop, training=self.training)
            out_proj_weight = F.dropout(self.out_proj_weight, p=self.weight_drop, training=self.training)

            if self.use_multiplicative:
                # multiply main weights with extra weights
                rm_q = torch.index_select(self.rm_q, 0, indices).squeeze(0)
                sm_q = torch.index_select(self.sm_q, 0, src_indices).squeeze(0)
                rm_kv = torch.index_select(self.rm_kv, 0, indices).squeeze(0)
                sm_kv = torch.index_select(self.sm_kv, 0, src_indices).squeeze(0)
                rm_o = torch.index_select(self.rm_o, 0, indices).squeeze(0)
                sm_o = torch.index_select(self.sm_o, 0, src_indices).squeeze(0)

                in_proj_weight_q = in_proj_weight_q * torch.bmm(rm_q.unsqueeze(-1), sm_q.unsqueeze(1)).sum(dim=0)
                in_proj_weight_kv = in_proj_weight_kv * torch.bmm(rm_kv.unsqueeze(-1), sm_kv.unsqueeze(1)).sum(dim=0)
                out_proj_weight = out_proj_weight * torch.bmm(rm_o.unsqueeze(-1), sm_o.unsqueeze(1)).sum(dim=0)

            # adding main weights with extra weights
            # sum(dim=0) sums over the rank dimension
            if not self.no_bias:
                if indices.size(0) == 1 and len(indices.shape) == 1:
                    r_q = torch.index_select(self.r_q, 0, indices).squeeze(0)
                    s_q = torch.index_select(self.s_q, 0, src_indices).squeeze(0)
                    r_kv = torch.index_select(self.r_kv, 0, indices).squeeze(0)
                    s_kv = torch.index_select(self.s_kv, 0, src_indices).squeeze(0)
                    r_o = torch.index_select(self.r_o, 0, indices).squeeze(0)
                    s_o = torch.index_select(self.s_o, 0, src_indices).squeeze(0)
                else:
                    print(indices.size(), input.size())
                    raise NotImplementedError

                in_proj_weight_q = in_proj_weight_q + torch.bmm(r_q.unsqueeze(-1), s_q.unsqueeze(1)).sum(dim=0)
                in_proj_weight_kv = in_proj_weight_kv + torch.bmm(r_kv.unsqueeze(-1), s_kv.unsqueeze(1)).sum(dim=0)
                out_proj_weight = out_proj_weight + torch.bmm(r_o.unsqueeze(-1), s_o.unsqueeze(1)).sum(dim=0)

            if self.mfw_activation == "none":
                in_proj_weight_q = in_proj_weight_q
            elif self.mfw_activation == "gelu":
                in_proj_weight_q = F.gelu(in_proj_weight_q)
                in_proj_weight_kv = F.gelu(in_proj_weight_kv)
                out_proj_weight = F.gelu(out_proj_weight)
            elif self.mfw_activation == "silu":
                in_proj_weight_q = F.silu(in_proj_weight_q)
                in_proj_weight_kv = F.silu(in_proj_weight_kv)
                out_proj_weight = F.silu(out_proj_weight)
            else:
                raise NotImplementedError

        outputs, coverage, = self.attn_func(recompute, is_training,
                                            self.num_heads, query, key,
                                            in_proj_weight_q, in_proj_weight_kv,
                                            out_proj_weight, attn_mask, self.dropout,
                                            incremental, incremental_cache, False, True)

        # TODO: add incremental cache

        return outputs, coverage



