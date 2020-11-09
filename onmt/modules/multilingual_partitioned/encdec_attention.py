import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
import math

from ..optimized.encdec_attention_func import encdec_attn_func


class MPEncdecMultiheadAttn(nn.Module):
    """Multi-headed encoder-decoder attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, num_heads, embed_dim, attn_drop=0., factor_size=8, rank_size=-1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = attn_drop
        self.head_dim = embed_dim // num_heads
        self.factor_size = factor_size
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.bias = False
        self.scaling = self.head_dim ** -0.5  # this value is hardcoded in the "fast" implementation

        if rank_size == -1:
            rank_size = factor_size

        self.rank_size = rank_size

        # factor size is the size of the language factor
        # rank size is to reduce the language factor size to a manageable number of parameters
        self.factor_to_rank = nn.Linear(self.factor_size, self.rank_size)

        self.in_proj_weight_q = Parameter(torch.Tensor(embed_dim * embed_dim, rank_size))
        self.in_proj_weight_kv = Parameter(torch.Tensor(2 * embed_dim * embed_dim, rank_size))
        self.out_proj_weight = Parameter(torch.Tensor(embed_dim * embed_dim, rank_size))

        self.in_proj_bias_q = None
        self.in_proj_bias_kv = None
        self.out_proj_bias = None

        self.attn_func = encdec_attn_func

        try:
            # the fast one requires apex and does not work with incremental so careful
            from ..optimized.encdec_attention_func import fast_encdec_attn_func
            self.attn_func_fast = fast_encdec_attn_func
            self.optimized = 1

        except ModuleNotFoundError as e:
            self.optimized = 2
            self.attn_func_fast = None

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

    def forward(self, query, key, value, src_factor=None, tgt_factor=None, attn_mask=None,
                incremental=False, incremental_cache=None):

        assert value is key, "ERROR: Keys and values must be the same."

        is_training = self.training
        time_masking = False
        len_key = key.size(0)

        # tgt_factor = self.factor_to_rank(tgt_factor)
        # src_factor = self.factor_to_rank(src_factor)

        in_proj_weight_q = torch.mv(self.in_proj_weight_q, tgt_factor).view(self.embed_dim, self.embed_dim)
        in_proj_weight_kv = torch.mv(self.in_proj_weight_kv, src_factor).view(self.embed_dim * 2, self.embed_dim)
        out_proj_weight = torch.mv(self.out_proj_weight, tgt_factor).view(self.embed_dim, self.embed_dim)

        if self.optimized == 1 and (self.training and not incremental) and len_key <= 1024 \
                and query.is_cuda and in_proj_weight_q.dtype == torch.half:
            if attn_mask is not None:
                if attn_mask.dim() == 3:
                    attn_mask = attn_mask.squeeze(1)
                attn_mask = attn_mask.byte()

            outputs = self.attn_func_fast(time_masking, is_training, self.num_heads,
                                          query, key.type_as(in_proj_weight_q),
                                          in_proj_weight_q, in_proj_weight_kv, out_proj_weight,
                                          attn_mask, self.dropout)

            coverage = None

        # during evaluation we use the python binding which is safer ....
        else:
            outputs, coverage, = self.attn_func(time_masking, is_training,
                                                self.num_heads, query, key,
                                                in_proj_weight_q, in_proj_weight_kv,
                                                out_proj_weight, attn_mask, self.dropout,
                                                incremental, incremental_cache)

        # TODO: add incremental cache

        return outputs, coverage

