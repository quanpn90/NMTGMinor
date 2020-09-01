import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from .encdec_attention_func import encdec_attn_func

if hasattr(torch._C, '_jit_set_profiling_executor'):
    torch._C._jit_set_profiling_executor(False)
if hasattr(torch._C, '_jit_set_profiling_mode'):
    torch._C._jit_set_profiling_mode(False)


class EncdecMultiheadAttn(nn.Module):
    """Multi-headed encoder-decoder attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, num_heads, embed_dim, attn_drop=0., n_ensemble=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = attn_drop
        self.head_dim = embed_dim // num_heads
        self.n_ensemble = n_ensemble
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.bias = False
        self.scaling = self.head_dim ** -0.5  # this value is hardcoded in the "fast" implementation

        self.in_proj_weight_q = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.in_proj_weight_kv = Parameter(torch.Tensor(2 * embed_dim, embed_dim))
        self.out_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        self.register_parameter('in_proj_bias_q', None)
        self.register_parameter('in_proj_bias_kv', None)
        self.in_proj_bias_q = None
        self.in_proj_bias_kv = None
        self.out_proj_bias = None

        # batch_ensemble weights
        self.be_r = Parameter(torch.Tensor(embed_dim))
        self.be_s = Parameter(torch.Tensor())

        self.attn_func = encdec_attn_func

        self.reset_parameters()
        try:
            # the fast one requires apex and does not work with incremental so careful
            from apex.contrib.multihead_attn.fast_encdec_multihead_attn_func import fast_encdec_attn_func
            self.attn_func_fast = fast_encdec_attn_func
            self.optimized = 2

        except ModuleNotFoundError as e:
            # print(e)
            # print("Cannot use fast self-attention implementation")
            self.optimized = 2
            self.attn_func_fast = None

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.in_proj_weight_q)
        # in_proj_weight_kv has shape [2 * hidden, hidden] but it should be
        # initialized like a [hidden, hidden] matrix.
        # sqrt(6 / (hidden + hidden)) / sqrt(6 / (2 * hidden + hidden)) = sqrt(1.5)
        # therefore xavier_uniform gain should be set to sqrt(1.5).
        # nn.init.xavier_uniform_(self.in_proj_weight_kv, gain=math.sqrt(1.5))
        # nn.init.xavier_uniform_(self.out_proj_weight)
        std_ = math.sqrt(2.0 / (self.embed_dim + self.embed_dim))
        nn.init.normal_(self.in_proj_weight_q, 0.0, std_)
        nn.init.normal_(self.in_proj_weight_kv, 0.0, std_)
        nn.init.normal_(self.out_proj_weight, 0.0, std_)

        nn.init.normal_(self.be_r, 0.0, std_)
        nn.init.normal_(self.be_s, 0.0, std_)

    def forward(self, query, key, value, attn_mask=None, incremental=False, incremental_cache=None):

        assert value is key, "ERROR: Keys and values must be the same."

        is_training = self.training
        time_masking = False
        len_key = key.size(0)

        if self.optimized == 1 and (self.training and not incremental) and len_key <= 1024 and query.is_cuda:
            if attn_mask is not None:
                if attn_mask.dim() == 3:
                    attn_mask = attn_mask.squeeze(1)
                attn_mask = attn_mask.byte()

            outputs = self.attn_func_fast(time_masking, is_training, self.num_heads, query, key,
                                          self.in_proj_weight_q, self.in_proj_weight_kv, self.out_proj_weight,
                                          attn_mask, self.dropout)

            coverage = None

        # during evaluation we use the python binding which is safer ....
        else:
            outputs, coverage, = self.attn_func(time_masking, is_training,
                                                self.num_heads, query, key,
                                                self.in_proj_weight_q, self.in_proj_weight_kv,
                                                self.out_proj_weight, attn_mask, self.dropout,
                                                incremental, incremental_cache)

        # TODO: add incremental cache

        return outputs, coverage

