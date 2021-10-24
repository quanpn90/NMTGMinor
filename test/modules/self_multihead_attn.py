import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from .self_multihead_attn_func import self_attn_func
from .fast_self_multihead_attn_func import fast_self_attn_func
# from .fast_self_multihead_attn_norm_add_func import fast_self_attn_norm_add_func
# from apex.normalization.fused_layer_norm import FusedLayerNorm
# from onmt.modules.layer_norm import LayerNorm

if hasattr(torch._C, '_jit_set_profiling_executor'):
    torch._C._jit_set_profiling_executor(False)
if hasattr(torch._C, '_jit_set_profiling_mode'):
    torch._C._jit_set_profiling_mode(False)


@torch.jit.script
def jit_dropout_add(x, residual, prob, is_training):
    # type: (Tensor, Tensor, float, bool) -> Tensor
    out = F.dropout(x, p=prob, training=True)
    out = residual + out
    return out


class SelfMultiheadAttn(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=False, impl='fast',
                 mask_additive=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.bias = bias
        self.impl = impl
        self.scaling = self.head_dim ** -0.5
        self.mask_additive = mask_additive
        if mask_additive:
            assert impl == 'default' or (
                    impl == 'fast' and bias), "additive mask not supported for fast mode without bias"

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.out_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj_bias = Parameter(torch.Tensor(embed_dim))

        self.reset_parameters()

        if impl == 'fast':
            self.attn_func = fast_self_attn_func
        elif impl == 'default':
            self.attn_func = self_attn_func

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj_bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None, is_training=True):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        input_bias = self.in_proj_bias
        input_weights = self.in_proj_weight

        if key_padding_mask is not None:
            assert (attn_mask is None), "ERROR attn_mask and key_padding_mask should not be both defined!"
            mask = key_padding_mask
        elif attn_mask is not None:
            assert self.mask_additive == False, "additive mask not supported for time mask"
            mask = attn_mask
        else:
            mask = None

        if self.impl == 'fast':
            outputs = self.attn_func(attn_mask is not None, is_training, self.num_heads, query,
                                     input_weights, self.out_proj_weight, input_bias, self.out_proj_bias, mask,
                                     self.mask_additive, self.dropout)
        else:
            outputs = self.attn_func(attn_mask is not None, is_training, self.num_heads, self.scaling, query,
                                     input_weights, self.out_proj_weight,
                                     input_bias, self.out_proj_bias,
                                     mask, self.mask_additive, self.dropout)

        return outputs, None
