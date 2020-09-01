import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from ..optimized.   relative_self_attention_func import relative_self_attn_func

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


class AdaptiveRelativeAttn(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, model_size, num_heads, factor_size, dropout=0., adaptive_type='shared'):
        super().__init__()
        self.model_size = model_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = model_size // num_heads
        self.factor_size = factor_size
        self.adaptive_type = adaptive_type
        assert self.head_dim * num_heads == self.model_size, "model_size must be divisible by num_heads"
        self.bias = True

        self.in_proj_weight = Parameter(torch.Tensor(3 * model_size, model_size, factor_size))
        self.out_proj_weight = Parameter(torch.Tensor(model_size, model_size, factor_size))
        self.pos_proj_weight = Parameter(torch.Tensor(model_size, model_size, factor_size))

        self.in_proj_bias = Parameter(torch.Tensor(3*model_size, factor_size))
        self.out_proj_bias = Parameter(torch.Tensor(model_size, factor_size))
        self.pos_proj_bias = Parameter(torch.Tensor(model_size, factor_size))

        self.r_w_bias = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim, factor_size))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim, factor_size))

        self.factor_map = nn.Linear(self.model_size, self.factor_size)

        self.reset_parameters()
        self.attn_func = relative_self_attn_func

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.in_proj_weight, gain=math.sqrt(2))
        # nn.init.xavier_uniform_(self.out_proj_weight)
        std_ = math.sqrt(2.0 / (self.model_size + self.model_size))
        nn.init.normal_(self.in_proj_weight, 0.0, std_)
        nn.init.normal_(self.out_proj_weight, 0.0, std_)
        nn.init.normal_(self.pos_proj_weight, 0.0, std_)

        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj_bias, 0.)
        nn.init.constant_(self.pos_proj_bias, 0.)

        nn.init.normal_(self.r_w_bias, 0.0, std_)
        nn.init.normal_(self.r_r_bias, 0.0, std_)

    def forward(self, input, pos, factor, key_padding_mask=None, attn_mask=None, mems=None,
                incremental=False, incremental_cache=None):

        factor = self.factor_map(factor).squeeze()

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

        in_proj_weight = torch.mv(self.in_proj_weight.view(-1, self.factor_size), factor) \
            .view(self.in_proj_weight.size(0), self.in_proj_weight.size(1))
        out_proj_weight = torch.mv(self.out_proj_weight.view(-1, self.factor_size), factor) \
            .view(self.out_proj_weight.size(0), self.out_proj_weight.size(1))
        pos_proj_weight = torch.mv(self.pos_proj_weight.view(-1, self.factor_size), factor) \
            .view(self.out_proj_weight.size(0), self.out_proj_weight.size(1))

        in_proj_bias = torch.mv(self.in_proj_bias, factor)
        out_proj_bias = torch.mv(self.out_proj_bias, factor)
        pos_proj_bias = torch.mv(self.pos_proj_bias, factor)

        r_w_bias = torch.mv(self.r_w_bias.view(-1, self.factor_size), factor) \
            .view(self.r_w_bias.size(0), self.r_w_bias.size(1))
        r_r_bias = torch.mv(self.r_r_bias.view(-1, self.factor_size), factor) \
            .view(self.r_r_bias.size(0), self.r_r_bias.size(1))

        is_training = self.training

        outputs, coverage = self.attn_func(input, pos, attn_mask is not None, is_training, self.num_heads,
                                           in_proj_weight, out_proj_weight, pos_proj_weight,
                                           in_proj_bias, out_proj_bias, pos_proj_bias,
                                           r_w_bias, r_r_bias,
                                           mask, self.dropout,
                                           incremental, incremental_cache, False, False)
        # last False is double precision

        return outputs, coverage
