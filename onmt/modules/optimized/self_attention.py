import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from .self_attention_func import self_attn_func
from onmt.constants import double_precision


# from .fast_self_multihead_attn_func          import fast_self_attn_func
# from .fast_self_multihead_attn_norm_add_func import fast_self_attn_norm_add_func
# from apex.normalization.fused_layer_norm     import FusedLayerNorm


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

    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.bias = True
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.out_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj_bias = Parameter(torch.Tensor(embed_dim))
        self.reset_parameters()

        self.attn_func = self_attn_func
        self.optimized = 1

        try:
            # the fast one requires apex and does not work with incremental so careful
            from apex.contrib.multihead_attn.fast_self_multihead_attn_func import fast_self_attn_func
            self.attn_func_fast = fast_self_attn_func
            self.optimized = 1
        except ModuleNotFoundError as e:
            # print(e)
            # print("Cannot use fast self-attention implementation")
            self.optimized = 2
            self.attn_func_fast = None

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.in_proj_weight, gain=math.sqrt(2))
        # nn.init.xavier_uniform_(self.out_proj_weight)
        std_ = math.sqrt(2.0 / (self.embed_dim + self.embed_dim))
        nn.init.normal_(self.in_proj_weight, 0.0, std_)
        nn.init.normal_(self.out_proj_weight, 0.0, std_)

        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj_bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None,
                incremental=False, incremental_cache=None):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        is_training = self.training

        assert key is value
        assert key is query

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

        if self.optimized == 1 and self.training:
            if mask is not None:
                mask = mask.byte()
            outputs = self.attn_func_fast(attn_mask is not None, is_training, self.num_heads, query,
                                          input_weights, self.out_proj_weight, input_bias, self.out_proj_bias, mask,
                                          False, self.dropout)
            coverage = None
        elif self.optimized == 2:
            outputs, coverage = self.attn_func(attn_mask is not None, is_training, self.num_heads, query,
                                               input_weights, self.out_proj_weight,
                                               input_bias, self.out_proj_bias,
                                               mask, self.dropout,
                                               incremental, incremental_cache)
        else:
            outputs, coverage = self.forward_autograd(attn_mask is not None, is_training, self.num_heads, query,
                                                      input_weights, self.out_proj_weight,
                                                      input_bias, self.out_proj_bias,
                                                      mask, self.dropout,
                                                      incremental, incremental_cache)

        return outputs, coverage

    def forward_autograd(self, use_time_mask, training, heads, inputs, input_weights, output_weights,
                         input_biases, output_biases, mask, dropout,
                         incremental, incremental_cache):

        bsz, len_q = inputs.size(1), inputs.size(0)
        head_dim = inputs.size(-1) // heads
        scale_t = torch.tensor([head_dim ** -0.5])

        input_lin_results = torch.addmm(input_biases,
                                        inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2)),
                                        input_weights.transpose(0, 1),
                                        beta=1., alpha=1.)

        input_lin_results = input_lin_results.view(inputs.size(0), inputs.size(1), input_weights.size(0))

        input_lin_results = input_lin_results.view(inputs.size(0), inputs.size(1) * heads, 3, head_dim)
        queries = input_lin_results[:, :, 0, :]
        keys = input_lin_results[:, :, 1, :]
        values = input_lin_results[:, :, 2, :]

        if incremental:
            keys = keys.contiguous().view(len_q, bsz, heads * head_dim)
            values = values.contiguous().view(len_q, bsz, heads * head_dim)
            if 'k' in incremental_cache and 'v' in incremental_cache:
                keys = torch.cat([incremental_cache['k'], keys], dim=0)  # time first
                incremental_cache['k'] = keys
                values = torch.cat([incremental_cache['v'], values], dim=0)  # time first
                incremental_cache['v'] = values
            else:
                incremental_cache['k'] = keys
                incremental_cache['v'] = values
            keys = keys.view(-1, bsz * heads, head_dim)
            values = values.view(-1, bsz * heads, head_dim)

        len_k = keys.size(0)
        scaled_bsz = queries.size(1)
        len_q = queries.size(0)

        qk_scores = queries.new_empty((scaled_bsz, len_q, len_k), requires_grad=True)
        qk_scores = torch.baddbmm(qk_scores, queries.transpose(0, 1), keys.transpose(0, 1).transpose(1, 2),
                                  beta=0.0, alpha=scale_t[0])

        if mask is not None:
            # Self Attention Time Mask
            if use_time_mask:
                assert (len(mask.size()) == 2), "Timing mask is not 2D!"
                # assert (mask.size(0) == mask.size(1)), "Sequence length should match!"
                mask = mask.to(torch.bool)
                qk_scores = qk_scores.masked_fill_(mask, float('-inf'))
            # Key Padding Mask
            else:
                batches, seql_q, seql_k = qk_scores.size()
                seqs = int(batches / heads)
                qk_scores = qk_scores.view(seqs, heads, seql_q, seql_k)
                mask = mask.to(torch.bool)
                qk_scores = qk_scores.masked_fill_(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
                qk_scores = qk_scores.view(seqs * heads, seql_q, seql_k)

        dtype_ = torch.float64 if double_precision else torch.float32
        attn_scores = F.softmax(qk_scores, dim=-1, dtype=dtype_).type_as(qk_scores)

        attn_scores = F.dropout(attn_scores, dropout, training=training)

        # matmul2_results = attn_scores.new_empty((attn_scores.size(1), attn_scores.size(0), attn_scores.size(2)),
        #                                         requires_grad=True).transpose(1, 0)
        matmul2_results = torch.bmm(attn_scores, values.transpose(0, 1))
        matmul2_results = matmul2_results.transpose(0, 1).contiguous().view(len_q, bsz, inputs.size(2))

        outputs = torch.addmm(output_biases,
                              matmul2_results.view(len_q * bsz, inputs.size(2)),
                              output_weights.transpose(0, 1),
                              beta=1., alpha=1.)

        outputs = outputs.view(len_q, bsz, output_weights.size(0))

        return outputs, attn_scores
