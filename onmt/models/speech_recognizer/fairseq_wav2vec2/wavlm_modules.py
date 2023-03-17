# --------------------------------------------------------
# WavLM: Large-Scale Self-Supervised  Pre-training  for Full Stack Speech Processing (https://arxiv.org/abs/2110.13900.pdf)
# Github source: https://github.com/microsoft/unilm/tree/master/wavlm
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------

import math
import warnings
from typing import Dict, Optional, Tuple
import torch
from torch import Tensor, nn
from torch.nn import Parameter
import torch.nn.functional as F

from onmt.modules.optimized.linear import Linear
from onmt.modules.optimized.self_attention_func import self_attn_bias_func


class WavLMMultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            self_attention=False,
            encoder_decoder_attention=False,
            q_noise=0.0,
            qn_block_size=8,
            has_relative_attention_bias=False,
            num_buckets=32,
            max_distance=128,
            gru_rel_pos=False,
            rescale_init=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

        self.head_dim = embed_dim // num_heads
        self.q_head_dim = self.head_dim
        self.k_head_dim = self.head_dim
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        k_bias = True
        if rescale_init:
            k_bias = False

        k_embed_dim = embed_dim
        q_embed_dim = embed_dim

        self.k_proj = Linear(self.kdim, k_embed_dim, bias=k_bias)
        self.v_proj = Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = Linear(embed_dim, q_embed_dim, bias=bias)

        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.gru_rel_pos = gru_rel_pos
        if self.gru_rel_pos:
            self.grep_linear = Linear(self.q_head_dim, 8)
            self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.reset_parameters()

        self.fast_attention = False

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
        if self.has_relative_attention_bias:
            nn.init.xavier_normal_(self.relative_attention_bias.weight)

    def _relative_positions_bucket(self, relative_positions, bidirectional=True):
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = 0

        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_positions > 0).to(torch.long) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = -torch.min(relative_positions, torch.zeros_like(relative_positions))

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_postion_if_large = max_exact + (
                torch.log(relative_positions.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_positions, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(
            relative_position,
            bidirectional=True
        )
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values

    def forward(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            need_weights: bool = True,
            static_kv: bool = False,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
            position_bias: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Input shape: Time x Batch x Channel
        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        if self.has_relative_attention_bias and position_bias is None:
            position_bias = self.compute_bias(tgt_len, src_len)
            position_bias = position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.num_heads, tgt_len, src_len)

        if (
                not is_tpu  # don't use PyTorch version on TPUs
                and incremental_state is None
                and not static_kv
                # A workaround for quantization to work. Otherwise JIT compilation
                # treats bias in linear module as method.
                and not torch.jit.is_scripting()
                and self.q_head_dim == self.head_dim
        ):
            assert key is not None and value is not None
            assert attn_mask is None

            attn_mask_rel_pos = None
            if position_bias is not None:
                attn_mask_rel_pos = position_bias
                if self.gru_rel_pos:

                    # from [T x B x H] to [B x T x H]
                    query_layer = query.transpose(0, 1)

                    # [B x T x head x -1]
                    new_x_shape = query_layer.size()[:-1] + (self.num_heads, -1)

                    # [B x T x head x head_size]
                    query_layer = query_layer.view(*new_x_shape)

                    # [B x H x T x head_size]
                    query_layer = query_layer.permute(0, 2, 1, 3)

                    _B, _H, _L, __ = query_layer.size()

                    gate_input = self.grep_linear(query_layer).view(
                        _B, _H, _L, 2, 4).sum(-1, keepdim=False)

                    # inplace sigmoid
                    gate_a, gate_b = gate_input.sigmoid_().chunk(2, dim=-1)
                    gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                    attn_mask_rel_pos = gate_a_1.view(bsz * self.num_heads, -1, 1) * position_bias

                attn_mask_rel_pos = attn_mask_rel_pos.view((-1, tgt_len, tgt_len))
            else:
                attn_mask_rel_pos = query.new_zeros(*(bsz * self.num_heads, tgt_len, tgt_len))

            # k_proj_bias = self.k_proj.bias
            # if k_proj_bias is None:
            #     k_proj_bias = torch.zeros_like(self.q_proj.bias)

            if self.fast_attention:
                is_training = self.training
                low_precision = True
                in_proj_weight = self.proj_weight
                out_proj_weight = self.out_proj.weight
                recompute = False
                rotary = False
                positions = None

                x, attn = self_attn_bias_func(False, is_training, self.num_heads, query, attn_mask_rel_pos,
                                                   in_proj_weight, out_proj_weight,
                                                   self.proj_bias, self.out_proj.bias,
                                                   key_padding_mask, self.dropout_module.p,
                                                   rotary, positions,
                                                   False, None,       # incremental and state and double precision
                                                   low_precision, True, recompute)  # learnable_pos + return-coverage
            else:

                x, attn = F.multi_head_attention_forward(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    torch.empty([0]),
                    torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                    self.bias_k,
                    self.bias_v,
                    self.add_zero_attn,
                    self.dropout_module.p,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    self.training,
                    # self.training or self.dropout_module.apply_during_inference,
                    key_padding_mask,
                    need_weights,
                    attn_mask_rel_pos,
                    use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj.weight,
                    k_proj_weight=self.k_proj.weight,
                    v_proj_weight=self.v_proj.weight,
                )

            return x, attn, position_bias

        else:
            # this code path is never reached with wavlm
            raise NotImplementedError

    def convert_fast_attention(self):
        pass

        # if self.fast_attention:
        #     return
        # self.fast_attention = True
        # assert self.qkv_same_dim, "Only works with QKV same dim."
        # w_q = self.q_proj.weight.clone()
        # w_k = self.k_proj.weight.clone()
        # w_v = self.v_proj.weight.clone()
        # weights = [w_q, w_k, w_v]
        # weight_ = torch.cat(weights, dim=0).contiguous()
        #
        # b_q = self.q_proj.bias.clone()
        # b_k = self.k_proj.bias.clone()
        # b_v = self.v_proj.bias.clone()
        # biases = [b_q, b_k, b_v]
        # bias_ = torch.cat(biases, dim=0).contiguous()
        #
        # head_dim = self.head_dim
        # heads = self.num_heads
        # input_dim = self.embed_dim
        #
        # # when we concatenate the weights, the output has the size 3 * D (3 -> heads -> head_dim)
        # # the fast attention module requires (heads -> 3 -> head_dim)
        # weight_ = weight_.reshape(3 * head_dim * heads, input_dim).view(3, heads, head_dim, input_dim).transpose(0, 1). \
        #     reshape(-1, input_dim)
        #
        # bias_ = bias_.reshape(3 * head_dim * heads).view(3, heads, head_dim).transpose(0, 1).reshape(-1)
        #
        # weight_t = torch.Tensor(3 * input_dim, input_dim)
        # bias_t = torch.Tensor(3 * input_dim)
        # weight_t.copy_(weight_)
        # bias_t.copy_(bias_)
        # self.proj_weight = Parameter(weight_t)
        # self.proj_bias = Parameter(bias_t)
        #
        # self.proj_weight.requires_grad = self.q_proj.weight.requires_grad
        # self.proj_bias.requires_grad = self.q_proj.bias.requires_grad

        # del self.q_proj, self.k_proj, self.v_proj


def get_activation_fn(activation: str):
    """Returns the activation function corresponding to `activation`"""

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        warnings.warn(
            "--activation-fn=gelu_fast has been renamed to gelu_accurate"
        )
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "glu":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))

def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )

def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)
