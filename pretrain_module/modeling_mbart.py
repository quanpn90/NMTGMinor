# coding=utf-8
# Copyright 2021, The Facebook AI Research Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch MBART model. """
import copy
import math
import random
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
import numpy as np

from torch.nn import CrossEntropyLoss, MSELoss
from onmt.modules.layer_norm import LayerNorm
from onmt.modules.optimized.self_attention_func import self_attn_func
from onmt.modules.optimized.encdec_attention_func_bias import encdec_attn_bias_func
from onmt.modules.dropout import embedded_dropout
from onmt.modules.optimized.dropout_add import fused_dropout_add
from onmt.modules.optimized.linear import linear_function
from torch.cuda.amp import custom_fwd, custom_bwd

from .activations import ACT2FN
from .modeling_outputs import (
    BaseModelOutput,
)
from .modeling_utils import PreTrainedModel
# from ...utils import logging
# from .configuration_bart import BartConfig
import onmt
from collections import defaultdict
from .configuration_mbart import MBartConfig

_CHECKPOINT_FOR_DOC = "facebook/mbart-large-cc25"
_CONFIG_FOR_DOC = "MBartConfig"
_TOKENIZER_FOR_DOC = "MBartTokenizer"

MBART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/mbart-large-cc25",
    # See all MBART models at https://huggingface.co/models?filter=mbart
]


class IndexCopy(torch.autograd.Function):
    """
    This function is kinda similar to rnn pad_packed_sequence
    It remaps nonpadded values for a (N-1)-d tensor into a (N)-d tensor
    """
    @staticmethod
    @custom_fwd
    def forward(ctx, input, non_pad_indices, total_batch_size):
        """
        :param ctx:
        :param input: 2D [bsz x ... ] bsz is the total number of elements after unpadding
        :param non_pad_indices: bsz * seq_len
        :param total_batch_size: (int) bsz * seq_len (before unpadding) > bsz
        :return:
        In the forward pass we create a new zero tensor and copy the inputs into it based on non_pad_indices
        """
        sizes = list(input.size())
        sizes[0] = total_batch_size

        output = input.new_zeros(*sizes)
        output.index_copy_(0, non_pad_indices, input)
        ctx.save_for_backward(non_pad_indices)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grads):
        """
        :param ctx:
        :param output_grads:
        :return:
        In the backward pass we simply
        """
        non_pad_indices, = ctx.saved_tensors

        grad_input = output_grads.index_select(0, non_pad_indices)

        return grad_input, None, None


index_copy = IndexCopy.apply


# Copied from transformers.models.bart.modeling_bart.BartLearnedPositionalEmbedding with Bart->MBart
class MBartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # MBart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions + self.offset)


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->MBart
class MBartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.fast_attention = False

        self.is_factorized = False
        self.multiplicative_factorize = False
        self.fast_factorize = False

        from onmt.modules.optimized.fast_mha import fast_bert_mha
        self.fast_bert_mha = fast_bert_mha

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def add_factorized_weights(self, n_languages, rank=4, multiplicative=False, fast=False,
                               sub_factors=0, sub_factor_rank=-1):
        """
        Add factorized weights for self-attention

        :param sub_factor_rank:
        :param n_languages:
        :param rank:
        :param multiplicative:
        :param fast:
        :param sub_factors:
        :return:
        """
        embed_dim = self.embed_dim
        self.is_factorized = True
        self.multiplicative_factorize = multiplicative
        self.fast_factorize = fast
        self.sub_factorized = sub_factors > 0

        self.r_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, 3 * embed_dim))
        self.s_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
        self.r_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
        self.s_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))

        std = 0.01 if fast else 0.02
        nn.init.normal_(self.r_i, 0.0, std)
        nn.init.normal_(self.s_i, 0.0, std)
        nn.init.normal_(self.r_o, 0.0, std)
        nn.init.normal_(self.s_o, 0.0, std)

        if multiplicative:
            rank = rank if fast else 1
            self.rm_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, 3 * embed_dim))
            self.sm_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.rm_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.sm_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))

            constant = math.sqrt(1.0 / rank) if fast else 1
            nn.init.constant_(self.rm_i, constant)
            nn.init.constant_(self.sm_i, constant)
            nn.init.constant_(self.rm_o, constant)
            nn.init.constant_(self.sm_o, constant)

        if self.sub_factorized:
            self.sub_r_i = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, 3 * embed_dim))
            self.sub_s_i = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, embed_dim))
            self.sub_r_o = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, embed_dim))
            self.sub_s_o = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, embed_dim))

            std = 0.01 if fast else 0.02
            nn.init.normal_(self.sub_r_i, 0.0, std)
            nn.init.normal_(self.sub_s_i, 0.0, std)
            nn.init.normal_(self.sub_r_o, 0.0, std)
            nn.init.normal_(self.sub_s_o, 0.0, std)

            if multiplicative:
                sub_factor_rank = sub_factor_rank if fast else 1
                self.sub_rm_i = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, 3 * embed_dim))
                self.sub_sm_i = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, embed_dim))
                self.sub_rm_o = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, embed_dim))
                self.sub_sm_o = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, embed_dim))

                constant = math.sqrt(1.0 / sub_factor_rank) if fast else 1
                nn.init.constant_(self.sub_rm_i, constant)
                nn.init.constant_(self.sub_sm_i, constant)
                nn.init.constant_(self.sub_rm_o, constant)
                nn.init.constant_(self.sub_sm_o, constant)

    def convert_fast_attention(self):

        if self.fast_attention:
            return

        # print("[INFO] Convert MBartAttention from slow to fast")
        self.fast_attention = True
        w_q = self.q_proj.weight.clone()
        w_k = self.k_proj.weight.clone()
        w_v = self.v_proj.weight.clone()
        weights = [w_q, w_k, w_v]
        weight_ = torch.cat(weights, dim=0).contiguous()

        b_q = self.q_proj.bias.clone()
        b_k = self.k_proj.bias.clone()
        b_v = self.v_proj.bias.clone()
        biases = [b_q, b_k, b_v]
        bias_ = torch.cat(biases, dim=0).contiguous()

        head_dim = self.head_dim
        heads = self.num_heads
        input_dim = self.embed_dim

        weight_ = weight_.reshape(3 * head_dim * heads, input_dim).view(3, heads, head_dim, input_dim).transpose(0, 1). \
            reshape(-1, input_dim)

        bias_ = bias_.reshape(3 * head_dim * heads).view(3, heads, head_dim).transpose(0, 1).reshape(-1)

        weight_t = torch.Tensor(3 * input_dim, input_dim)
        bias_t = torch.Tensor(3 * input_dim)
        weight_t.copy_(weight_)
        bias_t.copy_(bias_)
        self.proj_weight = Parameter(weight_t)
        self.proj_bias = Parameter(bias_t)

        self.proj_weight.requires_grad = self.q_proj.weight.requires_grad
        self.proj_bias.requires_grad = self.q_proj.bias.requires_grad
        del self.q_proj, self.k_proj, self.v_proj

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            cu_seqlens=None, max_len=None,
            lang=None, atb=None,
            incremental=False, incremental_cache=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        if not self.fast_attention:
            raise NotImplementedError("Slow attention by HuggingFace is deprecated.")

        else:

            in_proj_weight = self.proj_weight
            out_proj_weight = self.out_proj.weight

            if self.is_factorized:
                if self.multiplicative_factorize:
                    rm_i = torch.index_select(self.rm_i, 0, lang).squeeze(0)  # squeeze possible because only 1
                    sm_i = torch.index_select(self.sm_i, 0, lang).squeeze(0)
                    rm_o = torch.index_select(self.rm_o, 0, lang).squeeze(0)
                    sm_o = torch.index_select(self.sm_o, 0, lang).squeeze(0)

                    if self.fast_factorize:
                        mul_factor_in = torch.mm(rm_i.t(), sm_i)
                        mul_factor_out = torch.mm(rm_o.t(), sm_o)
                    else:
                        mul_factor_in = torch.bmm(rm_i.unsqueeze(-1), sm_i.unsqueeze(1)).sum(dim=0)
                        mul_factor_out = torch.bmm(rm_o.unsqueeze(-1), sm_o.unsqueeze(1)).sum(dim=0)

                    if self.sub_factorized and atb is not None:
                        # squeeze possible because only 1
                        rm_i = torch.index_select(self.sub_rm_i, 0, atb).squeeze(0)
                        sm_i = torch.index_select(self.sub_sm_i, 0, atb).squeeze(0)
                        rm_o = torch.index_select(self.sub_rm_o, 0, atb).squeeze(0)
                        sm_o = torch.index_select(self.sub_sm_o, 0, atb).squeeze(0)

                        if self.fast_factorize:
                            sub_mul_factor_in = torch.mm(rm_i.t(), sm_i)
                            sub_mul_factor_out = torch.mm(rm_o.t(), sm_o)
                        else:
                            sub_mul_factor_in = torch.bmm(rm_i.unsqueeze(-1), sm_i.unsqueeze(1)).sum(dim=0)
                            sub_mul_factor_out = torch.bmm(rm_o.unsqueeze(-1), sm_o.unsqueeze(1)).sum(dim=0)

                        mul_factor_in.mul_(sub_mul_factor_in)
                        mul_factor_out.mul_(sub_mul_factor_out)

                    # Has to be multiplicative here
                    in_proj_weight = in_proj_weight * mul_factor_in
                    out_proj_weight = out_proj_weight * mul_factor_out

                r_i = torch.index_select(self.r_i, 0, lang).squeeze(0)
                s_i = torch.index_select(self.s_i, 0, lang).squeeze(0)
                r_o = torch.index_select(self.r_o, 0, lang).squeeze(0)
                s_o = torch.index_select(self.s_o, 0, lang).squeeze(0)

                if self.fast_factorize:
                    add_factor_in = torch.mm(r_i.t(), s_i)
                    add_factor_out = torch.mm(r_o.t(), s_o)
                else:
                    add_factor_in = torch.bmm(r_i.unsqueeze(-1), s_i.unsqueeze(1)).sum(dim=0)
                    add_factor_out = torch.bmm(r_o.unsqueeze(-1), s_o.unsqueeze(1)).sum(dim=0)

                if self.sub_factorized and atb is not None:

                    r_i = torch.index_select(self.sub_r_i, 0, atb).squeeze(0)
                    s_i = torch.index_select(self.sub_s_i, 0, atb).squeeze(0)
                    r_o = torch.index_select(self.sub_r_o, 0, atb).squeeze(0)
                    s_o = torch.index_select(self.sub_s_o, 0, atb).squeeze(0)

                    if self.fast_factorize:
                        sub_add_factor_in = torch.mm(r_i.t(), s_i)
                        sub_add_factor_out = torch.mm(r_o.t(), s_o)
                    else:
                        sub_add_factor_in = torch.bmm(r_i.unsqueeze(-1), s_i.unsqueeze(1)).sum(dim=0)
                        sub_add_factor_out = torch.bmm(r_o.unsqueeze(-1), s_o.unsqueeze(1)).sum(dim=0)

                    # Has to be additive here
                    add_factor_in.add_(sub_add_factor_in)
                    add_factor_out.add_(sub_add_factor_out)

                in_proj_weight = in_proj_weight + add_factor_in
                out_proj_weight = out_proj_weight + add_factor_out

            if hidden_states.ndim == 3:

                use_time_mask = self.is_decoder
                qlen, klen = hidden_states.size(0), hidden_states.size(0)
                mask = attention_mask
                low_precision = True  # Use CUDA impl

                # print("USING FAST ATTENTION - DECODER=" + str(self.is_decoder))
                attn_output, coverage = self_attn_func(use_time_mask, self.training, self.num_heads, hidden_states,
                                                       in_proj_weight, out_proj_weight,
                                                       self.proj_bias, self.out_proj.bias,
                                                       mask, self.dropout,
                                                       False, None,
                                                       incremental, incremental_cache, low_precision, True)

                attn_output = attn_output

            else:
                assert self.fast_bert_mha is not None
                assert hidden_states.dtype == torch.half
                assert cu_seqlens is not None
                assert max_len is not None  # and max_len <= 512
                assert self.is_decoder is False # only encoder
                sm = torch.cuda.get_device_capability()

                # Only Ampere supported at the moment
                assert (sm[0] == 8 and (sm[1] in [0]) and max_len <= 512)
                total_bsz = hidden_states.size(0)
                qkv = linear_function(hidden_states, in_proj_weight, self.proj_bias)  # B x H
                # B x 3 x H x d

                # TODO: moving to CUDA to remove overhead?
                qkv = qkv.view(total_bsz, self.num_heads, 3, self.head_dim).transpose(1, 2).contiguous()

                context, coverage = self.fast_bert_mha(qkv, cu_seqlens, self.dropout, max_len, self.training)

                context = context.view(-1, self.num_heads * self.head_dim).contiguous()
                outputs = linear_function(context, out_proj_weight, self.out_proj.bias)

                attn_output = outputs

            return attn_output, coverage, incremental_cache



class MBartCrossAttention(MBartAttention):

    def convert_fast_attention(self):

        if self.fast_attention:
            return

        # print("[INFO] Convert MBartCrossAttention from slow to fast")
        self.fast_attention = True

        # Merge weight KV into one
        w_k = self.k_proj.weight.clone()
        w_v = self.v_proj.weight.clone()
        weights = [w_k, w_v]
        weight_ = torch.cat(weights, dim=0).contiguous()

        # b_q = self.q_proj.bias.clone()
        b_k = self.k_proj.bias.clone()
        b_v = self.v_proj.bias.clone()
        biases = [b_k, b_v]
        bias_ = torch.cat(biases, dim=0).contiguous()

        head_dim = self.head_dim
        heads = self.num_heads
        input_dim = self.embed_dim

        weight_ = weight_.reshape(2 * head_dim * heads, input_dim).view(2, heads, head_dim, input_dim).transpose(0, 1). \
            reshape(-1, input_dim)

        bias_ = bias_.reshape(2 * head_dim * heads).view(2, heads, head_dim).transpose(0, 1).reshape(-1)

        weight_t = torch.Tensor(2 * input_dim, input_dim)
        bias_t = torch.Tensor(2 * input_dim)
        weight_t.copy_(weight_)
        bias_t.copy_(bias_)
        self.proj_weight_kv = Parameter(weight_t)
        self.proj_bias_kv = Parameter(bias_t)

        self.proj_weight_kv.requires_grad = self.k_proj.weight.requires_grad
        self.proj_bias_kv.requires_grad = self.k_proj.bias.requires_grad

        del self.k_proj
        del self.v_proj

        # # Convert Weight_Q
        # w_q = self.q_proj.weight.clone()
        # weight_ = w_q
        #
        # b_q = self.q_proj.bias.clone()
        # bias_ = b_q
        #
        # head_dim = self.head_dim
        # heads = self.num_heads
        # input_dim = self.embed_dim
        #
        # weight_ = weight_.reshape(head_dim * heads, input_dim).view(1, heads, head_dim, input_dim).transpose(0, 1). \
        #     reshape(-1, input_dim)
        #
        # bias_ = bias_.reshape(head_dim * heads).view(1, heads, head_dim).transpose(0, 1).reshape(-1)
        #
        # weight_t = torch.Tensor(input_dim, input_dim)
        # bias_t = torch.Tensor(input_dim)
        # weight_t.copy_(weight_)
        # bias_t.copy_(bias_)
        # self.proj_weight_q = Parameter(weight_t)
        # self.proj_bias_q = Parameter(bias_t)
        # del self.q_proj

    def add_factorized_weights(self, n_languages, rank=4, multiplicative=False, fast=False,
                               sub_factors=0, sub_factor_rank=-1):

        embed_dim = self.embed_dim
        self.is_factorized = True
        self.multiplicative_factorize = multiplicative
        self.fast_factorize = fast
        self.sub_factorized = (sub_factors > 0)

        self.r_q = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
        self.s_q = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
        self.r_kv = torch.nn.Parameter(torch.Tensor(n_languages, rank, 2 * embed_dim))
        self.s_kv = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
        self.r_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
        self.s_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))

        std = 0.01 if fast else 0.02
        nn.init.normal_(self.r_q, 0.0, std)
        nn.init.normal_(self.s_q, 0.0, std)
        nn.init.normal_(self.r_kv, 0.0, std)
        nn.init.normal_(self.s_kv, 0.0, std)
        nn.init.normal_(self.r_o, 0.0, std)
        nn.init.normal_(self.s_o, 0.0, std)

        if multiplicative:
            rank = rank if fast else 1
            self.rm_q = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.sm_q = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.rm_kv = torch.nn.Parameter(torch.Tensor(n_languages, rank, 2 * embed_dim))
            self.sm_kv = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.rm_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.sm_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))

            constant = math.sqrt(1.0 / rank) if fast else 1
            nn.init.constant_(self.rm_q, constant)
            nn.init.constant_(self.sm_q, constant)
            nn.init.constant_(self.rm_kv, constant)
            nn.init.constant_(self.sm_kv, constant)
            nn.init.constant_(self.rm_o, constant)
            nn.init.constant_(self.sm_o, constant)

        if self.sub_factorized:
            self.sub_r_q = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, embed_dim))
            self.sub_s_q = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, embed_dim))
            self.sub_r_kv = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, 2 * embed_dim))
            self.sub_s_kv = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, embed_dim))
            self.sub_r_o = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, embed_dim))
            self.sub_s_o = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, embed_dim))

            std = 0.01 if fast else 0.02
            nn.init.normal_(self.sub_r_q, 0.0, std)
            nn.init.normal_(self.sub_s_q, 0.0, std)
            nn.init.normal_(self.sub_r_kv, 0.0, std)
            nn.init.normal_(self.sub_s_kv, 0.0, std)
            nn.init.normal_(self.sub_r_o, 0.0, std)
            nn.init.normal_(self.sub_s_o, 0.0, std)

            if multiplicative:
                sub_factor_rank = sub_factor_rank if fast else 1
                self.sub_rm_q = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, embed_dim))
                self.sub_sm_q = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, embed_dim))
                self.sub_rm_kv = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, 2 * embed_dim))
                self.sub_sm_kv = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, embed_dim))
                self.sub_rm_o = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, embed_dim))
                self.sub_sm_o = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, embed_dim))

                constant = math.sqrt(1.0 / sub_factor_rank) if fast else 1
                nn.init.constant_(self.sub_rm_q, constant)
                nn.init.constant_(self.sub_sm_q, constant)
                nn.init.constant_(self.sub_rm_kv, constant)
                nn.init.constant_(self.sub_sm_kv, constant)
                nn.init.constant_(self.sub_rm_o, constant)
                nn.init.constant_(self.sub_sm_o, constant)

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            lang=None, atb=None, checkpointing=False,
            incremental=False, incremental_cache=None, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        assert key_value_states is not None
        if not self.fast_attention:
            raise NotImplementedError("Slow Attention by HuggingFace not supported anymore")

        else:
            in_proj_weight_q = self.q_proj.weight
            in_proj_weight_kv = self.proj_weight_kv
            out_proj_weight = self.out_proj.weight

            if self.is_factorized:
                if self.multiplicative_factorize:
                    rm_q = torch.index_select(self.rm_q, 0, lang).squeeze(0)  # squeeze possible because only 1
                    sm_q = torch.index_select(self.sm_q, 0, lang).squeeze(0)
                    rm_kv = torch.index_select(self.rm_kv, 0, lang).squeeze(0)  # squeeze possible because only 1
                    sm_kv = torch.index_select(self.sm_kv, 0, lang).squeeze(0)
                    rm_o = torch.index_select(self.rm_o, 0, lang).squeeze(0)
                    sm_o = torch.index_select(self.sm_o, 0, lang).squeeze(0)

                    if self.fast_factorize:
                        mul_factor_q = torch.mm(rm_q.t(), sm_q)
                        mul_factor_kv = torch.mm(rm_kv.t(), sm_kv)
                        mul_factor_out = torch.mm(rm_o.t(), sm_o)
                    else:
                        mul_factor_q = torch.bmm(rm_q.unsqueeze(-1), sm_q.unsqueeze(1)).sum(dim=0)
                        mul_factor_kv = torch.bmm(rm_kv.unsqueeze(-1), sm_kv.unsqueeze(1)).sum(dim=0)
                        mul_factor_out = torch.bmm(rm_o.unsqueeze(-1), sm_o.unsqueeze(1)).sum(dim=0)

                    if self.sub_factorized and atb is not None:
                        # squeeze possible because only 1
                        rm_q = torch.index_select(self.sub_rm_q, 0, atb).squeeze(0)  # squeeze possible because only 1
                        sm_q = torch.index_select(self.sub_sm_q, 0, atb).squeeze(0)
                        rm_kv = torch.index_select(self.sub_rm_kv, 0, atb).squeeze(0)  # squeeze possible because only 1
                        sm_kv = torch.index_select(self.sub_sm_kv, 0, atb).squeeze(0)
                        rm_o = torch.index_select(self.sub_rm_o, 0, atb).squeeze(0)
                        sm_o = torch.index_select(self.sub_sm_o, 0, atb).squeeze(0)

                        if self.fast_factorize:
                            sub_mul_factor_q = torch.mm(rm_q.t(), sm_q)
                            sub_mul_factor_kv = torch.mm(rm_kv.t(), sm_kv)
                            sub_mul_factor_out = torch.mm(rm_o.t(), sm_o)
                        else:
                            sub_mul_factor_q = torch.bmm(rm_q.unsqueeze(-1), sm_q.unsqueeze(1)).sum(dim=0)
                            sub_mul_factor_kv = torch.bmm(rm_kv.unsqueeze(-1), sm_kv.unsqueeze(1)).sum(dim=0)
                            sub_mul_factor_out = torch.bmm(rm_o.unsqueeze(-1), sm_o.unsqueeze(1)).sum(dim=0)

                        # Has to be multiplicative (or additive???) here
                        mul_factor_q = mul_factor_q.mul_(sub_mul_factor_q)
                        mul_factor_kv = mul_factor_kv.mul_(sub_mul_factor_kv)
                        mul_factor_out = mul_factor_out.mul_(sub_mul_factor_out)

                    in_proj_weight_q = in_proj_weight_q * mul_factor_q
                    in_proj_weight_kv = in_proj_weight_kv * mul_factor_kv
                    out_proj_weight = out_proj_weight * mul_factor_out

                r_q = torch.index_select(self.r_q, 0, lang).squeeze(0)
                s_q = torch.index_select(self.s_q, 0, lang).squeeze(0)
                r_kv = torch.index_select(self.r_kv, 0, lang).squeeze(0)
                s_kv = torch.index_select(self.s_kv, 0, lang).squeeze(0)
                r_o = torch.index_select(self.r_o, 0, lang).squeeze(0)
                s_o = torch.index_select(self.s_o, 0, lang).squeeze(0)

                if self.fast_factorize:
                    add_factor_q = torch.mm(r_q.t(), s_q)
                    add_factor_kv = torch.mm(r_kv.t(), s_kv)
                    add_factor_out = torch.mm(r_o.t(), s_o)
                else:
                    add_factor_q = torch.bmm(r_q.unsqueeze(-1), s_q.unsqueeze(1)).sum(dim=0)
                    add_factor_kv = torch.bmm(r_kv.unsqueeze(-1), s_kv.unsqueeze(1)).sum(dim=0)
                    add_factor_out = torch.bmm(r_o.unsqueeze(-1), s_o.unsqueeze(1)).sum(dim=0)

                if self.sub_factorized and atb is not None:
                    r_q = torch.index_select(self.sub_r_q, 0, atb).squeeze(0)
                    s_q = torch.index_select(self.sub_s_q, 0, atb).squeeze(0)
                    r_kv = torch.index_select(self.sub_r_kv, 0, atb).squeeze(0)
                    s_kv = torch.index_select(self.sub_s_kv, 0, atb).squeeze(0)
                    r_o = torch.index_select(self.sub_r_o, 0, atb).squeeze(0)
                    s_o = torch.index_select(self.sub_s_o, 0, atb).squeeze(0)

                    if self.fast_factorize:
                        sub_add_factor_q = torch.mm(r_q.t(), s_q)
                        sub_add_factor_kv = torch.mm(r_kv.t(), s_kv)
                        sub_add_factor_out = torch.mm(r_o.t(), s_o)
                    else:
                        sub_add_factor_q = torch.bmm(r_q.unsqueeze(-1), s_q.unsqueeze(1)).sum(dim=0)
                        sub_add_factor_kv = torch.bmm(r_kv.unsqueeze(-1), s_kv.unsqueeze(1)).sum(dim=0)
                        sub_add_factor_out = torch.bmm(r_o.unsqueeze(-1), s_o.unsqueeze(1)).sum(dim=0)

                    add_factor_q = add_factor_q.add_(sub_add_factor_q)
                    add_factor_kv = add_factor_kv.add_(sub_add_factor_kv)
                    add_factor_out = add_factor_out.add_(sub_add_factor_out)

                # Has to be additive here
                in_proj_weight_q = in_proj_weight_q + add_factor_q
                in_proj_weight_kv = in_proj_weight_kv + add_factor_kv
                out_proj_weight = out_proj_weight + add_factor_out

            recompute = checkpointing
            key_value_states = key_value_states

            # TODO: Add factorize

            # attention_mask should have size Bxlen_k
            low_precision = True

            attn_output, coverage = encdec_attn_bias_func(recompute, self.training, self.num_heads,
                                                          hidden_states, key_value_states,
                                                          in_proj_weight_q, in_proj_weight_kv, out_proj_weight,
                                                          self.q_proj.bias, self.proj_bias_kv, self.out_proj.bias,
                                                          attention_mask, self.dropout,
                                                          incremental, incremental_cache,
                                                          False, None, None,   # no rotary encodings
                                                          low_precision, True)

        return attn_output, coverage, incremental_cache


class MBartCrossAttentionSlow(MBartAttention):

    def convert_fast_attention(self):

        pass

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            lang=None, atb=None,
            incremental=False, incremental_cache=None, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        # is_cross_attention = key_value_states is not None
        assert key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling

        if incremental and ('c_k' in incremental_cache and 'c_v' in incremental_cache):
            # these are stored
            key_states = incremental_cache['c_k']
            value_states = incremental_cache['c_v']
        else:
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)
            if incremental:
                incremental_cache['c_k'] = key_states
                incremental_cache['c_v'] = value_states

        # reshape into B x H x T x D ?
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, incremental_cache


class MBartAutoRegressiveSelfAttentionSLow(MBartAttention):

    def convert_fast_attention(self):

        pass

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            incremental=False, incremental_cache=None, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        # is_cross_attention = key_value_states is not None
        assert key_value_states is None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if incremental:
            if 'k' in incremental_cache and 'v' in incremental_cache:
                key_states = torch.cat([incremental_cache['k'], key_states], dim=1)  # time first
                value_states = torch.cat([incremental_cache['v'], value_states], dim=1)  # time first

                incremental_cache['k'] = key_states
                incremental_cache['v'] = value_states
            else:
                incremental_cache['k'] = key_states
                incremental_cache['v'] = value_states

        # reshape into B x H x T x D ?
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, incremental_cache


class MBartEncoderLayer(nn.Module):
    def __init__(self, config: MBartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = MBartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.activation_dropout

        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

        self.activation_fn_name = config.activation_function
        self.fused = False
        self.fused_function = None
        if self.activation_fn_name == 'relu':
            from onmt.modules.mlp.mlp import mlp_relu_function
            if mlp_relu_function is not None:
                self.fused_function = mlp_relu_function
                self.fused = True
        elif self.activation_fn_name == 'gelu':
            from onmt.modules.mlp.mlp import mlp_gelu_function
            if mlp_gelu_function is not None:
                self.fused_function = mlp_gelu_function
                self.fused = True

        from onmt.modules.optimized.fast_mha import fast_bert_mha
        self.fast_bert_mha = fast_bert_mha

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            output_attentions: bool = False,
            max_len=-1, cu_seqlens=None,
            checkpointing_ffn=False
    ):
        """
        :param checkpointing_ffn:
        :param output_attentions: Whether or not to return the attentions tensors of all attention layers.
        :param attention_mask:  `(batch, src_len)`
        :param hidden_states:  `(seq_len, batch, embed_dim)`
        :param cu_seqlens:
        :param max_len:
        """
        residual = hidden_states

        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            cu_seqlens=cu_seqlens,
            max_len=max_len
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.fused and hidden_states.is_cuda:
            weights = [self.fc1.weight, self.fc2.weight]
            biases = [self.fc1.bias, self.fc2.bias]

            dropout = self.activation_dropout if self.training else 0.0
            hidden_states = self.fused_function(dropout, checkpointing_ffn, hidden_states, *weights, *biases)
        else:
            hidden_states = self.activation_fn(self.fc1(hidden_states))
            hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
            hidden_states = self.fc2(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = hidden_states + residual

        if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class MBartDecoderLayer(nn.Module):
    def __init__(self, config: MBartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = MBartAttention(  #MBartAutoRegressiveSelfAttentionSLow(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = MBartCrossAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout
        )

        self.activation_fn_name = config.activation_function
        self.fused = False
        self.fused_function = None
        if self.activation_fn_name == 'relu':
            from onmt.modules.mlp.mlp import mlp_relu_function
            if mlp_relu_function is not None:
                self.fused_function = mlp_relu_function
                self.fused = True
        elif self.activation_fn_name == 'gelu':
            from onmt.modules.mlp.mlp import mlp_gelu_function
            if mlp_gelu_function is not None:
                self.fused_function = mlp_gelu_function
                self.fused = True

        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

        self.is_factorized = False
        self.multiplicative_factorize = False
        self.fast_factorize = False
        self.ffn_dim = config.decoder_ffn_dim

        self.n_languages = -1
        self.has_adapter = False
        self.adapter_location = -1

    @property
    def word_lut(self):
        return self.embed_tokens

    def freeze_self_attn_params(self):
        self.self_attn.q_proj.weight.requires_grad = False
        self.self_attn.k_proj.weight.requires_grad = False
        self.self_attn.v_proj.weight.requires_grad = False
        self.self_attn.out_proj.weight.requires_grad = False
        self.self_attn.q_proj.bias.requires_grad = False
        self.self_attn.k_proj.bias.requires_grad = False
        self.self_attn.v_proj.bias.requires_grad = False
        self.self_attn.out_proj.bias.requires_grad = False

    def freeze_ffn_params(self):
        self.fc1.weight.requires_grad = False
        self.fc2.weight.requires_grad = False
        self.fc1.bias.requires_grad = False
        self.fc2.bias.requires_grad = False

    def add_factorize(self, n_languages, rank=4, multiplicative=False, fast=False,
                      sub_factors=0, sub_factor_rank=-1):

        # add factorized weights for self-attention
        self.self_attn.add_factorized_weights(n_languages, rank=rank, multiplicative=multiplicative,
                                              fast=fast, sub_factors=sub_factors, sub_factor_rank=sub_factor_rank)
        self.encoder_attn.add_factorized_weights(n_languages, rank=rank, multiplicative=multiplicative,
                                                 fast=fast, sub_factors=sub_factors, sub_factor_rank=sub_factor_rank)

        # add factorized_weights for ffn
        self.is_factorized = True
        self.multiplicative_factorize = multiplicative
        self.fast_factorize = fast
        self.sub_factorized = (sub_factors > 0)

        self.r_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.ffn_dim))
        self.s_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.embed_dim))
        self.r_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.embed_dim))
        self.s_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.ffn_dim))

        nn.init.normal_(self.r_i, 0.0, 0.02)
        nn.init.normal_(self.s_i, 0.0, 0.02)
        nn.init.normal_(self.r_o, 0.0, 0.02)
        nn.init.normal_(self.s_o, 0.0, 0.02)

        if multiplicative:
            rank = rank if fast else 1
            self.rm_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.ffn_dim))
            self.sm_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.embed_dim))
            self.rm_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.embed_dim))
            self.sm_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.ffn_dim))

            constant = math.sqrt(1.0 / rank) if fast else 1
            nn.init.constant_(self.rm_i, constant)
            nn.init.constant_(self.sm_i, constant)
            nn.init.constant_(self.rm_o, constant)
            nn.init.constant_(self.sm_o, constant)

        if self.sub_factorized:
            self.sub_r_i = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, self.ffn_dim))
            self.sub_s_i = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, self.embed_dim))
            self.sub_r_o = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, self.embed_dim))
            self.sub_s_o = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, self.ffn_dim))

            nn.init.normal_(self.sub_r_i, 0.0, 0.02)
            nn.init.normal_(self.sub_s_i, 0.0, 0.02)
            nn.init.normal_(self.sub_r_o, 0.0, 0.02)
            nn.init.normal_(self.sub_s_o, 0.0, 0.02)

            if multiplicative:
                sub_factor_rank = sub_factor_rank if fast else 1
                self.sub_rm_i = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, self.ffn_dim))
                self.sub_sm_i = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, self.embed_dim))
                self.sub_rm_o = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, self.embed_dim))
                self.sub_sm_o = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, self.ffn_dim))

                constant = math.sqrt(1.0 / sub_factor_rank) if fast else 1
                nn.init.constant_(self.sub_rm_i, constant)
                nn.init.constant_(self.sub_sm_i, constant)
                nn.init.constant_(self.sub_rm_o, constant)
                nn.init.constant_(self.sub_sm_o, constant)

    def add_adapters(self, n_languages, downsampling_factor=4, adapter_location=1):
        """
        :param n_languages: one adapter per language
        :param downsampling_factor: downsampling rate size for the hidden layer
        :param adapter_location:
        :return:
        """

        self.n_languages = n_languages
        self.has_adapter = True
        self.adapter_location = adapter_location
        from .adapter import MultilingualAdapter
        self.adapter = MultilingualAdapter(n_languages, self.embed_dim, downsample_factor=downsampling_factor)

    def get_mlp_weights(self, lang=None, atb=None):

        in_weight = self.fc1.weight
        out_weight = self.fc2.weight
        in_bias = self.fc1.bias
        out_bias = self.fc2.bias

        if lang is not None:
            if self.is_factorized:
                if self.multiplicative_factorize:
                    rm_i = torch.index_select(self.rm_i, 0, lang).squeeze(0)  # squeeze possible because only 1
                    sm_i = torch.index_select(self.sm_i, 0, lang).squeeze(0)
                    rm_o = torch.index_select(self.rm_o, 0, lang).squeeze(0)
                    sm_o = torch.index_select(self.sm_o, 0, lang).squeeze(0)

                    if self.fast_factorize:
                        mul_factor_in = torch.mm(rm_i.t(), sm_i)
                        mul_factor_out = torch.mm(rm_o.t(), sm_o)
                    else:
                        mul_factor_in = torch.bmm(rm_i.unsqueeze(-1), sm_i.unsqueeze(1)).sum(dim=0)
                        mul_factor_out = torch.bmm(rm_o.unsqueeze(-1), sm_o.unsqueeze(1)).sum(dim=0)

                    # TODO: allow for multiple sub factorizers
                    if self.sub_factorized and atb is not None:
                        rm_i = torch.index_select(self.sub_rm_i, 0, atb).squeeze(0)  # squeeze possible because only 1
                        sm_i = torch.index_select(self.sub_sm_i, 0, atb).squeeze(0)
                        rm_o = torch.index_select(self.sub_rm_o, 0, atb).squeeze(0)
                        sm_o = torch.index_select(self.sub_sm_o, 0, atb).squeeze(0)

                        if self.fast_factorize:
                            sub_mul_factor_in = torch.mm(rm_i.t(), sm_i)
                            sub_mul_factor_out = torch.mm(rm_o.t(), sm_o)
                        else:
                            sub_mul_factor_in = torch.bmm(rm_i.unsqueeze(-1), sm_i.unsqueeze(1)).sum(dim=0)
                            sub_mul_factor_out = torch.bmm(rm_o.unsqueeze(-1), sm_o.unsqueeze(1)).sum(dim=0)

                        # has to be multiplicative here
                        mul_factor_in.mul_(sub_mul_factor_in)
                        mul_factor_out.mul_(sub_mul_factor_out)

                    in_weight = in_weight * mul_factor_in
                    out_weight = out_weight * mul_factor_out

                r_i = torch.index_select(self.r_i, 0, lang).squeeze(0)
                s_i = torch.index_select(self.s_i, 0, lang).squeeze(0)
                r_o = torch.index_select(self.r_o, 0, lang).squeeze(0)
                s_o = torch.index_select(self.s_o, 0, lang).squeeze(0)

                if self.fast_factorize:
                    add_factor_in = torch.mm(r_i.t(), s_i)
                    add_factor_out = torch.mm(r_o.t(), s_o)
                else:
                    add_factor_in = torch.bmm(r_i.unsqueeze(-1), s_i.unsqueeze(1)).sum(dim=0)
                    add_factor_out = torch.bmm(r_o.unsqueeze(-1), s_o.unsqueeze(1)).sum(dim=0)

                if self.sub_factorized and atb is not None:
                    r_i = torch.index_select(self.sub_r_i, 0, atb).squeeze(0)
                    s_i = torch.index_select(self.sub_s_i, 0, atb).squeeze(0)
                    r_o = torch.index_select(self.sub_r_o, 0, atb).squeeze(0)
                    s_o = torch.index_select(self.sub_s_o, 0, atb).squeeze(0)

                    if self.fast_factorize:
                        sub_add_factor_in = torch.mm(r_i.t(), s_i)
                        sub_add_factor_out = torch.mm(r_o.t(), s_o)
                    else:
                        sub_add_factor_in = torch.bmm(r_i.unsqueeze(-1), s_i.unsqueeze(1)).sum(dim=0)
                        sub_add_factor_out = torch.bmm(r_o.unsqueeze(-1), s_o.unsqueeze(1)).sum(dim=0)

                    # has to be additive here
                    add_factor_in.add_(sub_add_factor_in)
                    add_factor_out.add_(sub_add_factor_out)

                in_weight = in_weight + add_factor_in
                out_weight = out_weight + add_factor_out

        return in_weight, out_weight, in_bias, out_bias

    def call_mlp(self, x, in_weight, out_weight, in_bias, out_bias, activation_fn, dropout_p, training_,
                 fused, fused_function, checkpointing):
        """
        Move the MLP section to a different function to choose between pytorch and custom mlp
        :param x:
        :param in_weight:
        :param out_weight:
        :param in_bias:
        :param out_bias:
        :param activation_fn:
        :param dropout_p:
        :param training_:
        :param fused:
        :param fused_function:
        :return:
        """

        # TODO: check type x torch.half or torch.float32
        if fused and x.is_cuda:
            dropout_p_ = dropout_p if training_ else 0.0

            weights = [in_weight, out_weight]
            biases = [in_bias, out_bias]

            x = fused_function(dropout_p_, checkpointing, x, *weights, *biases)

        else:
            x = F.linear(x, in_weight, in_bias)
            x = activation_fn(x)
            x = F.dropout(x, dropout_p, training=training_)
            x = F.linear(x, out_weight, out_bias)

        return x

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            sub_encoder_hidden_states: Optional[torch.Tensor] = None,
            sub_encoder_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
            incremental: Optional[bool] = False,
            incremental_cache=None,
            checkpointing_ffn=False,
            checkpointing_cross_attn=False,
            lang=None, atb=None, **kwargs
    ):
        """
        :param checkpointing_cross_attn:
        :param checkpointing_ffn: Recompute the middle-layer of FFN to save memory
        :param hidden_states:
        :param attention_mask:
        :param encoder_hidden_states:
        :param encoder_attention_mask:
        :param sub_encoder_hidden_states:
        :param sub_encoder_attention_mask:
        :param output_attentions:
        :param incremental:
        :param incremental_cache:
        :param lang:
        :param atb:
        :param kwargs:
        :return:
        """
        if incremental and incremental_cache is None:
            incremental_cache = dict()

        # hidden_states = hidden_states.transpose(0, 1).contiguous()
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            incremental=incremental, incremental_cache=incremental_cache,
            lang=lang, atb=atb
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # hidden_states.add_(residual)
        hidden_states = hidden_states + residual
        # hidden_states = fused_dropout_add(hidden_states, residual, self.dropout, self.training)

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            attention_input = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            # cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, incremental_cache = self.encoder_attn(
                hidden_states=attention_input,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                incremental=incremental, incremental_cache=incremental_cache,
                checkpointing=checkpointing_cross_attn,
                lang=lang, atb=atb
            )

            # perform cross-attention on the sub-hidden states
            # if sub_encoder_hidden_states is not None:
            #     sub_hidden_states, sub_cross_attn_weights, _ = self.encoder_attn(
            #         hidden_states=attention_input,
            #         key_value_states=sub_encoder_hidden_states,
            #         attention_mask=sub_encoder_attention_mask,
            #         output_attentions=output_attentions,
            #         incremental=False, incremental_cache=None,
            #         lang=lang, mixture=mixture
            #     )
            #
            #     # t x b x h -> sum to 1
            #     contrastive_loss = F.mse_loss(hidden_states.float(), sub_hidden_states.float(), reduction='none')
            #
            # else:
            contrastive_loss = None

            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = hidden_states + residual
            # hidden_states = fused_dropout_add(hidden_states, residual, self.dropout, self.training)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        # if self.fused and hidden_states.is_cuda:
        #     weights = [self.fc1.weight, self.fc2.weight]
        #     biases = [self.fc1.bias, self.fc2.bias]
        #
        #     # seq_len, bsz, hidden_size = hidden_states.size(0), hidden_states.size(1), hidden_states.size(2)
        #     dropout = self.activation_dropout if self.training else 0.0
        #     hidden_states = self.fused_function(dropout, False, hidden_states, *weights, *biases).type_as(hidden_states)
        #
        #     # hidden_states = hidden_states.view(seq_len, bsz, hidden_size)
        # else:
        #     hidden_states = self.activation_fn(self.fc1(hidden_states))
        #     hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        #     hidden_states = self.fc2(hidden_states)

        in_weight, out_weight, in_bias, out_bias = self.get_mlp_weights(lang=lang, atb=atb)
        hidden_states = self.call_mlp(hidden_states, in_weight, out_weight, in_bias, out_bias,
                                      self.activation_fn, self.activation_dropout, self.training,
                                      self.fused, self.fused_function, checkpointing_ffn)

        # hidden_states = fused_dropout_add(hidden_states, residual, self.dropout, self.training)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = hidden_states + residual
        # hidden_states.add_(residual)

        if self.has_adapter:
            residual = hidden_states
            if self.adapter_location == 1:
                assert lang is not None
                hidden_states = self.adapter(hidden_states, lang=lang)

            hidden_states = hidden_states + residual

        # hidden_states = hidden_states.transpose(0, 1).contiguous()

        if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if contrastive_loss is not None:
            # print("Return contrastive loss here,", contrastive_loss.size())
            outputs += (contrastive_loss, )

        return outputs, incremental_cache


class MBartPreTrainedModel(PreTrainedModel):
    config_class = MBartConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (MBartDecoder, MBartDecoder)):
            module.gradient_checkpointing = value

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


MBART_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.MBartConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

MBART_GENERATION_EXAMPLE = r"""
    Summarization example::

        >>> from transformers import MBartTokenizer, MBartForConditionalGeneration, MBartConfig

        >>> model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-cc25')
        >>> tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')

        >>> ARTICLE_TO_SUMMARIZE = "Meine Freunde sind cool, aber sie essen zu viel Kuchen."
        >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')

        >>> # Generate Summary
        >>> summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        >>> print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

    Mask filling example::

        >>> from transformers import MBartTokenizer, MBartForConditionalGeneration
        >>> tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')
        >>> # de_DE is the language symbol id <LID> for German
        >>> TXT = "</s> Meine Freunde sind <mask> nett aber sie essen zu viel Kuchen. </s> de_DE"

        >>> model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-cc25')
        >>> input_ids = tokenizer([TXT], add_special_tokens=False, return_tensors='pt')['input_ids']
        >>> logits = model(input_ids).logits

        >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
        >>> probs = logits[0, masked_index].softmax(dim=0)
        >>> values, predictions = probs.topk(5)

        >>> tokenizer.decode(predictions).split()
"""

MBART_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using :class:`~transformers.MBartTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.MBartTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are decoder input IDs? <../glossary.html#decoder-input-ids>`__

            MBart uses a specific language id token as the starting token for :obj:`decoder_input_ids` generation that
            varies according to source and target language, *e.g.* 25004 for `en_XX`, and 25003 for `de_DE`. If
            :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).

            For translation and summarization training, :obj:`decoder_input_ids` should be provided. If no
            :obj:`decoder_input_ids` is provided, the model will create this tensor by shifting the :obj:`input_ids` to
            the right for denoising pre-training following the paper.
        decoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.

        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, `optional`: :obj:`hidden_states`, `optional`:
            :obj:`attentions`) :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`,
            `optional`) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
            cross-attention of the decoder.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2 tensors
            of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
            shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size, sequence_length)`.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation. If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_inputs_embeds`
            have to be input (see :obj:`past_key_values`). This is useful if you want more control over how to convert
            :obj:`decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both unset, :obj:`decoder_inputs_embeds`
            takes the value of :obj:`inputs_embeds`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


class MBartEncoder(MBartPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`MBartEncoderLayer`.

    Args:
        config: MBartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: MBartConfig, opt, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        config.dropout = opt.residual_dropout if opt.residual_dropout > 0 else opt.dropout
        config.attention_dropout = opt.attn_dropout
        config.activation_dropout = opt.ffn_dropout if opt.ffn_dropout > 0 else opt.dropout
        config.layerdrop = opt.death_rate
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.opt = opt
        self.word_dropout = opt.word_dropout

        embed_dim = config.d_model
        self.embed_dim = embed_dim
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = MBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([MBartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim)
        self.layer_norm = LayerNorm(config.d_model)

        self.init_weights()
        self.gradient_checkpointing = False
        from onmt.modules.optimized.fast_mha import fast_bert_mha
        self.fast_bert_mha = fast_bert_mha

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            checkpointing_ffn=False
    ):
        """
        :param input_ids: [T x B] discrete input tokens
        :param attention_mask: [B x T] attention mask (padded = 1, non-pad = 0]
        :param inputs_embeds: [T x B x H] optional
        :param output_attentions:
        :param output_hidden_states:
        :param return_dict:
        :return:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # # retrieve input_ids and inputs_embeds
        # if input_ids is not None and inputs_embeds is not None:
        #     raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # elif input_ids is not None:
        #     pass
        # elif inputs_embeds is not None:
        #     pass
        # else:
        #     raise ValueError("You have to specify either input_ids or inputs_embeds")
        #
        # if inputs_embeds is None:
        #     inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            bsz, seq_len = input_ids.size(0), input_ids.size(1)
            input_shape = torch.Size([bsz, seq_len])

        elif inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = embedded_dropout(self.embed_tokens, input_ids,
                                             dropout=self.word_dropout if self.training else 0)
            inputs_embeds = inputs_embeds * self.embed_scale

            inputs_embeds = inputs_embeds.view(bsz, seq_len, -1)

            embed_pos = self.embed_positions(input_shape)
            hidden_states = inputs_embeds + embed_pos
            hidden_states = self.layernorm_embedding(hidden_states)
        else:
            # use the input embeds from another stack
            # maybe don't use layernorm_embedding
            hidden_states = inputs_embeds
            hidden_states = self.layernorm_embedding(hidden_states)

        # should we use layernorm embedding here?
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # TODO: use fast bert mha
        can_run_fast_bert_mha = False
        # check if fast bert mha can be run
        seq_len = hidden_states.size(1)
        bsz = hidden_states.size(0)
        sm = torch.cuda.get_device_capability()
        total_bsz = 0

        # only run this when seq_len <= 512 and sm = 80/86 and type = half
        if self.fast_bert_mha and (seq_len <= 512 and bsz >= 4 and sm[0] == 8 and sm[1] in [0]) \
                and hidden_states.dtype == torch.half:
            can_run_fast_bert_mha = True
            # print("Can run FAST BERT MHA")

            x = hidden_states
            padding_mask = attention_mask  # [B x T]
            # masked positions = 1 so to compute length we need the (1 -)
            if padding_mask is None:
                padding_mask = x.new_zeros(bsz, seq_len)
            padding_mask = padding_mask.long()
            lengths = (1 - padding_mask).sum(dim=1)
            lengths = lengths.cpu().tolist()  # list of lengths for B seqs

            x = x.view(-1, x.size(-1))
            non_pad_indices = torch.nonzero(padding_mask.view(-1).ne(1)).squeeze(1)
            hidden_states = x.index_select(0, non_pad_indices)

            max_len = max(lengths)
            # cumulative sequence lengths (required input for fmha)
            a = torch.tensor(np.array([0] + lengths), dtype=torch.int32)
            cu_seqlens = torch.cumsum(a, 0).to(dtype=torch.int32, device=x.device)
        else:
            max_len = -1
            cu_seqlens = None
            non_pad_indices = None

        if not can_run_fast_bert_mha:
            # transpose from [B x T x H] to [T x B x H]
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                    max_len=max_len, cu_seqlens=cu_seqlens,
                    checkpointing_ffn=checkpointing_ffn
                )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        # if we remove padding before (for fast bert MHA) then remember to put padding back
        # to restore the form B x T X H
        if can_run_fast_bert_mha:
            # remove the patch
            # if x.size(0) > total_bsz:
            #     x = x[:total_bsz, :]
            hidden_states = index_copy(hidden_states, non_pad_indices, bsz * seq_len)
            hidden_states = hidden_states.view(bsz, seq_len, -1)
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)


class MBartDecoder(MBartPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`MBartDecoderLayer`
\
    Args:
        config: MBartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: MBartConfig, opt, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        config.dropout = opt.residual_dropout if opt.residual_dropout > 0 else opt.dropout
        config.activation_dropout = opt.ffn_dropout if opt.ffn_dropout > 0 else opt.dropout
        config.attention_dropout = opt.attn_dropout
        self.dropout = config.dropout

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = MBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([MBartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = LayerNorm(config.d_model)
        self.layer_norm = LayerNorm(config.d_model)

        self.init_weights()
        self.gradient_checkpointing = False

        self.model_size = config.d_model
        self.switchout = 0.0
        # self.word_lut = self.embed_tokens
        self.config.bert_hidden_size = config.d_model
        self.layerdrop = opt.death_rate_decoder
        self.dec_pretrained_model = 'mbart'
        if opt.freeze_embedding:
            self.embed_tokens.weight.requires_grad = False
        self.word_dropout = opt.word_dropout

        # freeze parameters if declared
        if opt.freeze_decoder_self_attn:
            self.freeze_self_attn_params()

        if opt.freeze_decoder_ffn:
            self.freeze_ffn_params()

        if opt.freeze_decoder:
            for p in self.parameters():
                p.requires_grad = False

            if not opt.freeze_cross_attention:
                # but we need to enable the cross attention
                for layer in self.layers:
                    for p in layer.encoder_attn.parameters():
                        p.requires_grad = True
                    for p in layer.encoder_attn_layer_norm.parameters():
                        p.requires_grad = True

        if opt.multilingual_factorized_weights_decoder:
            print("[INFO] Factorizing MBART model into %d languages and %d factors"
                  % (opt.n_languages, opt.n_attributes))
            self.add_factorize(opt.n_languages, rank=opt.mfw_rank,
                               multiplicative=opt.mfw_multiplicative,
                               fast=opt.fast_factorize,
                               sub_factors=opt.n_attributes,
                               sub_factor_rank=math.floor(opt.mfw_rank * opt.mfw_atb_rank_scale))

        # adapter
        if opt.decoder_adapter > 0:
            print("[INFO] Adding MBART Adapters for %d languages" % opt.n_languages)
            for layer in self.layers:
                layer.add_adapters(opt.n_languages, adapter_location=opt.decoder_adapter)

    def freeze_self_attn_params(self):

        self.layer_norm.weight.requires_grad = False
        self.layer_norm.bias.requires_grad = False

        for layer in self.layers:
            layer.freeze_self_attn_params()

    def freeze_ffn_params(self):

        for layer in self.layers:
            layer.freeze_ffn_params()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def add_factorize(self, n_languages, rank=4, multiplicative=False, fast=False,
                      sub_factors=0, sub_factor_rank=-1):

        for layer in self.layers:
            layer.add_factorize(n_languages, rank=rank, multiplicative=multiplicative,
                                fast=fast, sub_factors=sub_factors, sub_factor_rank=sub_factor_rank)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            sub_encoder_hidden_states=None,
            sub_encoder_attention_mask=None,
            inputs_embeds=None,
            incremental=False, incremental_cache=None,
            lang=None, atb=None,
            output_attentions=None,
            output_hidden_states=None,
            checkpointing_ffn=False,
            checkpointing_cross_attn=False,
    ):
        """
        :param checkpointing_cross_attn:
        :param input_ids: [batch_size x seq_len]
        :param attention_mask:
        :param encoder_hidden_states:
        :param encoder_attention_mask:
        :param sub_encoder_hidden_states:
        :param sub_encoder_attention_mask:
        :param inputs_embeds:
        :param incremental:
        :param incremental_cache:
        :param lang:
        :param atb:
        :param output_attentions:
        :param output_hidden_states:
        :param checkpointing_ffn:
        :return:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = 0

        if inputs_embeds is None:
            inputs_embeds = embedded_dropout(self.embed_tokens, input_ids,
                                             dropout=self.word_dropout if self.training else 0)
            inputs_embeds = inputs_embeds * self.embed_scale

        qlen = input_ids.size(1)
        klen = qlen

        # if attention_mask is None:
        attention_mask = torch.triu(
            inputs_embeds.new_ones(qlen, klen), diagonal=1).bool()

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            # encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            encoder_attention_mask = encoder_attention_mask

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        # hidden_states = hidden_states
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        # next_decoder_cache = () if use_cache else None
        contrastive_loss = 0

        hidden_states = hidden_states.transpose(0, 1).contiguous()

        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            # encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            encoder_attention_mask = encoder_attention_mask

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Stochastic Layer
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            layer_outputs, _ = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                sub_encoder_hidden_states=sub_encoder_hidden_states,
                sub_encoder_attention_mask=sub_encoder_attention_mask,
                output_attentions=output_attentions,
                lang=lang,
                atb=atb,
                checkpointing_ffn=checkpointing_ffn,
                checkpointing_cross_attn=checkpointing_cross_attn,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

            # add up the contrastive_loss per layer
            if sub_encoder_hidden_states is not None:
                contrastive_loss_ = layer_outputs[-1]
                # print("Receive contrastive loss after layer", contrastive_loss_.size())
                contrastive_loss = contrastive_loss + contrastive_loss_

        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return tuple(
            v
            for v in [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions, contrastive_loss]
            if v is not None
        )

    def step(self, input, decoder_state, **kwargs):

        # context is stored in the decoder state in [T B H] format
        encoder_hidden_states = decoder_state.context

        buffers = decoder_state.attention_buffers
        lang = decoder_state.tgt_lang
        atb = decoder_state.tgt_atb
        src_lang = decoder_state.src_lang
        buffering = decoder_state.buffering

        input_ids = input
        input_shape = input_ids.size()
        time_step = input.size(1)
        # print("[DEBUGGING] Current time step: %d" % time_step)

        input_ = input
        if buffering:
            # use the last value of input to continue decoding
            if input.size(1) > 1:
                input_ = input[:, -1:]
            past_key_values_length = input.size(1) - 1
        else:
            past_key_values_length = 0

        inputs_embeds = self.embed_tokens(input_) * self.embed_scale
        # inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        qlen = input_ids.size(1)
        klen = qlen
        attention_mask = torch.triu(
                inputs_embeds.new_ones(qlen, klen), diagonal=1).bool()

        if buffering:
            attention_mask = attention_mask[-1:, :]

        encoder_attention_mask = decoder_state.src_mask
        if not self.layers[0].encoder_attn.fast_attention:
            encoder_attention_mask = 1 - encoder_attention_mask
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_.size(-1))
        else:
            encoder_attention_mask = encoder_attention_mask.bool()

        # embed positions
        positions = self.embed_positions(input_.size(), past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = hidden_states.transpose(0, 1)

        hidden_states = self.layernorm_embedding(hidden_states)

        for idx, decoder_layer in enumerate(self.layers):

            if buffering:
                buffer = buffers[idx] if idx in buffers else None
            else:
                buffer = None

            layer_outputs, buffer = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=None,
                incremental=buffering, incremental_cache=buffer,
                lang=lang, atb=atb
            )

            if buffering:
                decoder_state.update_attention_buffer(buffer, idx)
            hidden_states = layer_outputs[0]

        hidden_states = self.layer_norm(hidden_states)
        output = hidden_states[-1].unsqueeze(0)

        # just a fake coverage
        coverage = hidden_states.new(hidden_states.size(1), 1, encoder_hidden_states.size(0)).zero_()

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = output
        output_dict['coverage'] = coverage
        output_dict['context'] = encoder_hidden_states
        return output_dict