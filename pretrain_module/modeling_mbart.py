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

from torch.nn import CrossEntropyLoss, MSELoss
from onmt.modules.layer_norm import LayerNorm
from onmt.modules.optimized.self_attention_func import self_attn_func
from onmt.modules.optimized.encdec_attention_func_bias import encdec_attn_bias_func
from onmt.modules.dropout import embedded_dropout
from onmt.modules.optimized.dropout_add import fused_dropout_add

from .activations import ACT2FN
# from ...file_utils import (
#     add_code_sample_docstrings,
#     add_end_docstrings,
#     add_start_docstrings,
#     add_start_docstrings_to_model_forward,
#     replace_return_docstrings,
# )
from .modeling_outputs import (
    BaseModelOutput,
    # BaseModelOutputWithPastAndCrossAttentions,
    # CausalLMOutputWithCrossAttentions,
    # Seq2SeqLMOutput,
    # Seq2SeqModelOutput,
    # Seq2SeqQuestionAnsweringModelOutput,
    # Seq2SeqSequenceClassifierOutput,
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


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    """
    prev_output_tokens = input_ids.clone()

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


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

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def add_factorized_weights(self, n_languages, rank=4, multiplicative=False, fast=False):

        embed_dim = self.embed_dim
        self.is_factorized = True
        self.multiplicative_factorize = multiplicative
        self.fast_factorize = fast

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
            lang=None, mixture=None,
            incremental=False, incremental_cache=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        # is_cross_attention = key_value_states is not None

        if not self.fast_attention:
            raise NotImplementedError("Slow attention by HuggingFace not supported anymore.")
            # assert key_value_states is None
            # bsz, tgt_len, embed_dim = hidden_states.size()
            #
            # # get query proj
            # query_states = self.q_proj(hidden_states) * self.scaling
            # key_states = self.k_proj(hidden_states)
            # value_states = self.v_proj(hidden_states)
            #
            # if incremental:
            #     if 'k' in incremental_cache and 'v' in incremental_cache:
            #         key_states = torch.cat([incremental_cache['k'], key_states], dim=1)  # time first
            #         value_states = torch.cat([incremental_cache['v'], value_states], dim=1)  # time first
            #
            #         incremental_cache['k'] = key_states
            #         incremental_cache['v'] = value_states
            #     else:
            #         incremental_cache['k'] = key_states
            #         incremental_cache['v'] = value_states
            #
            # # reshape into B x H x T x D ?
            # key_states = self._shape(key_states, -1, bsz)
            # value_states = self._shape(value_states, -1, bsz)
            #
            # proj_shape = (bsz * self.num_heads, -1, self.head_dim)
            # query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
            # key_states = key_states.view(*proj_shape)
            # value_states = value_states.view(*proj_shape)
            #
            # src_len = key_states.size(1)
            # attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
            #
            # if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            #     raise ValueError(
            #         f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            #     )
            #
            # if attention_mask is not None:
            #     if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            #         raise ValueError(
            #             f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            #         )
            #     attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            #     attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
            #
            # attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            #
            # if output_attentions:
            #     # this operation is a bit awkward, but it's required to
            #     # make sure that attn_weights keeps its gradient.
            #     # In order to do so, attn_weights have to be reshaped
            #     # twice and have to be reused in the following
            #     attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            #     attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
            # else:
            #     attn_weights_reshaped = None
            #
            # attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
            #
            # attn_output = torch.bmm(attn_probs, value_states)
            #
            # if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            #     raise ValueError(
            #         f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            #     )
            #
            # attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            # attn_output = attn_output.transpose(1, 2)
            # attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
            #
            # attn_output = self.out_proj(attn_output)
            # coverage = attn_weights_reshaped

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

                in_proj_weight = in_proj_weight + add_factor_in
                out_proj_weight = out_proj_weight + add_factor_out

            use_time_mask = self.is_decoder
            qlen, klen = hidden_states.size(0), hidden_states.size(0)
            mask = attention_mask
            low_precision = True  # Use CUDA impl

            #TODO: add factorize

            attn_output, coverage = self_attn_func(use_time_mask, self.training, self.num_heads, hidden_states,
                                                   in_proj_weight, out_proj_weight,
                                                   self.proj_bias, self.out_proj.bias,
                                                   mask, self.dropout,
                                                   False, None,
                                                   incremental, incremental_cache, low_precision, True)

            attn_output = attn_output

        return attn_output, coverage, incremental_cache


class MBartCrossAttention(MBartAttention):

    def convert_fast_attention(self):

        if self.fast_attention:
            return

        # print("[INFO] Convert MBartCrossAttention from slow to fast")
        self.fast_attention = True
        # w_q = self.q_proj.weight.clone()
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
        del self.k_proj, self.v_proj

    def add_factorized_weights(self, n_languages, rank=4, multiplicative=False, fast=False):

        embed_dim = self.embed_dim
        self.is_factorized = True
        self.multiplicative_factorize = multiplicative
        self.fast_factorize = fast

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

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            lang=None, mixture=None,
            incremental=False, incremental_cache=None
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

                in_proj_weight_q = in_proj_weight_q + add_factor_q
                in_proj_weight_kv = in_proj_weight_kv + add_factor_kv
                out_proj_weight = out_proj_weight + add_factor_out

            recompute = False
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
        self.dropout = config.dropout

        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

        self.activation_fn_name = config.activation_function
        self.fused = False
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

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

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

        self.self_attn = MBartAttention(
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
            dropout=config.attention_dropout,
            is_decoder=True,
        )

        self.activation_fn_name = config.activation_function
        self.fused = False
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

    @property
    def word_lut(self):
        return self.embed_tokens

    def freeze_self_attn_params(self):

        self.self_attn_layer_norm.weight.requires_grad = False
        self.self_attn_layer_norm.bias.requires_grad = False
        self.self_attn.q_proj.weight.requires_grad = False
        self.self_attn.k_proj.weight.requires_grad = False
        self.self_attn.v_proj.weight.requires_grad = False
        self.self_attn.out_proj.weight.requires_grad = False
        self.self_attn.q_proj.bias.requires_grad = False
        self.self_attn.k_proj.bias.requires_grad = False
        self.self_attn.v_proj.bias.requires_grad = False
        self.self_attn.out_proj.bias.requires_grad = False

    def freeze_ffn_params(self):
        self.final_layer_norm.weight.requires_grad = False
        self.final_layer_norm.bias.requires_grad = False
        self.fc1.weight.requires_grad = False
        self.fc2.weight.requires_grad = False
        self.fc1.bias.requires_grad = False
        self.fc2.bias.requires_grad = False

    def add_factorize(self, n_languages, rank=4, multiplicative=False, fast=False):

        self.self_attn.add_factorized_weights(n_languages, rank=rank, multiplicative=multiplicative, fast=fast)
        self.encoder_attn.add_factorized_weights(n_languages, rank=rank, multiplicative=multiplicative, fast=fast)

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

    def get_mlp_weights(self, lang=None, mixture=None):

        in_weight = self.fc1.weight
        out_weight = self.fc2.weight
        in_bias = self.fc1.bias
        out_bias = self.fc2.bias

        if lang is not None:
            assert mixture is None

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

                in_weight = in_weight + add_factor_in
                out_weight = out_weight + add_factor_out

        if mixture is not None:
            raise NotImplementedError

        return in_weight, out_weight, in_bias, out_bias

    def call_mlp(self, x, in_weight, out_weight, in_bias, out_bias, activation_fn, dropout_p, training_,
                 fused, fused_function):
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

            x = fused_function(dropout_p_, False, x, *weights, *biases)

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
            # past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            incremental: Optional[bool] = False,
            incremental_cache=None,
            lang=None, mixture=None
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            incremental_cache:
            incremental:
        """
        if incremental and incremental_cache is None:
            incremental_cache = dict()
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
            lang=lang, mixture=mixture
        )
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # hidden_states = residual + hidden_states
        hidden_states = fused_dropout_add(hidden_states, residual, self.dropout, self.training)

        # Cross-Attention Block
        # cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            # cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, incremental_cache = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                incremental=incremental, incremental_cache=incremental_cache,
                lang=lang, mixture=mixture
            )
            # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            # hidden_states = residual + hidden_states
            hidden_states = fused_dropout_add(hidden_states, residual, self.dropout, self.training)

            # add cross-attn to positions 3,4 of present_key_value tuple
            # present_key_value = present_key_value + cross_attn_present_key_value

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

        in_weight, out_weight, in_bias, out_bias = self.get_mlp_weights(lang=lang, mixture=mixture)
        hidden_states = self.call_mlp(hidden_states, in_weight, out_weight, in_bias, out_bias,
                                      self.activation_fn, self.activation_dropout, self.training,
                                      self.fused, self.fused_function)

        hidden_states = fused_dropout_add(hidden_states, residual, self.dropout, self.training)
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

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

    def __init__(self, config: MBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
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

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.MBartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__

            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class MBartDecoder(MBartPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`MBartDecoderLayer`

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

        if opt.multilingual_factorized_weights_decoder:
            print("[INFO] Factorizing MBART model into %d languages" % opt.n_languages)
            self.add_factorize(opt.n_languages, rank=opt.mfw_rank,
                               multiplicative=opt.mfw_multiplicative,
                               fast=opt.fast_factorize)

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

    def add_factorize(self, n_languages, rank=4, multiplicative=False, fast=False):

        for layer in self.layers:
            layer.add_factorize(n_languages, rank=rank, multiplicative=multiplicative, fast=fast)

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, dtype, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            # past_key_values=None,
            inputs_embeds=None,
            incremental=False, incremental_cache=None,
            lang=None, mixture=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.MBartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
                Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2
                tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional
                tensors of shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        hidden_states = hidden_states.transpose(0, 1)
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        # next_decoder_cache = () if use_cache else None

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
                output_attentions=output_attentions,
                lang=lang,
                mixture=mixture
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

    def step(self, input, decoder_state, **kwargs):

        # context is stored in the decoder state in [T B H] format
        encoder_hidden_states = decoder_state.context

        buffers = decoder_state.attention_buffers
        lang = decoder_state.tgt_lang
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
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
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
                lang=lang, mixture=None
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