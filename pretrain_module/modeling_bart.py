# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch BART model. """
import copy
import math
import random
import warnings
from typing import Optional, Tuple
import numpy as np

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import Parameter
from torch.cuda.amp import autocast

from onmt.modules.optimized.linear import linear_function, factorize_linear

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
from .configuration_bart import BartConfig
import onmt
from collections import defaultdict


_CHECKPOINT_FOR_DOC = "facebook/bart-large"
_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"


BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-large",
    # See all BART models at https://huggingface.co/models?filter=bart
]


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


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


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
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


class BartAttention(nn.Module):
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
        self.flex_factorize = False

        from onmt.modules.optimized.flash_mha import flash_bert_mha
        self.fast_bert_mha = flash_bert_mha

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def add_ensemble_weights(self, n_ensemble, rank=4, **kwargs):
        """
        Add ensemble weights for self-attention

        :param: n_ensemble
        :param: rank
        """
        # TODO
        # embed_dim = self.embed_dim
        # self.is_factorized = True
        # self.multiplicative_factorize = multiplicative
        # self.fast_factorize = fast
        # self.flex_factorize = flexible
        # self.dyrank = dyrank
        #
        # if multiplicative or flexible:
        #     _rank = rank if fast else 1
        #     self.rm_i = torch.nn.Parameter(torch.Tensor(n_languages, _rank, 3 * embed_dim))
        #     self.sm_i = torch.nn.Parameter(torch.Tensor(n_languages, _rank, embed_dim))
        #     self.rm_o = torch.nn.Parameter(torch.Tensor(n_languages, _rank, embed_dim))
        #     self.sm_o = torch.nn.Parameter(torch.Tensor(n_languages, _rank, embed_dim))
        #
        #     constant = 1
        #     nn.init.constant_(self.rm_i, constant)
        #     nn.init.constant_(self.sm_i, constant)
        #     nn.init.constant_(self.rm_o, constant)
        #     nn.init.constant_(self.sm_o, constant)
        #
        # if not self.flex_factorize:
        #
        #     self.r_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, 3 * embed_dim))
        #     self.s_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
        #     self.r_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
        #     self.s_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
        #
        #     if self.dyrank:
        #         nn.init.zeros_(self.r_i)
        #         nn.init.normal_(self.s_i, 0.0, 0.02)
        #         nn.init.zeros_(self.r_o)
        #         nn.init.normal_(self.s_o, 0.0, 0.02)
        #
        #     else:
        #         std = 0.01 if fast else 0.02
        #         nn.init.normal_(self.r_i, 0.0, std)
        #         nn.init.normal_(self.s_i, 0.0, std)
        #         nn.init.normal_(self.r_o, 0.0, std)
        #         nn.init.normal_(self.s_o, 0.0, std)

    def convert_fast_attention(self):

        # HuggingFace's MBart Attention uses a unoptimized memory layout that requires reshaping
        # This re-organizes the memory to fit FastAttention and FlashAttention codes

        if self.fast_attention:
            return

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
            incremental=False, incremental_cache=None,
            checkpointing=False, stacked_kv=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        if not self.fast_attention:
            raise NotImplementedError("Slow attention by HuggingFace is deprecated.")

        else:

            in_proj_weight = self.proj_weight
            out_proj_weight = self.out_proj.weight
            rm_i, sm_i, rm_o, sm_o = None, None, None, None

            if self.is_factorized and self.flex_factorize:

                n_languages, _rank = self.rm_o.size(0), self.rm_o.size(1)

                if lang.ndim == 1:

                    rm_i = torch.index_select(self.rm_i, 0, lang).squeeze(0)  # squeeze possible because only 1
                    sm_i = torch.index_select(self.sm_i, 0, lang).squeeze(0)
                    rm_o = torch.index_select(self.rm_o, 0, lang).squeeze(0)
                    sm_o = torch.index_select(self.sm_o, 0, lang).squeeze(0)

                elif lang.ndim == 2:  # for flash attention with nested tensor
                    rm_i = torch.mm(lang, self.rm_i.view(n_languages, _rank * self.rm_i.size(-1))).view(
                        lang.size(0), _rank,
                        self.rm_i.size(-1))
                    sm_i = torch.mm(lang, self.sm_i.view(n_languages, _rank * self.sm_i.size(-1))).view(
                        lang.size(0), _rank,
                        self.sm_i.size(-1))
                    rm_o = torch.mm(lang, self.rm_o.view(n_languages, _rank * self.rm_o.size(-1))).view(
                        lang.size(0), _rank,
                        self.rm_o.size(-1))
                    sm_o = torch.mm(lang, self.sm_o.view(n_languages, _rank * self.sm_o.size(-1))).view(
                        lang.size(0), _rank,
                        self.sm_o.size(-1))

                elif lang.ndim == 3:
                    _len, _bsz = lang.size(0), lang.size(1)
                    _lang = lang.view(_len * _bsz, lang.size(-1))
                    rm_i = torch.mm(_lang, self.rm_i.view(n_languages, _rank * self.rm_i.size(-1))).view(
                        _len, _bsz, _rank, self.rm_i.size(-1))
                    sm_i = torch.mm(_lang, self.sm_i.view(n_languages, _rank * self.sm_i.size(-1))).view(
                        _len, _bsz, _rank, self.sm_i.size(-1))
                    rm_o = torch.mm(_lang, self.rm_o.view(n_languages, _rank * self.rm_o.size(-1))).view(
                        _len, _bsz, _rank, self.rm_o.size(-1))
                    sm_o = torch.mm(_lang, self.sm_o.view(n_languages, _rank * self.sm_o.size(-1))).view(
                        _len, _bsz, _rank, self.sm_o.size(-1))

                if hidden_states.ndim == 3:
                    use_time_mask = self.is_decoder
                    bsz, qlen = hidden_states.size(1), hidden_states.size(0)
                    mask = attention_mask
                    low_precision = True  # Use CUDA impl

                    input_lin_results = factorize_linear(hidden_states, in_proj_weight, self.proj_bias, rm_i, sm_i)

                    attn_output, coverage = self_attn_compact_func(use_time_mask, self.training, self.num_heads,
                                                                   input_lin_results,
                                                                   mask, self.dropout,
                                                                   False, None,
                                                                   incremental, incremental_cache, low_precision,
                                                                   True, checkpointing)

                    attn_output = attn_output.view(qlen, bsz, -1).contiguous()

                    output = factorize_linear(attn_output, out_proj_weight, self.out_proj.bias, rm_o, sm_o)

                    return output, coverage, incremental_cache

                else:
                    """
                    flash attention
                    """
                    assert self.fast_bert_mha is not None
                    assert cu_seqlens is not None
                    assert max_len is not None

                    total_bsz = hidden_states.size(0)
                    # qkv = linear_function(hidden_states, in_proj_weight, self.proj_bias)  # B x H
                    qkv = factorize_linear(hidden_states, in_proj_weight, self.proj_bias, rm_i, sm_i)
                    # B x 3 x H x d

                    # TODO: moving to CUDA to remove overhead?
                    qkv = qkv.view(total_bsz, self.num_heads, 3, self.head_dim).transpose(1, 2).contiguous()

                    dropout_p = self.dropout if self.training else 0.0
                    causal = self.is_decoder
                    softmax_scale = 1.0 / math.sqrt(64)
                    context = self.fast_bert_mha(qkv, cu_seqlens, max_len, dropout_p, softmax_scale, causal, False)
                    coverage = None

                    context = context.view(-1, self.num_heads * self.head_dim).contiguous()
                    output = factorize_linear(context, out_proj_weight, self.out_proj.bias, rm_o, sm_o)

                    return output, coverage, incremental_cache

            # Code is twice as long TODO: merging two sections

            else:
                """
                flash attention
                """
                assert self.fast_bert_mha is not None
                assert cu_seqlens is not None
                assert max_len is not None
                # assert self.is_decoder is False  # only encoder
                # sm = torch.cuda.get_device_capability()

                # Only Ampere supported at the moment-
                total_bsz = hidden_states.size(0)
                qkv = linear_function(hidden_states, in_proj_weight, self.proj_bias)  # B x H
                # B x 3 x H x d

                # TODO: moving to CUDA to remove overhead?
                # qkv = qkv.view(total_bsz, self.num_heads, 3, self.head_dim).transpose(1, 2).contiguous()

                # context, coverage = self.fast_bert_mha(qkv, cu_seqlens, self.dropout, max_len, self.training)
                qkv = qkv.view(total_bsz, self.num_heads, 3, self.head_dim).transpose(1, 2).contiguous()

                dropout_p = self.dropout if self.training else 0.0
                causal = self.is_decoder
                softmax_scale = 1.0 / math.sqrt(64)
                context = self.fast_bert_mha(qkv, cu_seqlens, max_len, dropout_p, softmax_scale, causal, False)
                coverage = None

                context = context.view(-1, self.num_heads * self.head_dim).contiguous()
                outputs = linear_function(context, out_proj_weight, self.out_proj.bias)

                attn_output = outputs

            return attn_output, coverage, incremental_cache


class BartCrossAttention(BartAttention):

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
            convert_fast_attention=False,
            **kwargs
    ):
        super().__init__(embed_dim, num_heads, dropout, is_decoder, bias)

        from onmt.modules.optimized.flash_mha import flash_encdec_mha
        self.fast_bert_mha = flash_encdec_mha

        if convert_fast_attention:
            self.convert_fast_attention()

    def convert_fast_attention(self):

        if self.fast_attention:
            return

        self.fast_attention = True

        # Merge weight KV into one
        w_k = self.k_proj.weight.clone()
        w_v = self.v_proj.weight.clone()
        weights = [w_k, w_v]
        weight_ = torch.cat(weights, dim=0).contiguous()

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

    def add_factorized_weights(self, n_languages, rank=4,
                               multiplicative=False, fast=False,
                               flexible=False, dyrank=False, **kwargs):

        embed_dim = self.embed_dim
        self.is_factorized = True
        self.multiplicative_factorize = multiplicative
        self.fast_factorize = fast
        self.flex_factorize = flexible
        self.dyrank = dyrank

        # if not fast: the weights are calculated first
        # W = W_S * (rm \dot sm) + (r \dot s)
        # if fast: maybe using only W_S
        # WX + b = W(rm \dot sm)X + b
        #        = W(X \dot sm)rm + b

        if multiplicative or self.flex_factorize:
            _rank = rank if fast or flexible else 1
            self.rm_q = torch.nn.Parameter(torch.Tensor(n_languages, _rank, embed_dim))
            self.sm_q = torch.nn.Parameter(torch.Tensor(n_languages, _rank, embed_dim))
            self.rm_kv = torch.nn.Parameter(torch.Tensor(n_languages, _rank, 2 * embed_dim))
            self.sm_kv = torch.nn.Parameter(torch.Tensor(n_languages, _rank, embed_dim))
            self.rm_o = torch.nn.Parameter(torch.Tensor(n_languages, _rank, embed_dim))
            self.sm_o = torch.nn.Parameter(torch.Tensor(n_languages, _rank, embed_dim))

            constant = 1
            nn.init.constant_(self.rm_q, constant)
            nn.init.constant_(self.sm_q, constant)
            nn.init.constant_(self.rm_kv, constant)
            nn.init.constant_(self.sm_kv, constant)
            nn.init.constant_(self.rm_o, constant)
            nn.init.constant_(self.sm_o, constant)

        if not flexible:
            self.r_q = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.s_q = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.r_kv = torch.nn.Parameter(torch.Tensor(n_languages, rank, 2 * embed_dim))
            self.s_kv = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.r_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.s_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))

            if self.dyrank:
                nn.init.zeros_(self.r_q)
                nn.init.normal_(self.s_q, 0.0, 0.02)
                nn.init.zeros_(self.r_kv)
                nn.init.normal_(self.s_kv, 0.0, 0.02)
                nn.init.zeros_(self.r_o)
                nn.init.normal_(self.s_o, 0.0, 0.02)
            else:
                std = 0.01 if fast else 0.02
                nn.init.normal_(self.r_q, 0.0, std)
                nn.init.normal_(self.s_q, 0.0, std)
                nn.init.normal_(self.r_kv, 0.0, std)
                nn.init.normal_(self.s_kv, 0.0, std)
                nn.init.normal_(self.r_o, 0.0, std)
                nn.init.normal_(self.s_o, 0.0, std)

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            lang=None, checkpointing=False,
            src_lang=None,
            incremental=False, incremental_cache=None,
            cu_seqlens=None, max_len=None,
            cu_seqlens_kv=None, max_len_kv=None, stacked_kv=None, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        assert key_value_states is not None
        if not self.fast_attention:
            raise NotImplementedError("Slow Attention by HuggingFace not supported anymore")

        else:
            in_proj_weight_q = self.q_proj.weight
            in_proj_weight_kv = self.proj_weight_kv
            out_proj_weight = self.out_proj.weight

            if self.is_factorized and self.flex_factorize:

                # TODO: mm instead of index select
                n_languages, _rank = self.rm_o.size(0), self.rm_o.size(1)

                # if lang has only 1 element we can do this
                if lang.ndim == 1:

                    rm_q = torch.index_select(self.rm_q, 0, lang.long()).squeeze(0)  # squeeze possible because only 1
                    sm_q = torch.index_select(self.sm_q, 0, lang.long()).squeeze(0)
                    rm_o = torch.index_select(self.rm_o, 0, lang.long()).squeeze(0)
                    sm_o = torch.index_select(self.sm_o, 0, lang.long()).squeeze(0)


                elif lang.ndim == 2:  # for flash attention

                    rm_q = torch.mm(lang, self.rm_q.view(n_languages, _rank * self.rm_q.size(-1))).view(
                        lang.size(0), _rank,
                        self.rm_q.size(-1))
                    sm_q = torch.mm(lang, self.sm_q.view(n_languages, _rank * self.sm_q.size(-1))).view(
                        lang.size(0), _rank,
                        self.sm_q.size(-1))
                    rm_o = torch.mm(lang, self.rm_o.view(n_languages, _rank * self.rm_o.size(-1))).view(
                        lang.size(0), _rank,
                        self.rm_o.size(-1))
                    sm_o = torch.mm(lang, self.sm_o.view(n_languages, _rank * self.sm_o.size(-1))).view(
                        lang.size(0), _rank,
                        self.sm_o.size(-1))

                elif lang.ndim == 3:

                    _len, _bsz = lang.size(0), lang.size(1)
                    _lang = lang.view(_len * _bsz, lang.size(-1))

                    rm_q = torch.mm(_lang, self.rm_q.view(n_languages, _rank * self.rm_q.size(-1))).view(
                        _len, _bsz, _rank, self.rm_q.size(-1))
                    sm_q = torch.mm(_lang, self.sm_q.view(n_languages, _rank * self.sm_q.size(-1))).view(
                        _len, _bsz, _rank, self.sm_q.size(-1))
                    rm_o = torch.mm(_lang, self.rm_o.view(n_languages, _rank * self.rm_o.size(-1))).view(
                        _len, _bsz, _rank, self.rm_o.size(-1))
                    sm_o = torch.mm(_lang, self.sm_o.view(n_languages, _rank * self.sm_o.size(-1))).view(
                        _len, _bsz, _rank, self.sm_o.size(-1))
                else:
                    raise NotImplementedError("Unknown dimension for language IDs")

                if src_lang.ndim == 1:
                    rm_kv = torch.index_select(self.rm_kv, 0, src_lang.long()).squeeze(
                        0)  # squeeze possible because only 1
                    sm_kv = torch.index_select(self.sm_kv, 0, src_lang.long()).squeeze(0)
                elif src_lang.ndim == 2:
                    rm_kv = torch.mm(src_lang, self.rm_kv.view(n_languages, _rank * self.rm_kv.size(-1))).view(
                        src_lang.size(0), _rank,
                        self.rm_kv.size(-1))
                    sm_kv = torch.mm(src_lang, self.sm_kv.view(n_languages, _rank * self.sm_kv.size(-1))).view(
                        src_lang.size(0), _rank,
                        self.sm_kv.size(-1))
                elif src_lang.ndim == 3:
                    _len_src = src_lang.size(0)
                    _src_lang = src_lang.view(_len_src * _bsz, src_lang.size(-1))
                    rm_kv = torch.mm(_src_lang, self.rm_kv.view(n_languages, _rank * self.rm_kv.size(-1))).view(
                        _len_src, _bsz, _rank, self.rm_kv.size(-1))
                    sm_kv = torch.mm(_src_lang, self.sm_kv.view(n_languages, _rank * self.sm_kv.size(-1))).view(
                        _len_src, _bsz, _rank, self.sm_kv.size(-1))

                # if lang has size [T x B x L] we need to do a GEMM

                if hidden_states.ndim == 3:
                    use_time_mask = self.is_decoder
                    bsz, qlen = hidden_states.size(1), hidden_states.size(0)
                    mask = attention_mask
                    low_precision = True  # Use CUDA impl

                    input_lin_q_results = factorize_linear(hidden_states, in_proj_weight_q, self.q_proj.bias, rm_q,
                                                           sm_q)

                    input_lin_kv_results = factorize_linear(key_value_states, in_proj_weight_kv, self.proj_bias_kv,
                                                            rm_kv, sm_kv)

                    recompute = False
                    attn_output, coverage = encdec_attn_bias_compact_func(recompute, self.training, self.num_heads,
                                                                          input_lin_q_results, input_lin_kv_results,
                                                                          attention_mask, self.dropout,
                                                                          incremental, incremental_cache,
                                                                          False, None, None,  # no rotary encodings
                                                                          low_precision, True)

                    attn_output = attn_output.view(qlen, bsz, -1).contiguous()

                    output = factorize_linear(attn_output, out_proj_weight, self.out_proj.bias, rm_o, sm_o)

                    return output, coverage, incremental_cache

                else:
                    """
                    flash attention
                    """
                    assert self.fast_bert_mha is not None
                    assert cu_seqlens is not None
                    assert cu_seqlens_kv is not None
                    assert max_len is not None
                    assert max_len_kv is not None
                    assert incremental == False
                    assert incremental_cache is None

                    total_bsz_q = hidden_states.size(0)
                    total_bsz_kv = key_value_states.size(0)
                    q = factorize_linear(hidden_states, in_proj_weight_q, self.q_proj.bias, rm_q, sm_q)
                    # linear_function(hidden_states, in_proj_weight_q, self.q_proj.bias)

                    kv = factorize_linear(key_value_states, in_proj_weight_kv, self.proj_bias_kv, rm_kv, sm_kv)  #
                    # linear_function(key_value_states, in_proj_weight_kv, self.proj_bias_kv)

                    kv = kv.view(total_bsz_kv, self.num_heads, 2, self.head_dim).transpose(1, 2).contiguous()

                    q = q.view(total_bsz_q, self.num_heads, self.head_dim)

                    dropout_p = self.dropout if self.training else 0.0
                    causal = False
                    softmax_scale = 1.0 / math.sqrt(64)
                    context = self.fast_bert_mha(q, kv, cu_seqlens, cu_seqlens_kv,
                                                 max_len, max_len_kv, dropout_p, softmax_scale, causal, False)

                    context = context.view(-1, self.num_heads * self.head_dim).contiguous()
                    # output = linear_function(context, out_proj_weight, self.out_proj.bias)
                    output = factorize_linear(context, out_proj_weight, self.out_proj.bias, rm_o, sm_o)

                    coverage = None

                    return output, coverage, incremental_cache

            if self.is_factorized:
                if self.multiplicative_factorize:
                    rm_q = torch.index_select(self.rm_q, 0, lang).squeeze(0)  # squeeze possible because only 1
                    sm_q = torch.index_select(self.sm_q, 0, lang).squeeze(0)
                    rm_kv = torch.index_select(self.rm_kv, 0, lang).squeeze(0)  # squeeze possible because only 1
                    sm_kv = torch.index_select(self.sm_kv, 0, lang).squeeze(0)
                    rm_o = torch.index_select(self.rm_o, 0, lang).squeeze(0)
                    sm_o = torch.index_select(self.sm_o, 0, lang).squeeze(0)

                    if self.dyrank:
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

                if self.dyrank:
                    add_factor_q = torch.mm(r_q.t(), s_q)
                    add_factor_kv = torch.mm(r_kv.t(), s_kv)
                    add_factor_out = torch.mm(r_o.t(), s_o)
                else:
                    add_factor_q = torch.bmm(r_q.unsqueeze(-1), s_q.unsqueeze(1)).sum(dim=0)
                    add_factor_kv = torch.bmm(r_kv.unsqueeze(-1), s_kv.unsqueeze(1)).sum(dim=0)
                    add_factor_out = torch.bmm(r_o.unsqueeze(-1), s_o.unsqueeze(1)).sum(dim=0)

                # Has to be additive here
                in_proj_weight_q = in_proj_weight_q + add_factor_q
                in_proj_weight_kv = in_proj_weight_kv + add_factor_kv
                out_proj_weight = out_proj_weight + add_factor_out

            if hidden_states.ndim == 3 and key_value_states.ndim == 3:

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
                                                              False, None, None,  # no rotary encodings
                                                              low_precision, True)

            elif hidden_states.ndim == 2 and key_value_states.ndim == 2:

                assert self.fast_bert_mha is not None
                assert cu_seqlens is not None
                assert cu_seqlens_kv is not None
                assert max_len is not None
                assert max_len_kv is not None
                assert incremental == False
                assert incremental_cache is None

                total_bsz_q = hidden_states.size(0)
                total_bsz_kv = key_value_states.size(0)
                q = linear_function(hidden_states, in_proj_weight_q, self.q_proj.bias)

                kv = linear_function(key_value_states, in_proj_weight_kv, self.proj_bias_kv)

                kv = kv.view(total_bsz_kv, self.num_heads, 2, self.head_dim).transpose(1, 2).contiguous()

                q = q.view(total_bsz_q, self.num_heads, self.head_dim)

                dropout_p = self.dropout if self.training else 0.0
                causal = False
                softmax_scale = 1.0 / math.sqrt(64)
                context = self.fast_bert_mha(q, kv, cu_seqlens, cu_seqlens_kv,
                                             max_len, max_len_kv, dropout_p, softmax_scale, causal, False)

                context = context.view(-1, self.num_heads * self.head_dim).contiguous()
                attn_output = linear_function(context, out_proj_weight, self.out_proj.bias)

                coverage = None

        return attn_output, coverage, incremental_cache


class BartEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class BartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn_name = config.activation_function
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartCrossAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

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

    def forward(
        self,
        # hidden_states: torch.Tensor,
        # attention_mask: Optional[torch.Tensor] = None,
        # encoder_hidden_states: Optional[torch.Tensor] = None,
        # encoder_attention_mask: Optional[torch.Tensor] = None,
        # layer_head_mask: Optional[torch.Tensor] = None,
        # cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        # past_key_value: Optional[Tuple[torch.Tensor]] = None,
        # output_attentions: Optional[bool] = False,
        # use_cache: Optional[bool] = True,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        incremental: Optional[bool] = False,
        incremental_cache=None,
        max_len=None, cu_seqlens=None,
        max_len_kv=None, cu_seqlens_kv=None, **kwargs
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (:obj:`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            incremental=incremental, incremental_cache=incremental_cache,
            cu_seqlens=cu_seqlens, max_len=max_len
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            attention_input = hidden_states

            hidden_states, cross_attn_weights, incremental_cache = self.encoder_attn(
                hidden_states=attention_input,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                incremental=incremental, incremental_cache=incremental_cache,
                cu_seqlens=cu_seqlens, max_len=max_len,
                cu_seqlens_kv=cu_seqlens_kv, max_len_kv=max_len_kv
            )

            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = hidden_states + residual

        # Fully Connected
        residual = hidden_states

        # todo: replace this with fused function
        if self.fused and hidden_states.is_cuda:
            with autocast(enabled=False):
                weights = [self.fc1.weight.half(), self.fc2.weight.half()]
                biases = [self.fc1.bias.half(), self.fc2.bias.half()]

                seq_len, bsz, hidden_size = hidden_states.size(0), hidden_states.size(1), hidden_states.size(2)
                dropout = self.activation_dropout if self.training else 0.0
                hidden_states = self.fused_function(dropout, False, hidden_states.half().view(seq_len * bsz, -1),
                                        *weights, *biases).type_as(hidden_states)
                hidden_states = hidden_states.view(seq_len, bsz, hidden_size)
        else:
            hidden_states = self.activation_fn(self.fc1(hidden_states))
            hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
            hidden_states = self.fc2(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs



class BartPretrainedModel(PreTrainedModel):
    config_class = BartConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = [r"encoder\.version", r"decoder\.version"]

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
        if isinstance(module, (BartDecoder, BartEncoder)):
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


class PretrainedBartModel(BartPretrainedModel):
    def __init_subclass__(self):
        warnings.warn(
            "The class `PretrainedBartModel` has been depreciated, please use `BartPretrainedModel` instead.",
            FutureWarning,
        )


BART_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.BartConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BART_GENERATION_EXAMPLE = r"""
    Summarization example::

        >>> from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

        >>> model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

        >>> ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')

        >>> # Generate Summary
        >>> summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        >>> print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

    Mask filling example::

        >>> from transformers import BartTokenizer, BartForConditionalGeneration
        >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        >>> TXT = "My friends are <mask> but they eat too many carbs."

        >>> model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        >>> input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
        >>> logits = model(input_ids).logits

        >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
        >>> probs = logits[0, masked_index].softmax(dim=0)
        >>> values, predictions = probs.topk(5)

        >>> tokenizer.decode(predictions).split()
"""

BART_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using :class:`~transformers.BartTokenizer`. See
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

            Indices can be obtained using :class:`~transformers.BartTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are decoder input IDs? <../glossary.html#decoder-input-ids>`__

            Bart uses the :obj:`eos_token_id` as the starting token for :obj:`decoder_input_ids` generation. If
            :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).

            For translation and summarization training, :obj:`decoder_input_ids` should be provided. If no
            :obj:`decoder_input_ids` is provided, the model will create this tensor by shifting the :obj:`input_ids` to
            the right for denoising pre-training following the paper.
        decoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.

            If you want to change padding behavior, you should read :func:`modeling_bart._prepare_decoder_inputs` and
            modify to your needs. See diagram 1 in `the paper <https://arxiv.org/abs/1910.13461>`__ for more
            information on the default strategy.
        head_mask (:obj:`torch.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in ``[0,
            1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

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


class BartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
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

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
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

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

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

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
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
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class BartDecoder(BartPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, opt, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = opt.dropout
        self.layerdrop = opt.death_rate / 2
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.dec_pretrained_model = 'bart'

        config.attention_dropout = opt.attn_dropout if opt.attn_dropout > 0 else opt.dropout
        config.activation_dropout = opt.ffn_dropout if opt.ffn_dropout > 0 else opt.dropout

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )

        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()
        self.gradient_checkpointing = False

        self.model_size = config.d_model
        self.switchout = 0.0
        self.config.bert_hidden_size = config.d_model

    @property
    def word_lut(self):
        return self.embed_tokens

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
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
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
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

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

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

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        bsz = input_ids.size(0)
        qlen = input_ids.size(1)
        klen = qlen

        # if attention_mask is None:
        padding_mask = attention_mask
        attention_mask = torch.triu(
            inputs_embeds.new_ones(qlen, klen), diagonal=1).bool()

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        assert encoder_hidden_states.size(1) == hidden_states.size(
            0), "batch size has to match between Enc and Dec states"

        if self.fast_bert_mha is not None and hidden_states.dtype == torch.half:
            can_run_fast_bert_mha = True

            # lets unpad both hidden_states and context states
            if padding_mask is None:
                padding_mask = input_ids.new_zeros(bsz, qlen)
            padding_mask = padding_mask.contiguous().long()
            lengths = (1 - padding_mask).sum(dim=1)
            lengths = lengths.cpu().tolist()  # list of lengths for B seqs
            hidden_states = hidden_states.view(-1, hidden_states.size(-1))
            non_pad_indices = torch.nonzero(padding_mask.view(-1).ne(1)).squeeze(1)
            hidden_states = hidden_states.index_select(0, non_pad_indices)
            max_len = max(lengths)
            # cumulative sequence lengths (required input for fmha)
            a = torch.tensor(np.array([0] + lengths), dtype=torch.int32)
            cu_seqlens = torch.cumsum(a, 0).to(dtype=torch.int32, device=hidden_states.device)
            non_pad_indices_q = non_pad_indices

            # unpad the context
            encoder_hidden_states = encoder_hidden_states.transpose(0, 1).contiguous()
            padding_mask = encoder_attention_mask
            context_len = encoder_hidden_states.size(1)
            if padding_mask is None:
                padding_mask = input_ids.new_zeros(bsz, context_len)
            padding_mask = padding_mask.long()
            lengths = (1 - padding_mask).sum(dim=1)
            lengths = lengths.cpu().tolist()  # list of lengths for B seqs
            encoder_hidden_states = encoder_hidden_states.view(-1, encoder_hidden_states.size(-1))
            non_pad_indices = torch.nonzero(padding_mask.view(-1).ne(1)).squeeze(1)
            encoder_hidden_states = encoder_hidden_states.index_select(0, non_pad_indices)

            max_len_kv = max(lengths)
            # cumulative sequence lengths (required input for fmha)
            a = torch.tensor(np.array([0] + lengths), dtype=torch.int32)
            cu_seqlens_kv = torch.cumsum(a, 0).to(dtype=torch.int32, device=encoder_hidden_states.device)

        else:
            max_len, cu_seqlens = None, None
            max_len_kv, cu_seqlens_kv = None, None
            non_pad_indices_q = None
            can_run_fast_bert_mha = False

            hidden_states = hidden_states.transpose(0, 1).contiguous()

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Stochastic Layer (only applicable when not predicting language or idx > 0)
            if not (self.predict_language > 0 and idx == 0):
                dropout_probability = random.uniform(0, 1)
                if self.training and (dropout_probability < self.layerdrop):
                    continue

            layer_outputs, _ = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                max_len=max_len, cu_seqlens=cu_seqlens,
                max_len_kv=max_len_kv, cu_seqlens_kv=cu_seqlens_kv,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
            if v is not None
        )

    def step(self, input, decoder_state, **kwargs):

        # context is stored in the decoder state in [T B H] format
        encoder_hidden_states = decoder_state.context

        buffers = decoder_state.attention_buffers
        buffering = decoder_state.buffering

        input_ids = input
        input_shape = input_ids.size()
        time_step = input.size(1)

        input_ = input
        if buffering:
            # use the last value of input to continue decoding
            if input.size(1) > 1 and len(buffers) > 0:
                # if buffers has not been initilized and we have > 1 input length data
                # then its a prefix decoding step
                input_ = input[:, -1:]
                past_key_values_length = input.size(1) - 1
            else:
                past_key_values_length = 0
        else:
            past_key_values_length = 0

        inputs_embeds = self.embed_tokens(input_) * self.embed_scale

        # TODO: adding buffers
        qlen = input_ids.size(1)
        klen = qlen
        attention_mask = torch.triu(
            inputs_embeds.new_ones(qlen, klen), diagonal=1).bool()

        if input.size(1) > 1 and len(buffers) > 0:
            attention_mask = attention_mask[-1:, :]

        encoder_attention_mask = decoder_state.src_mask
        if not self.layers[0].encoder_attn.fast_attention:
            raise NotImplementedError
        else:
            encoder_attention_mask = encoder_attention_mask.bool()
        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = hidden_states.transpose(0, 1)

        # decoder layers
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
            )

            # layer_outputs = decoder_layer(
            #     hidden_states,
            #     attention_mask=attention_mask,
            #     encoder_hidden_states=encoder_hidden_states,
            #     encoder_attention_mask=encoder_attention_mask,
            #     layer_head_mask=(head_mask[idx] if head_mask is not None else None),
            #     cross_attn_layer_head_mask=(
            #         cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
            #     ),
            #     past_key_value=None,
            #     output_attentions=True,
            #     use_cache=False,
            # )
            hidden_states = layer_outputs[0]

        output = hidden_states[-1].unsqueeze(0)
        coverage = hidden_states.new(hidden_states.size(1), 1, encoder_hidden_states.size(0)).zero_()

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = output
        output_dict['coverage'] = coverage
        output_dict['context'] = encoder_hidden_states

        return output_dict

