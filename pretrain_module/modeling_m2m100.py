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
""" PyTorch M2M100 model. """

import math
import random
from typing import Optional, Tuple
import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from collections import defaultdict

from .activations import ACT2FN

from .modeling_outputs import (
    BaseModelOutput,
)
from .modeling_utils import PreTrainedModel
# from ...utils import logging
from .configuration_m2m100 import M2M100Config
from onmt.modules.layer_norm import LayerNorm
from onmt.modules.optimized.self_attention_func import self_attn_func
from onmt.modules.optimized.encdec_attention_func_bias import encdec_attn_bias_func
from onmt.modules.dropout import embedded_dropout
from onmt.modules.optimized.dropout_add import fused_dropout_add
from onmt.modules.optimized.linear import linear_function
from torch.cuda.amp import custom_fwd, custom_bwd

_CONFIG_FOR_DOC = "M2M100Config"
_TOKENIZER_FOR_DOC = "M2M100Tokenizer"
_CHECKPOINT_FOR_DOC = "facebook/m2m100_418M"

M2M_100_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/m2m100_418M",
    # See all M2M100 models at https://huggingface.co/models?filter=m2m_100
]


#
# # Copied from transformers.models.bart.modeling_bart.shift_tokens_right
# def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
#     """
#     Shift input ids one token to the right.
#     """
#     shifted_input_ids = input_ids.new_zeros(input_ids.shape)
#     shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
#     shifted_input_ids[:, 0] = decoder_start_token_id
#
#     if pad_token_id is None:
#         raise ValueError("self.model.config.pad_token_id has to be defined.")
#     # replace possible -100 values in labels by `pad_token_id`
#     shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
#
#     return shifted_input_ids
#
#
# # Copied from transformers.models.bart.modeling_bart._make_causal_mask
# def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
#     """
#     Make causal mask used for bi-directional self-attention.
#     """
#     bsz, tgt_len = input_ids_shape
#     mask = torch.full((tgt_len, tgt_len), float("-inf"))
#     mask_cond = torch.arange(mask.size(-1))
#     mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
#     mask = mask.to(dtype)
#
#     if past_key_values_length > 0:
#         mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
#     return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
#
#
# # Copied from transformers.models.bart.modeling_bart._expand_mask
# def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
#     """
#     Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
#     """
#     bsz, src_len = mask.size()
#     tgt_len = tgt_len if tgt_len is not None else src_len
#
#     expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
#
#     inverted_mask = 1.0 - expanded_mask
#
#     return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


class M2M100SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # in forward put the weights on the correct dtype and device of the param
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        self.weights = nn.Parameter(emb_weights)
        self.weights.requires_grad = False
        self.weights.detach_()

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb

    @torch.no_grad()
    def forward(
            self, input_ids: torch.Tensor = None, inputs_embeds: torch.Tensor = None, past_key_values_length: int = 0
    ):
        if input_ids is not None:
            bsz, seq_len = input_ids.size()
            # Create the position ids from the input token ids. Any padded tokens remain padded.
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
                input_ids.device
            )
        else:
            bsz, seq_len = inputs_embeds.size()[:-1]
            position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # expand embeddings if needed
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape).contiguous()


from .modeling_mbart import MBartAttention as M2M100Attention
from .modeling_mbart import MBartCrossAttention as M2M100CrossAttention
from .modeling_mbart import index_copy


# Copied from transformers.models.mbart.modeling_mbart.MBartEncoderLayer with MBart->M2M100
class M2M100EncoderLayer(nn.Module):
    def __init__(self, config: M2M100Config):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = M2M100Attention(
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
            max_len=-1, cu_seqlens=None, **kwargs
    ):
        """
        :param hidden_states:
        :param attention_mask:
        :param layer_head_mask:
        :param output_attentions:
        :param max_len:
        :param cu_seqlens:
        :param kwargs:
        :return:
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
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        if self.fused and hidden_states.is_cuda:
            weights = [self.fc1.weight, self.fc2.weight]
            biases = [self.fc1.bias, self.fc2.bias]

            dropout = self.activation_dropout if self.training else 0.0
            hidden_states = self.fused_function(dropout, False, hidden_states, *weights, *biases).type_as(hidden_states)
        else:
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


# Copied from transformers.models.mbart.modeling_mbart.MBartDecoderLayer with MBart->M2M100
class M2M100DecoderLayer(nn.Module):
    def __init__(self, config: M2M100Config):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = M2M100Attention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = M2M100CrossAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
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

        self.is_factorized = False
        self.multiplicative_factorize = False
        self.fast_factorize = False
        self.ffn_dim = config.decoder_ffn_dim

        self.n_languages = -1
        self.has_adapter = False
        self.adapter_location = -1

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
            output_attentions: Optional[bool] = False,
            incremental: Optional[bool] = False,
            incremental_cache=None,
            lang=None, mixture=None, **kwargs
    ):
        """
        :param hidden_states:
        :param attention_mask:
        :param encoder_hidden_states:
        :param encoder_attention_mask:
        :param output_attentions:
        :param incremental:
        :param incremental_cache:
        :param lang:
        :param mixture:
        :return:
        """
        if incremental and incremental_cache is None:
            incremental_cache = dict()

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # print(hidden_states.size(), attention_mask.size(), encoder_hidden_states.size())

        # Self Attention
        hidden_states, self_attn_weights, incremental_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            incremental=incremental, incremental_cache=incremental_cache,
            lang=lang
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            hidden_states, cross_attn_weights, incremental_cache = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                incremental=incremental, incremental_cache=incremental_cache,
                lang=lang
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        # hidden_states = self.activation_fn(self.fc1(hidden_states))
        # hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # hidden_states = self.fc2(hidden_states)
        in_weight, out_weight, in_bias, out_bias = self.get_mlp_weights(lang=lang, mixture=mixture)
        hidden_states = self.call_mlp(hidden_states, in_weight, out_weight, in_bias, out_bias,
                                      self.activation_fn, self.activation_dropout, self.training,
                                      self.fused, self.fused_function)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if self.has_adapter:
            residual = hidden_states
            if self.adapter_location == 1:
                assert lang is not None or mixture is not None
                hidden_states = self.adapter(hidden_states, lang=lang, mixture=mixture)

            hidden_states.add_(residual)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs, incremental_cache


class M2M100PreTrainedModel(PreTrainedModel):
    config_class = M2M100Config
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
        if isinstance(module, (M2M100Decoder, M2M100Encoder)):
            module.gradient_checkpointing = value


class M2M100Encoder(M2M100PreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`M2M100EncoderLayer`].

    Args:
        config: M2M100Config
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: M2M100Config, opt, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        config.dropout = opt.residual_dropout if opt.residual_dropout > 0 else opt.dropout
        config.attention_dropout = opt.attn_dropout
        config.activation_dropout = opt.ffn_dropout if opt.ffn_dropout > 0 else opt.dropout
        config.layerdrop = opt.death_rate

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

        self.embed_positions = M2M100SinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
            self.padding_idx,
        )
        self.layers = nn.ModuleList([M2M100EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        self.word_dropout = opt.word_dropout
        # Initialize weights and apply final processing
        # self.post_init()
        from onmt.modules.optimized.fast_mha import fast_bert_mha
        self.fast_bert_mha = fast_bert_mha

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`M2M100Tokenizer`]. See
                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`]
                for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert `input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

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
            inputs_embeds = embedded_dropout(self.embed_tokens, input_ids,
                                             dropout=self.word_dropout if self.training else 0)
            inputs_embeds = inputs_embeds * self.embed_scale

        embed_pos = self.embed_positions(input_ids, inputs_embeds)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # # expand attention_mask
        # if attention_mask is not None:
        #     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        #     attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        can_run_fast_bert_mha = False
        # check if fast bert mha can be run
        seq_len = hidden_states.size(1)
        bsz = hidden_states.size(0)
        sm = torch.cuda.get_device_capability()
        total_bsz = 0

        if torch.is_autocast_enabled():
            try:
                hidden_states = torch.cuda.amp.autocast_mode._cast(hidden_states, torch.get_autocast_gpu_dtype())
            except AttributeError:
                hidden_states = torch.cuda.amp.autocast_mode._cast(hidden_states, torch.half)

        # only run this when seq_len <= 512 and sm = 80/86 and type = half
        # if self.fast_bert_mha is not None and (seq_len <= 512 and bsz >= 4 and sm[0] == 8 and sm[1] in [0, 6]) \
        #         and hidden_states.dtype == torch.half:
        #     can_run_fast_bert_mha = True
        #
        #     x = hidden_states
        #     padding_mask = attention_mask  # [B x T]
        #     # masked positions = 1 so to compute length we need the (1 -)
        #     if padding_mask is None:
        #         padding_mask = x.new_zeros(bsz, seq_len)
        #     padding_mask = padding_mask.long()
        #     lengths = (1 - padding_mask).sum(dim=1)
        #     lengths = lengths.cpu().tolist()  # list of lengths for B seqs
        #
        #     x = x.view(-1, x.size(-1))
        #     non_pad_indices = torch.nonzero(padding_mask.view(-1).ne(1)).squeeze(1)
        #     hidden_states = x.index_select(0, non_pad_indices)
        #
        #     max_len = max(lengths)
        #     # cumulative sequence lengths (required input for fmha)
        #     a = torch.tensor(np.array([0] + lengths), dtype=torch.int32)
        #     cu_seqlens = torch.cumsum(a, 0).to(dtype=torch.int32, device=x.device)
        # else:
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
                )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

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


class M2M100Decoder(M2M100PreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`M2M100DecoderLayer`]

    Args:
        config: M2M100Config
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: M2M100Config, opt, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # override
        config.dropout = opt.residual_dropout if opt.residual_dropout > 0 else opt.dropout
        config.activation_dropout = opt.ffn_dropout if opt.ffn_dropout > 0 else opt.dropout
        config.attention_dropout = opt.attn_dropout
        self.dropout = config.dropout

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = M2M100SinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
        )
        self.layers = nn.ModuleList([M2M100DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layer_norm = LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        # self.post_init()

        self.model_size = config.d_model
        self.switchout = 0.0
        # self.word_lut = self.embed_tokens
        self.config.bert_hidden_size = config.d_model
        self.layerdrop = opt.death_rate_decoder
        self.dec_pretrained_model = 'm2m100'
        self.embed_tokens.weight.requires_grad = False
        self.word_dropout = opt.word_dropout

    def add_factorize(self, n_languages, rank=4, multiplicative=False, fast=False):

        for layer in self.layers:
            layer.add_factorize(n_languages, rank=rank, multiplicative=multiplicative, fast=fast)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            lang=None, mixture=None,
            **kwargs
    ):
        """
        :param input_ids:
        :param attention_mask:
        :param encoder_hidden_states:
        :param encoder_attention_mask:
        :param inputs_embeds:
        :param output_attentions:
        :param output_hidden_states:
        :param incremental:
        :param incremental_cache:
        :param lang:
        :param mixture:
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

        # past_key_values_length
        past_key_values_length = 0

        if inputs_embeds is None:
            inputs_embeds = embedded_dropout(self.embed_tokens, input_ids,
                                             dropout=self.word_dropout if self.training else 0)
            inputs_embeds = inputs_embeds * self.embed_scale

        # embed positions
        positions = self.embed_positions(input_ids, inputs_embeds, past_key_values_length)

        # create autoregressive mask
        qlen = input_ids.size(1)
        klen = qlen
        attention_mask = torch.triu(
            inputs_embeds.new_ones(qlen, klen), diagonal=1).bool()

        hidden_states = inputs_embeds + positions

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = hidden_states.transpose(0, 1).contiguous()

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
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
                all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return tuple(
            v
            for v in [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions]
            if v is not None
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

        input_ = input  # input for the current time step
        if buffering:
            # use the last value of input to continue decoding
            if input.size(1) > 1:
                input_ = input[:, -1:]
            past_key_values_length = input.size(1) - 1
        else:
            past_key_values_length = 0

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        qlen = input_ids.size(1)
        klen = qlen
        attention_mask = torch.triu(
            inputs_embeds.new_ones(qlen, klen), diagonal=1).bool()

        encoder_attention_mask = decoder_state.src_mask
        # the past_key_values_length probably gives us the last time step

        # print(input_ids.size(), inputs_embeds.size())
        positions = self.embed_positions(input_ids, inputs_embeds, 0)

        hidden_states = inputs_embeds + positions
        if buffering:
            hidden_states = hidden_states[:, -1:, :]
            attention_mask = attention_mask[-1:, :]

        hidden_states = hidden_states.transpose(0, 1).contiguous()

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
