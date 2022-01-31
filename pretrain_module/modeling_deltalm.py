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
from .modeling_mbart import MBartLearnedPositionalEmbedding, MBartAttention, MBartCrossAttention
# from ...utils import logging
# from .configuration_bart import BartConfig
import onmt
from collections import defaultdict
from .configuration_deltalm import DeltaLMConfig

_CHECKPOINT_FOR_DOC = "facebook/mbart-large-cc25"
_CONFIG_FOR_DOC = "DeltaLMConfig"
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


class DeltaLMEncoderLayer(nn.Module):
    def __init__(self, config: DeltaLMConfig):
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
        self.normalize_before = config.normalize_before

        # Optimization
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
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                where padding elements are indicated by very large negative values.
            output_attentions (:obj:`bool`, `optional`):

                :param output_attentions: Whether or not to return the attentions tensors of all attention layers.
                :param attention_mask:  `(batch, src_len)`
                :param hidden_states:  `(seq_len, batch, embed_dim)`
                :param cu_seqlens:
                :param max_len:
        """
        residual = hidden_states

        if self.normalize_before:
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

        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Second block (FFN)
        residual = hidden_states

        if self.normalize_before:
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

        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class DeltaLMDecoderLayer(nn.Module):
    def __init__(self, config: DeltaLMConfig):
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

        self.normalize_before = config.normalize_before

        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.fc3 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc4 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.ffn_layer_norm = LayerNorm(self.embed_dim)

        self.is_factorized = False
        self.multiplicative_factorize = False
        self.fast_factorize = False
        self.ffn_dim = config.decoder_ffn_dim

        self.n_languages = -1
        self.has_adapter = False
        self.adapter_location = -1

        # Optimization
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

        self.r2_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.ffn_dim))
        self.s2_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.embed_dim))
        self.r2_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.embed_dim))
        self.s2_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.ffn_dim))

        nn.init.normal_(self.r2_i, 0.0, 0.02)
        nn.init.normal_(self.s2_i, 0.0, 0.02)
        nn.init.normal_(self.r2_o, 0.0, 0.02)
        nn.init.normal_(self.s2_o, 0.0, 0.02)

        if multiplicative:
            rank = rank if fast else 1
            self.rm2_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.ffn_dim))
            self.sm2_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.embed_dim))
            self.rm2_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.embed_dim))
            self.sm2_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.ffn_dim))

            constant = math.sqrt(1.0 / rank) if fast else 1
            nn.init.constant_(self.rm2_i, constant)
            nn.init.constant_(self.sm2_i, constant)
            nn.init.constant_(self.rm2_o, constant)
            nn.init.constant_(self.sm2_o, constant)

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

    def get_interleaved_mlp_weights(self, lang=None, mixture=None):

        in_weight = self.fc3.weight
        out_weight = self.fc4.weight
        in_bias = self.fc3.bias
        out_bias = self.fc4.bias

        if lang is not None:
            assert mixture is None

            if self.is_factorized:
                if self.multiplicative_factorize:
                    rm_i = torch.index_select(self.rm2_i, 0, lang).squeeze(0)  # squeeze possible because only 1
                    sm_i = torch.index_select(self.sm2_i, 0, lang).squeeze(0)
                    rm_o = torch.index_select(self.rm2_o, 0, lang).squeeze(0)
                    sm_o = torch.index_select(self.sm2_o, 0, lang).squeeze(0)

                    if self.fast_factorize:
                        mul_factor_in = torch.mm(rm_i.t(), sm_i)
                        mul_factor_out = torch.mm(rm_o.t(), sm_o)
                    else:
                        mul_factor_in = torch.bmm(rm_i.unsqueeze(-1), sm_i.unsqueeze(1)).sum(dim=0)
                        mul_factor_out = torch.bmm(rm_o.unsqueeze(-1), sm_o.unsqueeze(1)).sum(dim=0)

                    in_weight = in_weight * mul_factor_in
                    out_weight = out_weight * mul_factor_out

                r_i = torch.index_select(self.r2_i, 0, lang).squeeze(0)
                s_i = torch.index_select(self.s2_i, 0, lang).squeeze(0)
                r_o = torch.index_select(self.r2_o, 0, lang).squeeze(0)
                s_o = torch.index_select(self.s2_o, 0, lang).squeeze(0)

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
            sub_encoder_hidden_states: Optional[torch.Tensor] = None,
            sub_encoder_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
            incremental: Optional[bool] = False,
            incremental_cache=None,
            lang=None, mixture=None
    ):
        """
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
        :param mixture:
        :return:
        """
        if incremental and incremental_cache is None:
            incremental_cache = dict()

        # hidden_states = hidden_states.transpose(0, 1).contiguous()
        residual = hidden_states

        if self.normalize_before:
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
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        # hidden_states = fused_dropout_add(hidden_states, residual, self.dropout, self.training)
        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        ###############################################

        # Interleaved FFN block
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.ffn_layer_norm(hidden_states)

        in_weight, out_weight, in_bias, out_bias = self.get_interleaved_mlp_weights(lang=lang, mixture=mixture)
        hidden_states = self.call_mlp(hidden_states, in_weight, out_weight, in_bias, out_bias,
                                      self.activation_fn, self.activation_dropout, self.training,
                                      self.fused, self.fused_function)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.ffn_layer_norm(hidden_states)

        ###############################################

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            if self.normalize_before:
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
                lang=lang, mixture=mixture
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
            hidden_states = residual + hidden_states
            # hidden_states = fused_dropout_add(hidden_states, residual, self.dropout, self.training)

            if not self.normalize_before:
                hidden_states = self.encoder_attn_layer_norm(hidden_states)

        ###############################################
        # Fully Connected
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        in_weight, out_weight, in_bias, out_bias = self.get_mlp_weights(lang=lang, mixture=mixture)
        hidden_states = self.call_mlp(hidden_states, in_weight, out_weight, in_bias, out_bias,
                                      self.activation_fn, self.activation_dropout, self.training,
                                      self.fused, self.fused_function)

        # hidden_states = fused_dropout_add(hidden_states, residual, self.dropout, self.training)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.has_adapter:
            residual = hidden_states
            if self.adapter_location == 1:
                assert lang is not None or mixture is not None
                hidden_states = self.adapter(hidden_states, lang=lang, mixture=mixture)

            hidden_states.add_(residual)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if contrastive_loss is not None:
            # print("Return contrastive loss here,", contrastive_loss.size())
            outputs += (contrastive_loss, )

        return outputs, incremental_cache


class MBartPreTrainedModel(PreTrainedModel):
    config_class = DeltaLMConfig
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




class DeltaLMEncoder(MBartPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`MBartEncoderLayer`.

    Args:
        config: DeltaLMConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: DeltaLMConfig, opt, embed_tokens: Optional[nn.Embedding] = None):
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

        # TODO: check this number
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = MBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([DeltaLMEncoderLayer(config) for _ in range(config.encoder_layers)])

        # this applies at the beginning of the encoder stack
        if config.normalize_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = nn.Identity()

        # this applies at the end of the encoder stack
        if config.normalize_before:
            self.layer_norm = LayerNorm(config.d_model)
        else:
            self.layer_norm = nn.Identity()

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
            output_hidden_states=None
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

        # print(self.embed_scale, self.layernorm_embedding, self.layer_norm)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            bsz, seq_len = input_ids.size(0), input_ids.size(1)
            input_shape = torch.Size([bsz, seq_len])

        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = embedded_dropout(self.embed_tokens, input_ids,
                                             dropout=self.word_dropout if self.training else 0)
            inputs_embeds = inputs_embeds * self.embed_scale

        inputs_embeds = inputs_embeds.view(bsz, seq_len, -1)

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
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
        if self.fast_bert_mha and (seq_len <= 512 and bsz >= 4 and sm[0] == 8 and sm[1] in [0, 6]) \
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


class DeltaLMDecoder(MBartPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`MBartDecoderLayer`
\
    Args:
        config: DeltaLMConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: DeltaLMConfig, opt, embed_tokens: Optional[nn.Embedding] = None):
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
        self.layers = nn.ModuleList([DeltaLMDecoderLayer(config) for _ in range(config.decoder_layers)])

        # applies before the decoder stack
        if config.normalize_embedding:
            self.layernorm_embedding = LayerNorm(config.d_model)
        else:
            self.layernorm_embedding = nn.Identity()

        # applies after the decoder stack
        if config.normalize_before:
            self.layer_norm = LayerNorm(config.d_model)
        else:
            self.layer_norm = nn.Identity()

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

            if not opt.freeze_cross_attention:
                # but we need to enable the cross attention
                for layer in self.layers:
                    for p in layer.encoder_attn.parameters():
                        p.requires_grad = True
                    for p in layer.encoder_attn_layer_norm.parameters():
                        p.requires_grad = True

        if opt.multilingual_factorized_weights_decoder:
            print("[INFO] Factorizing MBART model into %d languages" % opt.n_languages)
            self.add_factorize(opt.n_languages, rank=opt.mfw_rank,
                               multiplicative=opt.mfw_multiplicative,
                               fast=opt.fast_factorize)

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

    def add_factorize(self, n_languages, rank=4, multiplicative=False, fast=False):

        for layer in self.layers:
            layer.add_factorize(n_languages, rank=rank, multiplicative=multiplicative, fast=fast)

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
            lang=None, mixture=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        """
        :param input_ids:
        :param attention_mask:
        :param encoder_hidden_states:
        :param encoder_attention_mask:
        :param sub_encoder_hidden_states:
        :param sub_encoder_attention_mask:
        :param inputs_embeds:
        :param incremental:
        :param incremental_cache:
        :param lang:
        :param mixture:
        :param output_attentions:
        :param output_hidden_states:
        :param return_dict:
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
                mixture=mixture
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