# coding=utf-8
# Copyright 2022 The OpenAI Authors and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Whisper model."""

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, Parameter

from .whisper_config import WhisperConfig
from pretrain_module.activations import ACT2FN
from onmt.modules.optimized.linear import linear_function, factorize_linear
from onmt.modules.optimized.self_attention_func import self_attn_func, self_attn_compact_func


# TODO: copy the whisper code for Encoder and prepare for Decoder

def sinusoids(length: int, channels: int, max_timescale: float = 10000) -> torch.Tensor:
    """Returns sinusoids for positional embedding"""
    if channels % 2 != 0:
        raise ValueError(
            f"Number of channels has to be divisible by 2 for sinusoidal positional embeddings, got {channels} channels."
        )
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length).view(-1, 1) * inv_timescales.view(1, -1)
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _compute_mask_indices(
        shape: Tuple[int, int],
        mask_prob: float,
        mask_length: int,
        attention_mask: Optional[torch.LongTensor] = None,
        min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask


class WhisperPositionalEmbedding(nn.Embedding):
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__(num_positions, embedding_dim)

    def forward(self, input_ids, past_key_values_length=0, position_ids=None):
        if position_ids is None:
            return self.weight[past_key_values_length: past_key_values_length + input_ids.shape[1]]
        else:
            return self.weight[position_ids]


# TODO: split this into self attention and cross attention
# and only use flash attention for the former
class WhisperAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
            is_causal: bool = False,
            layer_idx: Optional[int] = None,
            config: Optional[WhisperConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        #
        # if layer_idx is None and is_decoder:
        #     logger.warning_once(
        #         f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "
        #         "will cause errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
        #         "when creating this class."
        #     )
        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.fast_attention = False
        from onmt.modules.optimized.flash_mha import flash_bert_mha
        self.fast_bert_mha = flash_bert_mha

    # Copied from transformers.models.bart.modeling_bart.BartAttention._shape with BART->whisper
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

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

        head_dim = self.head_dim
        heads = self.num_heads
        input_dim = self.embed_dim

        weight_ = weight_.reshape(3 * head_dim * heads, input_dim).view(3, heads, head_dim, input_dim).transpose(0, 1). \
            reshape(-1, input_dim)

        weight_t = torch.Tensor(3 * input_dim, input_dim)
        weight_t.copy_(weight_)
        self.proj_weight = Parameter(weight_t)
        self.proj_bias = None

        self.proj_weight.requires_grad = self.q_proj.weight.requires_grad
        del self.q_proj, self.k_proj, self.v_proj

    # the main difference here is the cache position
    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value=None,
            attention_mask: Optional[torch.Tensor] = None,
            cu_seqlens=None, max_len=None,
            incremental=False, incremental_cache=None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # """Input shape: Batch x Time x Channel"""
        """
        Args:
            hidden_states:
            key_value_states:
            past_key_value:
            attention_mask:
            cu_seqlens:
            max_len:
            lang:
            atb:
            incremental:
            incremental_cache:
            **kwargs:

        Returns:

        """

        if not self.fast_attention:
            raise NotImplementedError("Slow attention by HuggingFace is deprecated.")

        in_proj_weight = self.proj_weight
        out_proj_weight = self.out_proj.weight

        if hidden_states.ndim == 3:
            use_time_mask = self.is_decoder
            qlen, klen = hidden_states.size(0), hidden_states.size(0)
            mask = attention_mask
            low_precision = True  # Use CUDA impl?
            checkpointing = False

            attn_output, coverage = self_attn_func(use_time_mask, self.training, self.num_heads, hidden_states,
                                                   in_proj_weight, out_proj_weight,
                                                   self.proj_bias, self.out_proj.bias,
                                                   mask, self.dropout,
                                                   False, None,
                                                   incremental, incremental_cache, low_precision,
                                                   True, checkpointing)

            attn_output = attn_output

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
            qkv = qkv.view(total_bsz, self.num_heads, 3, self.head_dim).transpose(1, 2).contiguous()

            dropout_p = self.dropout if self.training else 0.0
            causal = self.is_decoder
            softmax_scale = 1.0 / math.sqrt(self.head_dim)
            context = self.fast_bert_mha(qkv, cu_seqlens, max_len, dropout_p, softmax_scale, causal, False)
            coverage = None

            context = context.view(-1, self.num_heads * self.head_dim).contiguous()
            outputs = linear_function(context, out_proj_weight, self.out_proj.bias)

            attn_output = outputs

        return attn_output, coverage, incremental_cache

        # at the point of writing this code, the [TxBxH] scheme is possibly a bit worse than previously
        # due to the fused function in Torch is faster than manual matrix multiplication and softmax
        # TODO Quan: testing torch _fused_forward and _fused_backward vs MM + softmax

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        # is_cross_attention = key_value_states is not None
        # tgt_len, bsz, _ = hidden_states.size()
        #
        # # get query proj
        # query_states = self._shape(self.q_proj(hidden_states) * self.scaling, tgt_len, bsz)
        #
        # # this is only for cross attention
        # if past_key_value is not None:
        #     is_updated = past_key_value.is_updated.get(self.layer_idx)
        #     if is_cross_attention:
        #         # after the first generated id, we can subsequently re-use all key/value_states from cache
        #         past_key_value.is_updated[self.layer_idx] = True
        #         past_key_value = past_key_value.cross_attention_cache
        #     else:
        #         past_key_value = past_key_value.self_attention_cache
        #
        # # use key_value_states if cross attention
        # current_states = key_value_states if key_value_states is not None else hidden_states
        # if is_cross_attention and past_key_value and is_updated:
        #     # reuse k,v, cross_attentions
        #     key_states = past_key_value.key_cache[self.layer_idx]
        #     value_states = past_key_value.value_cache[self.layer_idx]
        # else:
        #     key_states = self._shape(self.k_proj(current_states), -1, bsz)
        #     value_states = self._shape(self.v_proj(current_states), -1, bsz)
        #     if past_key_value is not None:
        #         # save all key/value_states to cache to be re-used for fast auto-regressive generation
        #         cache_position = cache_position if not is_cross_attention else None
        #         key_states, value_states = past_key_value.update(
        #             key_states, value_states, self.layer_idx, {"cache_position": cache_position}
        #         )
        #
        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        #
        # if attention_mask is not None:  # no matter the length, we just slice it
        #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        #     attn_weights = attn_weights + causal_mask
        #
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        #
        # if layer_head_mask is not None:
        #     if layer_head_mask.size() != (self.num_heads,):
        #         raise ValueError(
        #             f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
        #             f" {layer_head_mask.size()}"
        #         )
        #     attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights
        #
        # attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # attn_output = torch.matmul(attn_probs, value_states)
        #
        # if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
        #         f" {attn_output.size()}"
        #     )
        #
        # attn_output = attn_output.transpose(1, 2)
        # # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # # partitioned across GPUs when using tensor-parallelism.
        # attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        #
        # attn_output = self.out_proj(attn_output)
        #
        # return attn_output, attn_weights, past_key_value


# Copied from transformers.models.mbart.modeling_mbart.MBartEncoderLayer with MBart->Whisper, MBART->WHISPER
class WhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
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
            output_attentions,
            max_len=-1, cu_seqlens=None,
            **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:
            attention_mask:
            max_len:
            cu_seqlens:
            output_attentions:
            **kwargs:

        Returns:

        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=None,
            output_attentions=output_attentions,
            cu_seqlens=cu_seqlens,
            max_len=max_len
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training, inplace=True)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout,
                                              training=self.training, inplace=True)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training, inplace=True)
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


from pretrain_module.modeling_utils import PreTrainedModel


class WhisperPreTrainedModel(PreTrainedModel):
    config_class = WhisperConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WhisperEncoderLayer", "WhisperDecoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, WhisperEncoder):
            with torch.no_grad():
                embed_positions = module.embed_positions.weight
                embed_positions.copy_(sinusoids(*embed_positions.shape))

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """
        input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths


class WhisperEncoder(WhisperPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: WhisperConfig
    """

    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        self.model_size = config.d_model
        self.input_type = "audio"

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)

        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

        from onmt.modules.optimized.flash_mha import flash_bert_mha
        self.fast_bert_mha = flash_bert_mha

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
            self,
            input_features,
            batch_first_output=False,
            output_attentions=False,
            output_hidden_states=False,
            **kwargs
    ):
        """
        Args:
            input_features:
            attention_mask:
            head_mask:
            output_attentions:
            output_hidden_states:
            return_dict:

        Returns:

        """
        feature_size = input_features.size(2)
        assert feature_size == 129

        # remove the padding dimension
        with torch.no_grad():
            input_features = input_features.narrow(2, 1, feature_size - 1)

        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, "
                f"but found {input_features.shape[1]}. "
                f"Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # B x T x H -> B x H x T (for convolution)
        input_features = input_features.transpose(1, 2).contiguous()

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        # B x H x T_ -> B x T_ x H
        inputs_embeds = inputs_embeds.permute(0, 2, 1).contiguous()
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states,
                                              p=self.dropout,
                                              training=self.training,
                                              inplace=True)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # record the seq_len after downsampling
        bsz, seq_len = hidden_states.size(0), hidden_states.size(1)

        is_autocast = torch.is_autocast_enabled()
        autocast_dtype = torch.get_autocast_gpu_dtype()
        # print(is_autocast, autocast_dtype)

        condition_1 = self.fast_bert_mha and is_autocast and (autocast_dtype == torch.float16 or autocast_dtype == torch.bfloat16)

        condition_2 = self.fast_bert_mha and (hidden_states.dtype == torch.float16 or hidden_states.dtype == torch.bfloat16)

        if condition_1 or condition_2:
            can_run_fast_bert_mha = True

            # whisper doesn't use padding mask, so the list of length is simple
            lengths = [seq_len] * bsz

            # resize for 2D
            hidden_states = hidden_states.view(-1, hidden_states.size(-1)).contiguous()  # flatten [B x T]

            max_len = lengths[0]
            # cumulative sequence lengths (required input for fmha)
            a = torch.tensor(np.array([0] + lengths), dtype=torch.int32)
            cu_seqlens = torch.cumsum(a, 0).to(dtype=torch.int32, device=hidden_states.device)

        else:
            can_run_fast_bert_mha = False
            max_len = -1
            cu_seqlens = None
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        if can_run_fast_bert_mha:
            assert hidden_states.ndim == 2, "Flash Attention is used. Expecting 2D inputs!"

        # print("[INFO] Transformer Encoder input: ", hidden_states.size())

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                # if self.gradient_checkpointing and self.training:
                #     layer_outputs = self._gradient_checkpointing_func(
                #         encoder_layer.__call__,
                #         hidden_states,
                #         None,
                #         output_attentions,
                #         max_len, cu_seqlens
                #     )
                # else:
                # print('[INFO] Encoder layer %d ' % idx, hidden_states.size())

                layer_outputs = encoder_layer(
                    hidden_states,
                    output_attentions=output_attentions,
                    max_len=max_len, cu_seqlens=cu_seqlens

                )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if can_run_fast_bert_mha:
            hidden_states = hidden_states.view(bsz, seq_len, -1)

            if not batch_first_output:
                # B x T x H --> T x B x H
                hidden_states = hidden_states.transpose(0, 1).contiguous()
        else:
            # T x B x H -> B x T x H
            if batch_first_output:
                hidden_states = hidden_states.transpose(0, 1).contiguous()

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # print("[INFO] Encoder output: ", hidden_states.size())

        return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
