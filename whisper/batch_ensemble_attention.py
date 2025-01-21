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

from transformers.models.whisper.modeling_whisper import (WhisperAttention,
                                                          WhisperFlashAttention2,
                                                          WhisperSdpaAttention)

from batch_ensemble_linear import BatchEnsembleLinear

from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache, StaticCache
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10, logging

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

def group_first_two_dims(x):
    if x.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions")
    # Compute the new shape
    new_shape = (x.size(0) * x.size(1), *x.size()[2:])
    return x.view(*new_shape).contiguous()



class BatchEnsembleWhisperAttention(WhisperAttention):
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
        n_ensembles=4,
    ):
        torch.nn.Module.__init__(self)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config
        self.n_ensembles = n_ensembles

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        if layer_idx is None and is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "
                "will to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.layer_idx = layer_idx

        self.k_proj = BatchEnsembleLinear(embed_dim, embed_dim, n_ensembles, bias=False)
        self.v_proj = BatchEnsembleLinear(embed_dim, embed_dim, n_ensembles, bias=bias)
        self.q_proj = BatchEnsembleLinear(embed_dim, embed_dim, n_ensembles, bias=bias)
        self.out_proj = BatchEnsembleLinear(embed_dim, embed_dim, n_ensembles, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):

        if tensor.ndim == 3:
            return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        elif tensor.ndim == 4:
            n_ensemble = tensor.size(0)
            return tensor.view(n_ensemble, bsz, seq_len, self.num_heads, self.head_dim).transpose(2, 3).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[EncoderDecoderCache] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape during training: Batch x Time x Channel,
        however during testing it should be n_ensemble x Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        if self.training:
            bsz, tgt_len, _ = hidden_states.size()
        else:
            n_ensembles, bsz, tgt_len, _ = hidden_states.size()
            assert n_ensembles == self.n_ensembles

        # get query proj
        query_states = self._shape(self.q_proj(hidden_states) * self.scaling, tgt_len, bsz)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = key_value_states if key_value_states is not None else hidden_states

        if is_cross_attention and past_key_value and is_updated:

            # first, assume that this doesn't happen during training
            assert not self.training

            # reuse k,v, cross_attentions
            # the cache is rearranged during beam search by using the index 0 (batchxbeam dimension)
            # so we need to transpose (0, 1) to have batchxbeam x n_ensemble into n_ensemble x batchxbeam
            key_states = past_key_value.key_cache[self.layer_idx].transpose(0, 1).contiguous()
            value_states = past_key_value.value_cache[self.layer_idx].transpose(0, 1).contiguous()
        else:
            # for the self-attention case

            key_states = self._shape(self.k_proj(current_states), -1, bsz)
            value_states = self._shape(self.v_proj(current_states), -1, bsz)
            if past_key_value is not None:

                assert not training

                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None

                # similarly we have to transpose (0, 1) before updating the cache
                # note: the cache values are concatenated at dimension -2 so it should work out of the box
                key_states, value_states = past_key_value.update(
                    key_states.transpose(0, 1).contiguous(), value_states.transpose(0, 1).contiguous(), self.layer_idx, {"cache_position": cache_position}
                )

                # and after update
                key_states, value_states = key_states.transpose(0, 1), value_states.transpose(0, 1)

        if self.training:

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(3, 4))

        if attention_mask is not None:  # no matter the length, we just slice it

            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

            if not self.training:
                causal_mask = causal_mask.unsqueeze(0)

            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_probs, value_states)

        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        if self.training:
            attn_output = attn_output.transpose(1, 2)
            # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
            # partitioned across GPUs when using tensor-parallelism.
            attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        else:
            attn_output = attn_output.transpose(2, 3)
            # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
            # partitioned across GPUs when using tensor-parallelism.
            attn_output = attn_output.reshape(n_ensembles, bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class BatchEnsembleWhisperFlashAttention2(BatchEnsembleWhisperAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def group_first_two_dim(self, tensor):

        ens, batch_size = tensor.size(0), tensor.size(1)
        new_shape = (ens * batch_size, *tensor.shape[2:])

        return tensor.view(new_shape)

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[EncoderDecoderCache] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "The `static` cache implementation is not compatible with `attn_implementation='flash_attention_2'`. "
                "Use `attn_implementation='sdpa'` in the meantime, and open an issue at https://github.com/huggingface/transformers"
            )
        # WhisperFlashAttention2 attention does not support output_attentions
        if output_attentions:
            raise ValueError("WhisperFlashAttention2 attention does not support output_attentions")

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        if self.training:
            bsz, tgt_len, _ = hidden_states.size()
            n_ensembles = 1
        else:
            n_ensembles, bsz, tgt_len, _ = hidden_states.size()
            assert n_ensembles == self.n_ensembles, f"Tensor has invalid shape: {hidden_states.size()}, while n_ensembles={self.n_ensembles}"

        # get query proj
        # if not self.training and tgt_len == 1500:
        #     print("Query states input: ", hidden_states.flatten()[:20])

        query_states = self.q_proj(hidden_states)

        # if not self.training and tgt_len == 1500:
        #     print("Query states output: ", query_states.flatten()[:20])

        if self.training:
            query_states = torch.reshape(query_states, (bsz, tgt_len, self.num_heads, self.head_dim))
        else:
            query_states = torch.reshape(query_states, (n_ensembles, bsz, tgt_len, self.num_heads, self.head_dim))

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_value and is_updated:
            # first, assume that this doesn't happen during training
            assert not self.training

            # reuse k,v, cross_attentions
            # the cache is rearranged during beam search by using the index 0 (batchxbeam dimension)
            # so we need to transpose (0, 1) to have batchxbeam x n_ensemble into n_ensemble x batchxbeam
            key_states = past_key_value.key_cache[self.layer_idx].transpose(0, 1).contiguous()
            value_states = past_key_value.value_cache[self.layer_idx].transpose(0, 1).contiguous()
        else:
            # key_states = self._shape(self.k_proj(current_states), -1, bsz)
            # value_states = self._shape(self.v_proj(current_states), -1, bsz)
            # if past_key_value is not None:
            #     # save all key/value_states to cache to be re-used for fast auto-regressive generation
            #     cache_position = cache_position if not is_cross_attention else None
            #     key_states, value_states = past_key_value.update(
            #         key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            #     )

            # for the self-attention case
            key_states = self._shape(self.k_proj(current_states), -1, bsz)
            value_states = self._shape(self.v_proj(current_states), -1, bsz)
            if past_key_value is not None:
                assert not self.training or len(past_key_value) == 0

                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None

                # similarly we have to transpose (0, 1) before updating the cache
                # note: the cache values are concatenated at dimension -2 so it should work out of the box
                key_states, value_states = past_key_value.update(
                    key_states.transpose(0, 1).contiguous(), value_states.transpose(0, 1).contiguous(), self.layer_idx,
                    {"cache_position": cache_position}
                )

                # and after update
                key_states, value_states = key_states.transpose(0, 1), value_states.transpose(0, 1)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]
        #  We would need to refactor the KV cache to be able to avoid many of these transpose/reshape/view.
        if self.training:
            key_states = key_states.transpose(1, 2).contiguous()
            value_states = value_states.transpose(1, 2).contiguous()
        else:
            key_states = key_states.transpose(2, 3).contiguous()
            value_states = value_states.transpose(2, 3).contiguous()


        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, : key_states.shape[-2]]

            if not self.training:
                causal_mask = causal_mask.unsqueeze(0)

        if not self.training:
            #
            # print(query_states.size(), value_states.size(), key_states.size())

            query_states = self.group_first_two_dim(query_states)
            value_states = self.group_first_two_dim(value_states)
            key_states = self.group_first_two_dim(key_states)

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)



        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            causal_mask,
            tgt_len,
            dropout=self.dropout if self.training else 0.0,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )
        # if not self.training and tgt_len == 1500:
        #     print("Self Attention output: ", attn_output.flatten()[:20])

        if self.training:
            attn_output = attn_output.reshape(bsz, tgt_len, -1)
        else:
            attn_output = attn_output.reshape(n_ensembles, bsz, tgt_len, -1)

        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class BatchEnsembleWhisperSdpaAttention(BatchEnsembleWhisperAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[EncoderDecoderCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        if output_attentions or layer_head_mask is not None:
            # TODO: Improve this warning with e.g. `model.config._attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "WhisperModel is using WhisperSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True` or `layer_head_mask` not None. Falling back to the manual attention"
                ' implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states,
                key_value_states=key_value_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        if self.training:
            bsz, tgt_len, _ = hidden_states.size()
            n_ensembles = 1
        else:
            n_ensembles, bsz, tgt_len, _ = hidden_states.size()
            assert n_ensembles == self.n_ensembles

        # get query proj
        query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = key_value_states if key_value_states is not None else hidden_states
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

        if is_cross_attention and past_key_value and is_updated:

            # first, assume that this doesn't happen during training

            # reuse k,v, cross_attentions
            # the cache is rearranged during beam search by using the index 0 (batchxbeam dimension)
            # so we need to transpose (0, 1) to have batchxbeam x n_ensemble into n_ensemble x batchxbeam
            key_states = past_key_value.key_cache[self.layer_idx].transpose(0, 1).contiguous()
            value_states = past_key_value.value_cache[self.layer_idx].transpose(0, 1).contiguous()
        else:
            # for the self-attention case

            key_states = self._shape(self.k_proj(current_states), -1, bsz)
            value_states = self._shape(self.v_proj(current_states), -1, bsz)
            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None

                # similarly we have to transpose (0, 1) before updating the cache
                # note: the cache values are concatenated at dimension -2 so it should work out of the box
                key_states, value_states = past_key_value.update(
                    key_states.transpose(0, 1).contiguous(),
                    value_states.transpose(0, 1).contiguous(),
                    self.layer_idx,
                    {"cache_position": cache_position}
                )

                # and after update
                key_states, value_states = key_states.transpose(0, 1), value_states.transpose(0, 1)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

            # TODO: broadcast?

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case tgt_len == 1.
        is_causal = True if self.is_causal and causal_mask is None and tgt_len > 1 else False

        if not self.training:
            query_states = query_states.flatten(0, 1).contiguous()
            value_states = value_states.flatten(0, 1).contiguous()
            key_states = key_states.flatten(0, 1).contiguous()

        # NOTE: SDPA with memory-efficient backend is currently (torch==2.1.2) bugged when using non-contiguous inputs and a custom attn_mask,
        # but we are fine here as `_shape` do call `.contiguous()`. Reference: https://github.com/pytorch/pytorch/issues/112577
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        if self.training:
            if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

        else:
            if attn_output.size() != (n_ensembles * bsz, self.num_heads, tgt_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(n_ensembles, bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        if self.training:
            attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        else:
            attn_output = attn_output.reshape(n_ensembles, bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value


BATCH_ENSEMBLE_WHISPER_ATTENTION_CLASSES = {
    "eager": BatchEnsembleWhisperAttention,
    "flash_attention_2": BatchEnsembleWhisperFlashAttention2,
    "sdpa": BatchEnsembleWhisperSdpaAttention,
}
