import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, Parameter

from .whisper_config import WhisperConfig
from .whisper_encoder import WhisperPositionalEmbedding, WhisperAttention, WhisperPreTrainedModel
from pretrain_module.activations import ACT2FN


from onmt.modules.optimized.linear import linear_function, factorize_linear
from onmt.modules.optimized.self_attention_func import self_attn_func, self_attn_compact_func
from onmt.modules.optimized.encdec_attention_func_bias import encdec_attn_bias_func, encdec_attn_bias_compact_func



class WhisperCrossAttention(WhisperAttention):

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

        super().__init__(embed_dim, num_heads, dropout, is_decoder, bias,
                         is_causal, layer_idx, config)

        from onmt.modules.optimized.flash_mha import flash_encdec_mha
        self.fast_bert_mha = flash_encdec_mha

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

    # TODO: rewriting this function into:
    # forward without cache
    # forward with cache
    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value=None,
            attention_mask: Optional[torch.Tensor] = None,
            cu_seqlens=None, max_len=None,
            lang=None, atb=None,
            incremental=False, incremental_cache=None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
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

        in_proj_weight_q = self.q_proj.weight
        in_proj_weight_kv = self.proj_weight_kv
        out_proj_weight = self.out_proj.weight

        if hidden_states.ndim == 3 and key_value_states.ndim == 3:
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
                                                          False, None, None,  # no rotary encodings
                                                          low_precision, True)

        else:
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
                                                          False, None, None,  # no rotary encodings
                                                          low_precision, True)

        return attn_output, coverage, incremental_cache

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        # is_cross_attention = key_value_states is not None
        # bsz, tgt_len, _ = hidden_states.size()
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


class WhisperDecoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig, layer_idx: int = None):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            layer_idx=layer_idx,
            config=config,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = WhisperCrossAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            layer_idx=layer_idx,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
            incremental: Optional[bool] = False,
            incremental_cache=None,
            max_len=None, cu_seqlens=None,
            max_len_kv=None, cu_seqlens_kv=None, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:
            attention_mask:
            encoder_hidden_states:
            encoder_attention_mask:
            output_attentions:
            incremental:
            incremental_cache:
            max_len:
            cu_seqlens:
            max_len_kv:
            cu_seqlens_kv:
            **kwargs:

        Returns:

        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            incremental=incremental, incremental_cache=incremental_cache,
            cu_seqlens=cu_seqlens, max_len=max_len
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

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
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)


        return outputs, incremental_cache


class WhisperDecoder(WhisperPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`WhisperDecoderLayer`]

    Args:
        config: WhisperConfig
    """

    main_input_name = "input_ids"

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.embed_positions = WhisperPositionalEmbedding(self.max_target_positions, config.d_model)

        self.layers = nn.ModuleList(
            [WhisperDecoderLayer(config, layer_idx) for layer_idx in range(config.decoder_layers)]
        )
        # self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        # self._use_sdpa = config._attn_implementation == "sdpa"

        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        # input_ids=None,
        # attention_mask=None,
        # encoder_hidden_states=None,
        # head_mask=None,
        # cross_attn_head_mask=None,
        # past_key_values=None,
        # inputs_embeds=None,
        # position_ids=None,
        # use_cache=None,
        # output_attentions=None,
        # output_hidden_states=None,
        # return_dict=None,
        # cache_position=None,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        """
        Args:
            input_ids:
            attention_mask:
            encoder_hidden_states:
            encoder_attention_mask:
            inputs_embeds:
            output_attentions:
            output_hidden_states:
            **kwargs:

        Returns:
        """

        # This model doesn't seem to need any mask at all
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

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_key_values_length = 0
        position_ids = None

        # if position_ids is None:
        #     position_ids = cache_position.unsqueeze(0)

        # embed positions
        if input_ids is not None:
            positions = self.embed_positions(
                input_ids, past_key_values_length=past_key_values_length, position_ids=position_ids
            )
        else:
            positions = self.embed_positions(
                inputs_embeds, past_key_values_length=past_key_values_length, position_ids=position_ids
        )

        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # causal_mask = self._update_causal_mask(
        #     attention_mask,
        #     inputs_embeds,
        #     cache_position,
        #     past_key_values.self_attention_cache if past_key_values is not None else None,
        #     output_attentions,
        # )
        qlen = klen = hidden_states.size(1)

        causal_mask = torch.triu(
            inputs_embeds.new_ones(qlen, klen), diagonal=1).bool()

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            # if self.gradient_checkpointing and self.training:
            #     layer_outputs = self._gradient_checkpointing_func(
            #         decoder_layer.__call__,
            #         hidden_states,
            #         causal_mask,
            #         encoder_hidden_states,
            #         None,  # encoder attention mask
            #         head_mask[idx] if head_mask is not None else None,
            #         cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
            #         None,  # past_key_value
            #         output_attentions,
            #         use_cache,
            #         cache_position,
            #     )
            # else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                encoder_hidden_states=encoder_hidden_states,
                output_attentions=output_attentions,
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

        return tuple(
            v
            for v in [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions]
            if v is not None
        )

    def step(self, input, decoder_state, **kwargs):

        raise NotImplementedError

        return

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    # def _update_causal_mask(
    #     self,
    #     attention_mask: torch.Tensor,
    #     input_tensor: torch.Tensor,
    #     cache_position: torch.Tensor,
    #     past_key_values,
    #     output_attentions: bool,
    # ):
    #     if self.config._attn_implementation == "flash_attention_2":
    #         if attention_mask is not None and 0.0 in attention_mask:
    #             return attention_mask
    #         return None
    #
    #     # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
    #     # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
    #     # to infer the attention mask.
    #     past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    #     using_static_cache = isinstance(past_key_values, StaticCache)
    #
    #     # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
    #     if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
    #         if AttentionMaskConverter._ignore_causal_mask_sdpa(
    #             attention_mask,
    #             inputs_embeds=input_tensor,
    #             past_key_values_length=past_seen_tokens,
    #             is_training=self.training,
    #         ):
    #             return None
    #
    #     dtype, device = input_tensor.dtype, input_tensor.device
    #     sequence_length = input_tensor.shape[1]
    #     if using_static_cache:
    #         target_length = past_key_values.get_max_cache_shape()
    #     else:
    #         target_length = (
    #             attention_mask.shape[-1]
    #             if isinstance(attention_mask, torch.Tensor)
    #             else past_seen_tokens + sequence_length + 1
    #         )
    #
    #     # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
    #     causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
    #         attention_mask,
    #         sequence_length=sequence_length,
    #         target_length=target_length,
    #         dtype=dtype,
    #         device=device,
    #         cache_position=cache_position,
    #         batch_size=input_tensor.shape[0],
    #     )
    #
    #     if (
    #         self.config._attn_implementation == "sdpa"
    #         and attention_mask is not None
    #         and attention_mask.device.type == "cuda"
    #         and not output_attentions
    #     ):
    #         # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
    #         # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
    #         # Details: https://github.com/pytorch/pytorch/issues/110213
    #         min_dtype = torch.finfo(dtype).min
    #         causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
    #
    #     return causal_mask
    #
    # @staticmethod
    # # Copied from transformers.models.llama.modeling_llama.LlamaModel._prepare_4d_causal_attention_mask_with_cache_position
    # def _prepare_4d_causal_attention_mask_with_cache_position(
    #     attention_mask: torch.Tensor,
    #     sequence_length: int,
    #     target_length: int,
    #     dtype: torch.dtype,
    #     device: torch.device,
    #     cache_position: torch.Tensor,
    #     batch_size: int,
    #     **kwargs,
    # ):
    #     """
    #     Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    #     `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.
    #
    #     Args:
    #         attention_mask (`torch.Tensor`):
    #             A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
    #             `(batch_size, 1, query_length, key_value_length)`.
    #         sequence_length (`int`):
    #             The sequence length being processed.
    #         target_length (`int`):
    #             The target length: when generating with static cache, the mask should be as long as the static cache,
    #             to account for the 0 padding, the part of the cache that is not filled yet.
    #         dtype (`torch.dtype`):
    #             The dtype to use for the 4D attention mask.
    #         device (`torch.device`):
    #             The device to plcae the 4D attention mask on.
    #         cache_position (`torch.Tensor`):
    #             Indices depicting the position of the input sequence tokens in the sequence.
    #         batch_size (`torch.Tensor`):
    #             Batch size.
    #     """
    #     if attention_mask is not None and attention_mask.dim() == 4:
    #         # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
    #         causal_mask = attention_mask
    #     else:
    #         min_dtype = torch.finfo(dtype).min
    #         causal_mask = torch.full(
    #             (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
    #         )
    #         if sequence_length != 1:
    #             causal_mask = torch.triu(causal_mask, diagonal=1)
    #         causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    #         causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    #         if attention_mask is not None:
    #             causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
    #             mask_length = attention_mask.shape[-1]
    #             padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
    #             padding_mask = padding_mask == 0
    #             causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
    #                 padding_mask, min_dtype
    #             )
    #
    #     return causal_mask