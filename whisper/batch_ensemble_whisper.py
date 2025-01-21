from transformers import WhisperForConditionalGeneration
from transformers.models.whisper.modeling_whisper import (WhisperEncoderLayer, WhisperDecoderLayer,
                                                          WhisperEncoder, WhisperDecoder, WhisperModel)

import torch
import torch.nn.functional as F
import os

from typing import Optional, Tuple, Union
from dataclasses import dataclass
from collections import OrderedDict
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.cache_utils import EncoderDecoderCache, Cache
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.activations import ACT2FN

os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "0"

from memory_efficient_whisper import (MemoryEfficientWhisper,
                                      MemoryEfficientWhisperEncoderLayer,
                                      MemoryEfficientLayerNorm)

from memory_efficient_whisper import SoftmaxCrossEntropyLoss, shift_tokens_right

from batch_ensemble_linear import BatchEnsembleLinear

from batch_ensemble_attention import BATCH_ENSEMBLE_WHISPER_ATTENTION_CLASSES
from batch_ensemble_attention import (BatchEnsembleWhisperAttention,
                                      BatchEnsembleWhisperFlashAttention2,
                                      BatchEnsembleWhisperSdpaAttention)

from batch_ensemble_whisper_config import WhisperConfig, BatchEnsembleWhisperConfig, convert_to_whisper_config

from memory_efficient_whisper import softmax_xentropy, fast_xentropy, MemoryEfficientWhisper, DistilledSeq2SeqLMOutput


class BatchEnsembleWhisperEncoderLayer(MemoryEfficientWhisperEncoderLayer):
    """
    This class simply replaces the residual layer with in-place
    Improves by about 5% (1.75 -> 1.74 it/s)
    """

    def __init__(self, config: BatchEnsembleWhisperConfig):
        torch.nn.Module.__init__(self)
        self.embed_dim = config.d_model
        self.n_ensembles = config.n_ensembles

        self.self_attn = BATCH_ENSEMBLE_WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
            n_ensembles=self.n_ensembles
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        # self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        # self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)

        self.fc1 = BatchEnsembleLinear(self.embed_dim, config.encoder_ffn_dim, self.n_ensembles)
        self.fc2 = BatchEnsembleLinear(config.encoder_ffn_dim, self.embed_dim, self.n_ensembles)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def check_input_dim(self, x):

        if self.training:
            assert x.ndim == 3

        else:
            assert x.ndim == 4
            assert x.size(0) == self.n_ensembles

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            layer_head_mask: torch.Tensor,
            output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """

        self.check_input_dim(hidden_states)

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        if self.dropout > 0:
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
        else:
            hidden_states.add_(residual)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)

        if self.dropout > 0:
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
        else:
            hidden_states.add_(residual)

        if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class BatchEnsembleWhisperDecoderLayer(WhisperDecoderLayer):
    def __init__(self, config: BatchEnsembleWhisperConfig, layer_idx: int = None):
        torch.nn.Module.__init__(self)
        self.embed_dim = config.d_model
        self.n_ensembles = config.n_ensembles
        self.layer_idx = layer_idx

        self.self_attn = BATCH_ENSEMBLE_WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            layer_idx=layer_idx,
            config=config,
            n_ensembles=self.n_ensembles
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BATCH_ENSEMBLE_WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            layer_idx=layer_idx,
            config=config,
            n_ensembles=self.n_ensembles
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = BatchEnsembleLinear(self.embed_dim, config.encoder_ffn_dim, self.n_ensembles)
        self.fc2 = BatchEnsembleLinear(config.encoder_ffn_dim, self.embed_dim, self.n_ensembles)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def check_input_dim(self, x):

        if self.training:
            assert x.ndim == 3

        else:
            assert x.ndim == 4
            assert x.size(0) == self.n_ensembles

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[EncoderDecoderCache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """

        self.check_input_dim(hidden_states)
        if encoder_hidden_states is not None:
            self.check_input_dim(encoder_hidden_states)

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )

        # if self.layer_idx == 0:
        #     print("Self Attention, hidden states: ", hidden_states.flatten()[:100])

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

            # add cross-attn to positions 1 of present_key_value tuple
            present_key_value = (present_key_value, cross_attn_present_key_value)

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

        if use_cache:
            outputs += (present_key_value,)

        return outputs


from transformers import WhisperForConditionalGeneration


def calculate_average_logits(logits):
    assert logits.ndim == 4

    probs = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)

    probs = torch.mean(probs, dim=0)

    logits = torch.log(probs) - torch.log(probs.max(dim=-1, keepdim=True).values)

    return logits


class BatchEnsembleWhisperEncoder(WhisperEncoder):

    def __init__(self, config: BatchEnsembleWhisperConfig):
        _config = convert_to_whisper_config(config)
        super().__init__(_config)

        self.layers = nn.ModuleList([BatchEnsembleWhisperEncoderLayer(config) for _ in range(config.encoder_layers)])

        self.config = config
        self.n_ensembles = config.n_ensembles

    # the forward pass should stay the same ... or not

    def check_input_dim(self, x):

        if self.training:
            assert x.ndim == 3
        else:
            assert x.ndim == 4
            assert x.shape[0] == self.n_ensembles

    def forward(
            self,
            input_features,
            attention_mask=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, "
                f"but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self.check_input_dim(input_features)

        if not self.training:
            E, B, D, T = input_features.shape
            conv_input = input_features.contiguous().view(E * B, D, T).contiguous()
        else:
            B, D, T = input_features.shape
            E = self.n_ensembles
            conv_input = input_features

        inputs_embeds = nn.functional.gelu(self.conv1(conv_input))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        if self.training:
            pass
        else:
            hidden_states = hidden_states.view(E, B, hidden_states.size(1), hidden_states.size(2))

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
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        None,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        None,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
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


class BatchEnsembleWhisperDecoder(WhisperDecoder):
    """
        Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`WhisperDecoderLayer`]

        Args:
            config: WhisperConfig
        """

    def __init__(self, config: BatchEnsembleWhisperConfig):
        _config = convert_to_whisper_config(config)

        super().__init__(_config)

        self.layers = nn.ModuleList(
            [BatchEnsembleWhisperDecoderLayer(config, layer_idx) for layer_idx in range(config.decoder_layers)]
        )

        self.config = config
        self.n_ensembles = config.n_ensembles

    def check_input_dim(self, encoder_output, x, is_embed=False):

        if self.training:

            if encoder_output is not None:
                assert encoder_output.ndim == 3
                assert encoder_output.size(0) == x.size(0)  # check batch size

            if is_embed:
                assert x.ndim == 3
            else:
                assert x.ndim == 2

        else:
            if is_embed:
                assert x.ndim == 4
            else:
                assert x.ndim == 3
            assert x.size(0) == self.n_ensembles

            if encoder_output is not None:
                assert encoder_output.ndim == 4
                assert encoder_output.size(0) == self.n_ensembles  # check batch size

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            position_ids=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cache_position=None,
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`WhisperTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`EncoderDecoderCache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
                Pre-computed hidden-states that can be used to speed up auto-regressive (sequential) decoding. There are
                four sets of pre-computed hidden-states: key and values states in the self-attention blocks (2) and
                in the cross-attention blocks (2). The `past_key_values` are returned when `use_cache=True` is passed or
                when `config.use_cache=True`

                Two formats are allowed:
                - An [`~cache_utils.EncoderDecoderCache`] instance;
                - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
                `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of
                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence. It is used to update the
                cache in the correct position and to infer the complete sequence length.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:

            self.check_input_dim(encoder_hidden_states, input_ids, is_embed=False)
            if self.training:
                pass
            else:
                E, B = input_ids.size(0), input_ids.size(1)
                new_shape = (E * B, *input_ids.shape[2:])
                input_ids = input_ids.contiguous().view(new_shape)

            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:

            self.check_input_dim(encoder_hidden_states, input_ids, is_embed=True)
            if self.training:
                pass
            else:
                E, B = inputs_embeds.size(0), inputs_embeds.size(1)
                new_shape = (E * B, *inputs_embeds.shape[2:])
                inputs_embeds = inputs_embeds.view(new_shape)

            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        return_self_attention_cache = False
        if use_cache or past_key_values is not None:
            if isinstance(past_key_values, Cache) and not isinstance(past_key_values, EncoderDecoderCache):
                return_self_attention_cache = True
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            elif not isinstance(past_key_values, EncoderDecoderCache):
                return_legacy_cache = True
                logger.warning_once(
                    "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.43.0. "
                    "You should pass an instance of `EncoderDecoderCache` instead, e.g. "
                    "`past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`."
                )
                past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)

        past_key_values_length = 0
        if cache_position is not None:
            past_key_values_length = cache_position[0]
        elif past_key_values is not None:
            past_key_values_length = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + input_shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

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

        # print("Encoder states", hidden_states)
        #
        # if attention_mask is not None:
        #     print("attention_mask", attention_mask.size())
        # else:
        #     print("attention_mask: None")

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values.self_attention_cache if past_key_values is not None else None,
            output_attentions,
        )

        # it seems causal mask is always None if we use FlashAttn or SDPA

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`..."
                )
                use_cache = False
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (len(self.layers)), (
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        if self.training:
            pass
        else:
            T, D = hidden_states.size(1), hidden_states.size(2)
            hidden_states = hidden_states.view(E, B, T, D)

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    encoder_hidden_states,
                    None,  # encoder attention mask
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,  # past_key_value
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_values if use_cache else None,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None
        if return_self_attention_cache:
            next_cache = past_key_values.self_attention_cache
        if return_legacy_cache:
            next_cache = past_key_values.to_legacy_cache()
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class BatchEnsembleWhisperModel(WhisperModel):

    def __init__(self, config: BatchEnsembleWhisperConfig):
        super().__init__(config)

        self.encoder = BatchEnsembleWhisperEncoder(config)
        self.decoder = BatchEnsembleWhisperDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()


class BatchEnsembleWhisperForConditionalGeneration(MemoryEfficientWhisper):
    """
    Uses fast xentropy loss during training (the loss computation is about 5x faster than pytorch)
    We need to replace
    """

    config_class = BatchEnsembleWhisperConfig

    def __init__(self, config: BatchEnsembleWhisperConfig):
        super().__init__(config)
        # TODO: add label smoothing during training

        self.model = BatchEnsembleWhisperModel(config)

        self.teacher = None
        self.teacher_distillation = 0
        self.n_ensembles = config.n_ensembles

    def forward(
            self,
            input_features: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]] = None,
            decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
            decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:

        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`. `sequence_length` should be smaller than or equal to `config.max_target_positions`.

        Returns:

        Example:

        ```python
        # >>> import torch
        # >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        # >>> from datasets import load_dataset
        #
        # >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        # >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        #
        # >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        #
        # >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        # >>> input_features = inputs.input_features
        #
        # >>> generated_ids = model.generate(inputs=input_features)
        #
        # >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if labels.shape[1] > self.max_target_positions:
                raise ValueError(
                    f"Labels' sequence length {labels.shape[1]} cannot exceed the maximum allowed length of {self.max_target_positions} tokens."
                )
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if self.teacher is not None and self.teacher_distillation > 0 and self.training:
            with torch.no_grad():
                teacher_outputs = self.teacher.model(
                    input_features,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    encoder_outputs=encoder_outputs,
                    decoder_attention_mask=decoder_attention_mask,
                    head_mask=head_mask,
                    decoder_head_mask=decoder_head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    past_key_values=past_key_values,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    decoder_position_ids=decoder_position_ids,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position,
                )

                teacher_lm_logits = self.teacher.proj_out(teacher_outputs[0])
        else:
            teacher_lm_logits = None

        if not self.training:
            # TODO: if n_emsebles is too large we have OOM
            n_ensembles = self.n_ensembles
            input_features = input_features.unsqueeze(0).expand(n_ensembles, -1, -1, -1)
            decoder_input_ids = decoder_input_ids.unsqueeze(0).expand(n_ensembles, -1, -1)

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        lm_logits = self.proj_out(outputs[0])

        if not self.training:
            lm_logits = calculate_average_logits(lm_logits)

        loss = None
        ce_loss = None
        distilled_loss = None
        if labels is not None:

            labels = labels.to(lm_logits.device).reshape(-1)
            logits = lm_logits.view(-1, self.config.vocab_size)

            if fast_xentropy:
                half_to_float = (logits.dtype == torch.float16) or (logits.dtype == torch.bfloat16)
                loss = softmax_xentropy(logits, labels,
                                        0.0,  # smoothing
                                        -100,  # padding
                                        half_to_float)
                bsz = labels.size(0)
                loss = loss.sum().div(bsz)
                ce_loss = loss
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-100,
                                            reduction='mean',
                                            label_smoothing=0.0)
                # move labels to correct device to enable PP
                loss = loss_fct(logits, labels)
                ce_loss = loss

            if teacher_lm_logits is not None:
                distilled_loss = torch.nn.functional.mse_loss(logits.view(-1, logits.size(-1)),
                                                              teacher_lm_logits.view(-1, logits.size(-1)),
                                                              reduction='none')
                distilled_loss = distilled_loss.sum().div(bsz)

                loss = ce_loss + self.teacher_distillation * distilled_loss
            else:
                distilled_loss = 0

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return DistilledSeq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            ce_loss=ce_loss,
            distilled_loss=distilled_loss,
        )

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Call the parent method to get the original state_dict
        original_state_dict = super().state_dict(destination, prefix, keep_vars)
        # Create a new OrderedDict excluding the parameters you don't want
        filtered_state_dict = OrderedDict(
            (k, v) for k, v in original_state_dict.items() if "teacher." not in k
        )
        return filtered_state_dict

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        n_ensembles=1, ensemble_init="random_sign",
                        freeze_params=False,
                        init_weights=False,
                        **kwargs):

        config = (kwargs.pop("config", None) or
                  BatchEnsembleWhisperConfig.from_pretrained(pretrained_model_name_or_path, **kwargs))

        # If it's a WhisperConfig, convert it to BatchEnsembleWhisperConfig
        if isinstance(config, WhisperConfig) and not isinstance(config, BatchEnsembleWhisperConfig):
            config = BatchEnsembleWhisperConfig.from_whisper_config(config, n_ensembles=n_ensembles,
                                                                    ensemble_init=ensemble_init)

            if not init_weights:
                print("[INFO] Initializing BE Weights is set to False, "
                      "but model is trained from scratch! "
                      "The BE weights will be initialized properly to avoid empty tensors.")
                init_weights = True

        # Call the base class's from_pretrained method with the updated config
        kwargs["config"] = config
        model, loading_info = super().from_pretrained(pretrained_model_name_or_path, *model_args,
                                                      output_loading_info=True,  # Ensure loading info is returned,
                                                      **kwargs)

        # Extract missing keys (newly initialized weights)
        missing_keys = loading_info.get("missing_keys", [])
        unexpected_keys = loading_info.get("unexpected_keys", [])
        print(f"Missing keys (newly initialized): {missing_keys}")
        print(f"Unexpected keys in the checkpoint: {unexpected_keys}")

        if freeze_params:
            freezing_keys = ['attn', '.fc', 'layer_norm', 'conv']

            # Optionally, freeze weights not in the missing_keys list
            for name, param in model.named_parameters():
                if name not in missing_keys:

                    # don't freeze the batch ensemble weights
                    if name.endswith(".r") or name.endswith(".s"):
                        param.requires_grad = True

                    else:
                        freeze = False
                        for key in freezing_keys:
                            if key in name:
                                freeze = True;
                                break

                        if freeze:

                            param.requires_grad = False
                        else:
                            print("Un-Freezing param.... ", name)

        # Perform any additional setup if needed (e.g., initializing new BatchEnsemble parameters)
        # self.post_load_setup(init_weights=init_weights)
        model.post_load_setup(init_weights=init_weights)

        return model

    def post_load_setup(self, init_weights=False):

        # note: the model might be created on a meta device
        # so after loading the parameters with from_pretrained, we need to call them here

        if init_weights:
            for module in self.modules():  # Iterate over all submodules
                if isinstance(module, BatchEnsembleLinear):  # Check if it's a BatchEnsembleLinear layer
                    module.reset_modulation_parameters()  # Call the method


def create_batch_ensemble_whisper(model_name, torch_dtype,
                                  attn_implementation="flash_attention_2",
                                  low_cpu_mem_usage=False,
                                  device_map="none",
                                  mem_efficient=True,
                                  n_ensembles=1,
                                  freeze_params=False,
                                  ensemble_init="random_sign",
                                  init_weights=False):
    # First we create a normal Whisper Model

    whisper_class = BatchEnsembleWhisperForConditionalGeneration

    if device_map != "none":
        model = whisper_class.from_pretrained(model_name,
                                              low_cpu_mem_usage=low_cpu_mem_usage,
                                              torch_dtype=torch_dtype,
                                              attn_implementation=attn_implementation,
                                              device_map=device_map,
                                              n_ensembles=n_ensembles, ensemble_init=ensemble_init,
                                              freeze_params=freeze_params,
                                              init_weights=init_weights
                                              )
    else:
        model = whisper_class.from_pretrained(model_name,
                                              low_cpu_mem_usage=low_cpu_mem_usage,
                                              torch_dtype=torch_dtype,
                                              attn_implementation=attn_implementation,
                                              n_ensembles=n_ensembles, ensemble_init=ensemble_init,
                                              freeze_params=freeze_params,
                                              init_weights=init_weights
                                              )

    return model
