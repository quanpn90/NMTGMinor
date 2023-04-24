import copy
import math
import random
from typing import Optional, Tuple, Any, Dict, List, Union

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
from onmt.models.speech_recognizer.fairseq_wav2vec2.fairseq_modules import index_copy

from .activations import ACT2FN
from .modeling_outputs import (
    BaseModelOutput,
)
from .modeling_utils import PreTrainedModel
# from ...utils import logging
# from .configuration_bart import BartConfig
import onmt
from collections import defaultdict
from .configuration_whisper import WhisperConfig


from .modeling_mbart import MBartAttention
from .modeling_mbart import MBartCrossAttention
from .modeling_mbart import MBartEncoderLayer as WhisperEncoderLayer
from .modeling_mbart import MBartDecoderLayer as WhisperDecoderLayer


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

    def forward(self, input_ids, past_key_values_length=0):
        return self.weight[past_key_values_length : past_key_values_length + input_ids.shape[-1]]



class WhisperPreTrainedModel(PreTrainedModel):
    config_class = WhisperConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WhisperEncoderLayer", "WhisperDecoderLayer"]

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

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (WhisperDecoder, WhisperEncoder)):
            module.gradient_checkpointing = value

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
        self.config = config
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

        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def _mask_input_features(
            self,
            input_features: torch.FloatTensor,
            attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # B x T x H -> B x H x T

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return input_features

        # generate indices & apply SpecAugment along time axis
        batch_size, hidden_size, sequence_length = input_features.size()

        if self.config.mask_time_prob > 0 and self.training:
            # generate indices & apply SpecAugment along time axis
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=input_features.device, dtype=torch.bool)
            mask_time_indices = mask_time_indices[:, None].expand(-1, hidden_size, -1)
            input_features[mask_time_indices] = 0

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=input_features.device, dtype=torch.bool)
            input_features[mask_feature_indices] = 0

        return input_features

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
        # The data has been constructed that the first dimension is padding mask
        # 0 for tokens that are not masked, 1 for tokens that are masked
        with torch.no_grad():
            long_mask = input_features.narrow(2, 0, 1).squeeze(2).eq(0).long()
            input_features = input.narrow(2, 1, input.size(2) - 1)

        attention_mask = long_mask

        # [ B x H x T ] -> [ B x T x H ]
        input_features = input_features.permute(0, 2, 1)

        # apply spectral augmentation
        input_features = self._mask_input_features(input_features, attention_mask=attention_mask)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # downsampling stuffs
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        # remove the diluted values in the input
        inputs_embeds.masked_fill_(attention_mask.unsqueeze(1), 0)

        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        # recomputing attention mask
        attention_mask = attention_mask[:, 2::2]
        # remove the diluted values in the input
        inputs_embeds.masked_fill_(attention_mask.unsqueeze(1), 0)

        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        # find position encodings
        bsz, seq_len = inputs_embeds.size(0), inputs_embeds.size(1)

        positions = torch.arange(
            0, seq_len, dtype=torch.long, device=inputs_embeds.device
        ).clamp_(max = self.max_source_positions - 1)
        embed_pos = self.embed_positions(positions)

        hidden_states = inputs_embeds + embed_pos.unsqueeze(0)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # # check if head_mask has a correct number of layers specified if desired
        # if head_mask is not None:
        #     assert head_mask.size()[0] == (
        #         len(self.layers)
        #     ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        can_run_fast_bert_mha = False
        # check if fast bert mha can be run
        seq_len = hidden_states.size(1)
        bsz = hidden_states.size(0)
        sm = torch.cuda.get_device_capability()
        total_bsz = 0

        if self.fast_bert_mha and torch.is_autocast_enabled():
            can_run_fast_bert_mha = True

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
                if self.gradient_checkpointing and self.training:
                    raise NotImplementedError
                    # def create_custom_forward(module):
                    #     def custom_forward(*inputs):
                    #         return module(*inputs, output_attentions)
                    #
                    #     return custom_forward
                    #
                    # layer_outputs = torch.utils.checkpoint.checkpoint(
                    #     create_custom_forward(encoder_layer),
                    #     hidden_states,
                    #     attention_mask,
                    #     (head_mask[idx] if head_mask is not None else None),
                    # )
                else:
                    layer_outputs = encoder_layer(
                        # hidden_states,
                        # None,
                        # layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        # output_attentions=output_attentions,
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

        return tuple(v for v in [hidden_states, encoder_states, all_attentions, attention_mask] if v is not None)


class WhisperDecoder(WhisperPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`WhisperDecoderLayer`]
    Args:
        config: WhisperConfig
    """

    def __init__(self, config: WhisperConfig, opt, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = WhisperPositionalEmbedding(self.max_target_positions, config.d_model)

        self.layers = nn.ModuleList([WhisperDecoderLayer(config) for _ in range(config.decoder_layers)])

        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

        self.model_size = config.d_model
        self.switchout = 0.0
        # self.word_lut = self.embed_tokens
        self.config.bert_hidden_size = config.d_model
        self.layerdrop = opt.death_rate_decoder
        self.dec_pretrained_model = 'mbart'
        if opt.freeze_embedding:
            self.embed_tokens.weight.requires_grad = False
        self.word_dropout = opt.word_dropout

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
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

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
            sub_encoder_hidden_states=None,
            sub_encoder_attention_mask=None,
            inputs_embeds=None,
            incremental=False, incremental_cache=None,
            lang=None, atb=None,
            output_attentions=None,
            output_hidden_states=None,
            checkpointing_ffn=False,
            checkpointing_cross_attn=False,
            checkpointing_self_attn=False,
            **kwargs):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)

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
            inputs_embeds = self.embed_tokens(input_ids)

        bsz = input_ids.size(0)
        qlen = input_ids.size(1)
        klen = qlen

        # if attention_mask is None:
        padding_mask = attention_mask
        attention_mask = torch.triu(
            inputs_embeds.new_ones(qlen, klen), diagonal=1).bool()

        # embed positions
        if input_ids is not None:
            positions = self.embed_positions(input_ids, past_key_values_length=past_key_values_length)
        else:
            positions = self.embed_positions(inputs_embeds, past_key_values_length=past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        self.fast_bert_mha = None
        if self.fast_bert_mha is not None and hidden_states.dtype == torch.half:
            can_run_fast_bert_mha = True

            # lets unpad both
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

            # bsz, seq_len = hidden_states.size(0), hidden_states.size(1)
            # lengths = [seq_len] * bsz
            # a = torch.tensor(np.array([0] + lengths), dtype=torch.int32)
            # cu_seqlens = torch.cumsum(a, 0).to(dtype=torch.int32, device=hidden_states.device)
            # max_len = seq_len
            # total_bsz = hidden_states.size(0)
            # hidden_states = hidden_states.view(-1, hidden_states.size(-1))

            # unpad the context
            encoder_hidden_states = encoder_hidden_states.transpose(0, 1).contiguous()
            padding_mask = encoder_attention_mask
            if padding_mask is None:
                context_len = encoder_hidden_states.size(1)
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
                sub_encoder_hidden_states=sub_encoder_hidden_states,
                sub_encoder_attention_mask=sub_encoder_attention_mask,
                output_attentions=output_attentions,
                lang=lang,
                atb=atb,
                checkpointing_ffn=checkpointing_ffn,
                checkpointing_cross_attn=checkpointing_cross_attn,
                checkpointing_self_attn=checkpointing_self_attn,
                max_len=max_len, cu_seqlens=cu_seqlens,
                max_len_kv=max_len_kv, cu_seqlens_kv=cu_seqlens_kv
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        if can_run_fast_bert_mha:
            seq_len = qlen
            hidden_states = index_copy(hidden_states, non_pad_indices_q, bsz * seq_len)
            hidden_states = hidden_states.view(bsz, seq_len, -1).transpose(0, 1).contiguous()

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

        # embed positions. here it takes input instead of input_size
        positions = self.embed_positions(input_, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = hidden_states.transpose(0, 1)

        hidden_states = self.layernorm_embedding(hidden_states)

        max_len = None
        cu_seqlens = None

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
                lang=lang, atb=atb,
                max_len=max_len, cu_seqlens=cu_seqlens
            )

            if buffering:
                decoder_state.update_attention_buffer(buffer, idx)
            hidden_states = layer_outputs[0]

        hidden_states = self.layer_norm(hidden_states)
        output = hidden_states[-1].unsqueeze(0)

        # just a fake coverage, at the moment coverage is not returned during step
        coverage = hidden_states.new(hidden_states.size(1), 1, encoder_hidden_states.size(0)).zero_()

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = output
        output_dict['coverage'] = coverage
        output_dict['context'] = encoder_hidden_states
        return output_dict


class WhisperFeatureExtractor:



    def __init__(self, config: WhisperConfig):

        self.num_mel_bins = config.num_mel_bins

        try:
            self.sampling_rate = config.sampling_rate
        except Exception:
            self.sampling_rate = 16000

    def __call__(
            self,
            raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
            # truncation: bool = True,
            # sampling_rate: Optional[int] = None,
            do_normalize: Optional[bool] = None,
            **kwargs,
    ):
        from onmt.data.whisper_audio import get_mel_filters, fram_wave, \
            np_extract_fbank_features, zero_mean_unit_var_norm

        if do_normalize:
            x = zero_mean_unit_var_norm(raw_speech)
        else:
            x = raw_speech



            # padded_inputs["input_features"] = self.zero_mean_unit_var_norm(
            #     padded_inputs["input_features"],
            #     attention_mask=padded_inputs["attention_mask"],
            #     padding_value=self.padding_value,
            # )
            # padded_inputs["input_features"] = np.stack(padded_inputs["input_features"], axis=0)