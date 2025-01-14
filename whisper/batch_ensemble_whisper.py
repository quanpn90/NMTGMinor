from transformers import WhisperForConditionalGeneration
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer, WhisperDecoderLayer

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

from transformers.cache_utils import EncoderDecoderCache
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.activations import ACT2FN



os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "0"

from .memory_efficient_whisper import (MemoryEfficientWhisper,
                                       MemoryEfficientWhisperEncoderLayer,
                                       MemoryEfficientLayerNorm)

from .memory_efficient_whisper import SoftmaxCrossEntropyLoss, shift_tokens_right

from .batch_ensemble_linear import BatchEnsembleLinear

from .batch_ensemble_attention import BATCH_ENSEMBLE_WHISPER_ATTENTION_CLASSES
from .batch_ensemble_attention import (BatchEnsembleWhisperAttention,
                                       BatchEnsembleWhisperFlashAttention2,
                                       BatchEnsembleWhisperSdpaAttention)

from .batch_ensemble_whisper_config import WhisperConfig, BatchEnsembleWhisperConfig

from .memory_efficient_whisper import softmax_xentropy, fast_xentropy


class BatchEnsembleWhisperEncoderLayer(MemoryEfficientWhisperEncoderLayer):
    """
    This class simply replaces the residual layer with in-place
    Improves by about 5% (1.75 -> 1.74 it/s)
    """

    def __init__(self, config: BatchEnsembleWhisperConfig):
        super().__init__()
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
        super().__init__()
        self.embed_dim = config.d_model
        self.n_ensembles = config.n_ensembles

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
from .memory_efficient_whisper import MemoryEfficientWhisper

def calculate_lm_logits(logits):

    assert logits.ndim == 4

    probs = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)

    probs = torch.mean(probs, dim=0)

    logits = torch.log(probs) - torch.log(probs.max(dim=-1, keepdim=True))

    return logits


class BatchEnsembleWhisper(MemoryEfficientWhisper):

    """
    Uses fast xentropy loss during training (the loss computation is about 5x faster than pytorch)
    We need to replace
    """
    def __init__(self, config: BatchEnsembleWhisperConfig):
        super().__init__(config)
        # TODO: add label smoothing during training

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

@dataclass
class DistilledSeq2SeqLMOutput(Seq2SeqLMOutput):
    ce_loss: Optional[torch.FloatTensor] = None  # Add distillation_loss
    distilled_loss: Optional[torch.FloatTensor] = None  # Add distillation_loss