from transformers import WhisperForConditionalGeneration
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer
from transformers.models.whisper.configuration_whisper import WhisperConfig

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

os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "0"


class SoftmaxCrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, smoothing=0.0, padding_idx=-100, half_to_float=False):
        losses, max_log_sum_exp = xentropy_cuda.forward(
            logits, labels, smoothing, half_to_float)
        losses.masked_fill_(labels == padding_idx, 0)

        ctx.save_for_backward(logits, max_log_sum_exp, labels,
                              torch.FloatTensor([smoothing]),
                              torch.LongTensor([padding_idx]))

        return losses

    @staticmethod
    def backward(ctx, grad_loss):
        logits, max_log_sum_exp, labels, smoothing, padding_idx = ctx.saved_tensors

        if not grad_loss.is_contiguous():
            grad_loss = grad_loss.contiguous()
        grad_loss.masked_fill_(labels == padding_idx.item(), 0)
        grad_logits = xentropy_cuda.backward(
            grad_loss.contiguous(), logits, max_log_sum_exp,
            labels, smoothing.item())

        return grad_logits, None, None, None, None


try:
    import xentropy_cuda

    softmax_xentropy = SoftmaxCrossEntropyLoss.apply
    fast_xentropy = True
except (ModuleNotFoundError, AttributeError):
    softmax_xentropy = None
    fast_xentropy = False

if fast_xentropy:
    print("[INFO] Fast entropy is active.")

try:
    import fast_layer_norm_cuda
except (ModuleNotFoundError, ImportError) as e:
    fast_layer_norm_cuda = None

if fast_layer_norm_cuda is not None:
    print("[INFO] Fast & Efficient layer norm implementation detected.")


#### LAYER NORM IMPLEMENTATION ##############
class FastLayerNormFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, epsilon, memory_efficient=False):
        ctx.x_shape = x.shape
        ctx.memory_efficient = memory_efficient

        x = x.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()
        hidden_size = gamma.numel()

        xmat = x.view((-1, hidden_size))
        ymat, mu, rsigma = fast_layer_norm_cuda.ln_fwd(xmat, gamma, beta, epsilon)
        if ctx.memory_efficient:
            ctx.save_for_backward(ymat, gamma, None, rsigma, beta)
        else:
            ctx.save_for_backward(xmat, gamma, mu, rsigma, None)

        return ymat.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()  # this happens!
        x_or_y_mat, gamma, mu, rsigma, beta = ctx.saved_tensors
        dymat = dy.view(x_or_y_mat.shape)
        dxmat, dgamma, dbeta, _, _ = fast_layer_norm_cuda.ln_bwd(dymat, x_or_y_mat, mu, rsigma, gamma, beta,
                                                                 ctx.memory_efficient)
        dx = dxmat.view(ctx.x_shape)
        return dx, dgamma, dbeta, None, None


def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        try:
            return torch.amp.autocast_mode._cast(args, 'cuda', torch.get_autocast_gpu_dtype())
        except AttributeError:
            return torch.amp.autocast_mode._cast(args, 'cuda', torch.half)


def fast_layer_norm_affine(input, weight, bias, normalized_shape, eps=1e-5, memory_efficient=False):
    args = _cast_if_autocast_enabled(input, weight, bias, eps, memory_efficient)
    with torch.amp.autocast('cuda', enabled=False):
        return FastLayerNormFN.apply(*args)


class MemoryEfficientLayerNorm(torch.nn.LayerNorm):
    """
    See LayerNorm for details.

    Note, however, that unlike LayerNorm this norm includes a batch component.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # set the memory efficient = True by default.
        # by benchmarking there is no speed difference and we can save ~2 GB VRAM in WhisperEncoder
        self.memory_efficient = True

    def forward(self, input, fast=True):
        if input.is_cuda and fast and fast_layer_norm_cuda is not None \
                and input.size(-1) in [768, 1024, 1280, 1536, 2048, 3072, 4096] and self.bias is not None:
            # Note: this layer norm only supports a number of dimension
            return fast_layer_norm_affine(input, self.weight, self.bias,
                                          self.normalized_shape, self.eps, self.memory_efficient)

        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


# Copied from the transformers modeling_whisper.py
def shift_tokens_right(input_ids: torch.Tensor,
                       pad_token_id: int,
                       decoder_start_token_id: int):
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


class MemoryEfficientWhisperEncoderLayer(WhisperEncoderLayer):
    """
    This class simply replaces the residual layer with in-place
    Improves by about 5% (1.75 -> 1.74 it/s)
    """

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


# from .batch_ensemble_whisper_config import WhisperConfig, BatchEnsembleWhisperConfig


class MemoryEfficientWhisper(WhisperForConditionalGeneration):
    """
    Uses fast xentropy loss during training (the loss computation is about 5x faster than pytorch)
    """

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        # TODO: add label smoothing during training

        self.teacher = None
        self.teacher_distillation = 0

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
        original_state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # Create a new OrderedDict excluding the parameters you don't want
        filtered_state_dict = OrderedDict(
            (k, v) for k, v in original_state_dict.items() if "teacher." not in k
        )
        return filtered_state_dict


@dataclass
class DistilledSeq2SeqLMOutput(Seq2SeqLMOutput):
    ce_loss: Optional[torch.FloatTensor] = None  # Add distillation_loss
    distilled_loss: Optional[torch.FloatTensor] = None  # Add distillation_loss


def create_whisper_model(model_name, torch_dtype,
                         attn_implementation="flash_attention_2",
                         low_cpu_mem_usage=False,
                         device_map="none",
                         mem_efficient=True):
    # here model_name can be either the huggingface path, or the local path
    print("[INFO] Creating Whisper model from %s " % model_name)

    def replace_layer_with_weights(model, config):
        for i in range(len(model.model.encoder.layers)):
            old_layer = model.model.encoder.layers[i]
            new_layer = MemoryEfficientWhisperEncoderLayer(config)

            # Copy weights from the old layer to the new one
            new_layer.load_state_dict(old_layer.state_dict())

            dtype = next(old_layer.parameters()).dtype
            new_layer.to(dtype)

            # Replace the layer in the encoder
            model.model.encoder.layers[i] = new_layer

    def replace_layernorm_with_memory_efficient(model):
        for name, module in model.named_children():
            # Check if the current module is LayerNorm
            if isinstance(module, torch.nn.LayerNorm):

                custom_layer = MemoryEfficientLayerNorm(module.normalized_shape, module.eps)

                # Copy weights and biases
                custom_layer.weight.data.copy_(module.weight.data)
                custom_layer.bias.data.copy_(module.bias.data)

                # convert to the right type
                custom_layer.to(module.weight.dtype)
                # Replace with MemoryEfficientLayerNorm
                setattr(model, name, custom_layer)
            else:
                # Recursively apply to submodules
                replace_layernorm_with_memory_efficient(module)

    whisper_class = MemoryEfficientWhisper if mem_efficient else WhisperForConditionalGeneration

    if device_map != "none":
        model = whisper_class.from_pretrained(model_name,
                                              low_cpu_mem_usage=low_cpu_mem_usage,
                                              torch_dtype=torch_dtype,
                                              attn_implementation=attn_implementation,
                                              device_map=device_map,
                                              )
    else:
        model = whisper_class.from_pretrained(model_name,
                                              low_cpu_mem_usage=low_cpu_mem_usage,
                                              torch_dtype=torch_dtype,
                                              attn_implementation=attn_implementation
                                              )
    if mem_efficient:
        replace_layer_with_weights(model, model.config)
        replace_layernorm_with_memory_efficient(model)

    return model



