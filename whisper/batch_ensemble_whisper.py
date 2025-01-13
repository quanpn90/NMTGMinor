from transformers import WhisperForConditionalGeneration
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer

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

from .memory_efficient_whisper import (MemoryEfficientWhisper,
                                       MemoryEfficientWhisperEncoderLayer,
                                       MemoryEfficientLayerNorm)

from .memory_efficient_whisper import SoftmaxCrossEntropyLoss, shift_tokens_right

from .batch_ensemble_linear import BatchEnsembleLinear

from .batch_ensemble_attention import BATCH_ENSEMBLE_WHISPER_ATTENTION_CLASSES
from .batch_ensemble_attention import (BatchEnsembleWhisperAttention,
                                       BatchEnsembleWhisperFlashAttention2,
                                       BatchEnsembleWhisperSdpaAttention)


class BatchEnsembleWhisperEncoderLayer(MemoryEfficientWhisperEncoderLayer):
    """
    This class simply replaces the residual layer with in-place
    Improves by about 5% (1.75 -> 1.74 it/s)
    """

    def __init__(self, config: WhisperConfig, n_ensembles=4):
        super().__init__()
        self.embed_dim = config.d_model
        self.n_ensembles = n_ensembles

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

        pass

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