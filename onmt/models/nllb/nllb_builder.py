# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple

from onmt.data.vocabulary_info import VocabularyInfo

from onmt.models.registry_utils import ArchitectureRegistry
from onmt.nn.transformer.multihead_attention import MultiheadAttention, StandardMultiheadAttention
from onmt.nn.transformer.ffn import FeedForwardNetwork, StandardFeedForwardNetwork
from onmt.nn.transformer.norm_order import TransformerNormOrder
from onmt.nn.transformer.encoder import TransformerEncoder, TransformerEncoderLayer, \
    StandardTransformerEncoder, StandardTransformerEncoderLayer

from onmt.typing import DataType, Device
from onmt.nn.embedding import Embedding, StandardEmbedding, init_scaled_embedding
from .nllb_frontend import TransformerFrontend, TransformerEmbeddingFrontend
from onmt.nn.position_encoder import SinusoidalPositionEncoder


@dataclass
class NllbConfig:
    """Holds the configuration of an NLLB model."""

    model_dim: int
    """The dimensionality of the model."""

    max_seq_len: int
    """The expected maximum sequence length."""

    vocab_info: VocabularyInfo
    """The vocabulary information."""

    num_encoder_layers: int
    """The number of Transformer encoder layers."""

    num_decoder_layers: int
    """The number of Transformer decoder layers."""

    num_encoder_attn_heads: int
    """The number of attention heads in Transformer encoder layers."""

    num_decoder_attn_heads: int
    """The number of attention heads in Transformer decoder layers."""

    ffn_inner_dim: int
    """The inner dimensionality of Transformer feed-forward networks."""

    dropout_p: float
    """The dropout probability in Transformer layers."""


nllb_archs = ArchitectureRegistry[NllbConfig]("nllb")

nllb_arch = nllb_archs.decorator

@nllb_arch("dense_1b")
def _dense_1b() -> NllbConfig:
    return NllbConfig(
        model_dim=1024,
        max_seq_len=1024,
        vocab_info=VocabularyInfo(
            size=256206, unk_idx=1, bos_idx=2, eos_idx=3, pad_idx=0
        ),
        num_encoder_layers=24,
        num_decoder_layers=24,
        num_encoder_attn_heads=16,
        num_decoder_attn_heads=16,
        ffn_inner_dim=1024 * 8,
        dropout_p=0.1,
    )


@nllb_arch("dense_3b")
def _dense_3b() -> NllbConfig:
    return NllbConfig(
        model_dim=2048,
        max_seq_len=1024,
        vocab_info=VocabularyInfo(
            size=256206, unk_idx=1, bos_idx=2, eos_idx=3, pad_idx=0
        ),
        num_encoder_layers=24,
        num_decoder_layers=24,
        num_encoder_attn_heads=16,
        num_decoder_attn_heads=16,
        ffn_inner_dim=1024 * 8,
        dropout_p=0.1,
    )


@nllb_arch("dense_600m")
def _dense_600m() -> NllbConfig:
    return NllbConfig(
        model_dim=1024,
        max_seq_len=1024,
        vocab_info=VocabularyInfo(
            size=256206, unk_idx=1, bos_idx=2, eos_idx=3, pad_idx=0
        ),
        num_encoder_layers=12,
        num_decoder_layers=12,
        num_encoder_attn_heads=16,
        num_decoder_attn_heads=16,
        ffn_inner_dim=1024 * 4,
        dropout_p=0.1,
    )


class NllbBuilder:
    """Builds modules of an NLLB model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2207.04672`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    config: NllbConfig
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
            self,
            config: NllbConfig,
            *,
            device: Optional[Device] = None,
            dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self.config = config

        self.device, self.dtype = device, dtype

    def build_frontend(self, embed: Embedding) -> TransformerFrontend:
        """Build a Transformer encoder/decoder front-end."""
        pos_encoder = SinusoidalPositionEncoder(
            self.config.model_dim,
            self.config.max_seq_len,
            _legacy_pad_idx=1,
            device=self.device,
        )

        return TransformerEmbeddingFrontend(
            embed,
            pos_encoder,
            dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_attention(self, num_heads: int) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer."""
        sdpa = create_default_sdpa(attn_dropout_p=self.config.dropout_p)

        return StandardMultiheadAttention(
            self.config.model_dim,
            num_heads,
            sdpa=sdpa,
            device=self.device,
            dtype=self.dtype,
        )

    def build_ffn(self) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return StandardFeedForwardNetwork(
            self.config.model_dim,
            self.config.ffn_inner_dim,
            bias=True,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )

    def build_encoder_layer(self) -> TransformerEncoderLayer:
        """Build a Transformer encoder layer."""
        self_attn = self.build_attention(self.config.num_encoder_attn_heads)

        ffn = self.build_ffn()

        return StandardTransformerEncoderLayer(
            self_attn,
            ffn,
            dropout_p=self.config.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )

    def build_encoder(self) -> TransformerEncoder:
        """Build a Transformer encoder."""
        num_layers = self.config.num_encoder_layers

        layers = [self.build_encoder_layer() for _ in range(num_layers)]

        return StandardTransformerEncoder(
            layers,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )

    