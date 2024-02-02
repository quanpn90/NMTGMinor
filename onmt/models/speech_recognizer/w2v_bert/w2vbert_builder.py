# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple

from torch.nn import GELU, SiLU

from .w2vbert_config import Wav2Vec2EncoderConfig, Wav2Vec2Config, W2VBertConfig
from .w2vbert_feature_extractor import Wav2Vec2FeatureExtractor, Wav2Vec2FbankFeatureExtractor, SequenceFeatureExtractor
from .w2vbert_ffn import StandardFeedForwardNetwork, FeedForwardNetwork
from .w2vbert_multihead_attention import StandardMultiheadAttention
from .typing import Device, DataType
from .w2vbert_attention import SDPA, create_default_sdpa, ShawRelativePositionSDPA
from .w2vbert_convolution import ConformerConvolution
from .frontend import Wav2Vec2Frontend
from .w2vbert_transformer_encoder import TransformerEncoderLayer, StandardTransformerEncoderLayer
from .w2vbert_transformer_encoder import TransformerEncoder, StandardTransformerEncoder
from .w2vbert_multihead_attention import MultiheadAttention
from .w2vbert_wav2vec import Wav2Vec2Model
from .masker import Wav2Vec2Masker
from .vector_quantizer import VectorQuantizer, GumbelVectorQuantizer
from .w2vbert_conformer import ConformerBlock, ConformerConvolution


class Wav2Vec2EncoderBuilder:
    """Builds modules of a wav2vec 2.0 encoder as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    config: Wav2Vec2EncoderConfig

    def __init__(
            self,
            config: Wav2Vec2EncoderConfig,
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

        # if config.use_conformer and config.norm_order != TransformerNormOrder.POST:
        #     raise ValueError(
        #         f"`config.norm_order` must be `POST` when `config.use_conformer` is `True`, but is `{config.norm_order}` instead."
        #     )

        self.config = config

        self.rel_pos_encoding = None

        self.device, self.dtype = device, dtype

    def build_frontend(self) -> Wav2Vec2Frontend:
        """Build a wav2vec 2.0 encoder front-end, which is the module extracting features from either wav or logmel."""
        feature_extractor = self.build_feature_extractor()

        pos_encoder = self.build_position_encoder()

        return Wav2Vec2Frontend(
            self.config.model_dim,
            self.config.feature_dim,
            feature_extractor,
            pos_encoder,
            first_pass_dropout_p=self.config.first_pass_dropout_p,
            layer_norm=self.config.layer_norm_features,
            dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_feature_extractor(self) -> Optional[SequenceFeatureExtractor]:
        """Build a feature extractor."""
        if self.config.use_fbank:
            return Wav2Vec2FbankFeatureExtractor(
                self.config.num_fbank_channels,
                self.config.fbank_stride,
                sample_every_k=self.config.sample_fbank_every_k,
            )

        return Wav2Vec2FeatureExtractor(
            self.config.feature_extractor_layer_descs,
            self.config.feature_extractor_bias,
            layer_norm=self.config.feature_extractor_layer_norm_convs,
            grad_scale=self.config.feature_grad_scale,
            device=self.device,
            dtype=self.dtype,
        )

    def build_position_encoder(self):
        """Build a position encoder."""
        if self.config.pos_encoder_type != "conv":
            return None

        if self.config.pos_encoder_depth == 1:
            return Wav2Vec2PositionEncoder(
                self.config.model_dim,
                self.config.pos_conv_kernel_size,
                self.config.num_pos_conv_groups,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            return Wav2Vec2StackedPositionEncoder(
                self.config.model_dim,
                self.config.pos_conv_kernel_size,
                self.config.num_pos_conv_groups,
                self.config.pos_encoder_depth,
                device=self.device,
                dtype=self.dtype,
            )

    def build_encoder(self):
        """Build a Transformer encoder."""
        num_layers = self.config.num_encoder_layers

        layers = [self.build_encoder_layer() for _ in range(num_layers)]

        return StandardTransformerEncoder(
            layers,
            layer_drop_p=self.config.layer_drop_p,
            norm_order=self.config.norm_order,
            device=self.device,
            dtype=self.dtype,
        )

    def build_ffn(self, use_swish: bool = False) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return StandardFeedForwardNetwork(
            self.config.model_dim,
            self.config.ffn_inner_dim,
            bias=True,
            inner_activation=SiLU() if use_swish else GELU(),
            norm_order=self.config.norm_order,
            device=self.device,
            dtype=self.dtype,
        )

    def build_encoder_layer(self) -> TransformerEncoderLayer:
        """Build a Transformer encoder layer."""
        if self.config.use_conformer:
            return self.build_conformer_block()

        self_attn = self.build_attention()

        ffn = self.build_ffn()

        return StandardTransformerEncoderLayer(
            self_attn,
            ffn,
            dropout_p=self.config.dropout_p,
            norm_order=self.config.norm_order,
            device=self.device,
            dtype=self.dtype,
        )

    def build_conformer_block(self) -> TransformerEncoderLayer:
        """Build a Conformer block."""
        ffn1 = self.build_ffn(use_swish=True)

        self_attn = self.build_attention()

        conv = self.build_conformer_conv()

        ffn2 = self.build_ffn(use_swish=True)

        return ConformerBlock(
            ffn1,
            self_attn,
            conv,
            ffn2,
            dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_attention(self) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer."""
        if self.config.pos_encoder_type == "rotary":
            pos_encoder = RotaryEncoder(
                self.config.model_dim // self.config.num_encoder_attn_heads,
                self.config.max_seq_len,
                device=self.device,
            )
        else:
            pos_encoder = None

        sdpa = self.build_sdpa()

        return StandardMultiheadAttention(
            self.config.model_dim,
            self.config.num_encoder_attn_heads,
            pos_encoder=pos_encoder,
            sdpa=sdpa,
            device=self.device,
            dtype=self.dtype,
        )

    def build_sdpa(self) -> SDPA:
        sdpa = create_default_sdpa(attn_dropout_p=self.config.attn_dropout_p)

        if self.config.pos_encoder_type == "relative":
            if self.rel_pos_encoding is None:
                self.rel_pos_encoding = RelativePositionalEncoding(
                    self.config.model_dim,
                    self.config.max_seq_len,
                    device=self.device,
                    dtype=self.dtype,
                )

            sdpa = RelativePositionSDPA(
                self.config.model_dim,
                self.config.num_encoder_attn_heads,
                self.rel_pos_encoding,
                inner_sdpa=sdpa,
                device=self.device,
                dtype=self.dtype,
            )

        return sdpa

    def build_conformer_conv(self) -> ConformerConvolution:
        return ConformerConvolution(
            self.config.model_dim,
            self.config.depthwise_conv_kernel_size,
            device=self.device,
            dtype=self.dtype,
        )


class Wav2Vec2Builder:
    """Builds modules of a wav2vec 2.0 model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    config: Wav2Vec2Config
    encoder_builder: Wav2Vec2EncoderBuilder
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
            self,
            config: Wav2Vec2Config,
            encoder_builder: Wav2Vec2EncoderBuilder,
            *,
            device: Optional[Device] = None,
            dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration.
        :param encoder_builder_cls:
            The wav2vec 2.0 encoder builder.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self.config = config

        self.encoder_builder = encoder_builder

        self.device, self.dtype = device, dtype

    def build_model(self) -> Wav2Vec2Model:
        """Build a model."""
        encoder_frontend = self.encoder_builder.build_frontend()

        encoder = self.encoder_builder.build_encoder()

        masker = self.build_masker()

        quantizer = self.build_quantizer()

        return Wav2Vec2Model(
            encoder_frontend,
            encoder,
            masker,
            quantizer,
            self.config.final_dim,
            final_proj_bias=self.config.final_proj_bias,
            num_distractors=self.config.num_distractors,
            logit_temp=self.config.logit_temp,
            diversity_loss_weight=self.config.diversity_loss_weight,
            device=self.device,
            dtype=self.dtype,
        )

    def build_masker(self) -> Wav2Vec2Masker:
        """Build a temporal/spatial feature masker."""
        return Wav2Vec2Masker(
            self.config.encoder_config.model_dim,
            self.config.temporal_mask_span_len,
            self.config.max_temporal_mask_prob,
            self.config.spatial_mask_span_len,
            self.config.max_spatial_mask_prob,
            device=self.device,
            dtype=self.dtype,
        )

    def build_quantizer(self) -> VectorQuantizer:
        """Build a vector quantizer."""
        return GumbelVectorQuantizer(
            self.config.encoder_config.feature_dim,
            self.config.quantized_dim,
            self.config.num_codebooks,
            self.config.num_codebook_entries,
            codebook_sampling_temperature=self.config.codebook_sampling_temperature,
            device=self.device,
            dtype=self.dtype,
        )


from .w2vbert_config import ConformerShawEncoderConfig


class ConformerShawEncoderBuilder(Wav2Vec2EncoderBuilder):
    """
    Builds modules of a `ConformerShawEncoderBuilder`.

    This is a Conformer architecture with these differences:
    - ShawRelativePositionSDPA as the SDPA.
    - ConformerConvolution with causal depthwise convolution
    and norm_type "layer_norm".
    """

    config: ConformerShawEncoderConfig

    def __init__(
            self,
            config: ConformerShawEncoderConfig,
            *,
            device: Optional[Device] = None,
            dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration to use.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        super().__init__(config, device=device, dtype=dtype)

        assert self.config.use_conformer, "This architecture only supports a Conformer."
        assert (
                self.config.pos_encoder_type == "shaw_relative"
        ), "This architecture only supports ShawRelativePositionSDPA."

    def build_sdpa(self) -> SDPA:
        if self.config.shaw_rel_pos_sdpa_config is None:
            raise ValueError(
                "`shaw_rel_pos_sdpa_config` must be specified when `pos_encoder_type` is 'shaw_relative'."
            )

        sdpa = create_default_sdpa(attn_dropout_p=self.config.attn_dropout_p)

        sdpa_config = self.config.shaw_rel_pos_sdpa_config

        return ShawRelativePositionSDPA(
            self.config.model_dim,
            self.config.num_encoder_attn_heads,
            sdpa_config.max_left_rel_pos,
            max_right_rel_pos=sdpa_config.max_right_rel_pos,
            use_rel_pos_values=sdpa_config.use_rel_pos_values,
            inner_sdpa=sdpa,
            device=self.device,
            dtype=self.dtype,
        )

    def build_conformer_conv(self) -> ConformerConvolution:
        return ConformerConvolution(
            self.config.model_dim,
            self.config.depthwise_conv_kernel_size,
            causal_depthwise_conv=True,
            norm_type="layer_norm",
            device=self.device,
            dtype=self.dtype,
        )


def create_conformer_shaw_model(
        config: Wav2Vec2Config,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
) -> Wav2Vec2Model:
    """Create a conformer shaw model.

    :param config:
        The configuration.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    assert isinstance(config.encoder_config, ConformerShawEncoderConfig)

    encoder_builder = ConformerShawEncoderBuilder(
        config.encoder_config, device=device, dtype=dtype
    )

    builder = Wav2Vec2Builder(config, encoder_builder, device=device, dtype=dtype)

    return builder.build_model()
