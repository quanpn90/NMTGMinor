    # Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from dataclasses import asdict, dataclass
from typing import Optional

# from fairseq2.models.conformer import ConformerConvolution
from .w2vbert_conformer import ConformerConvolution
from .w2vbert_model_registry import ArchitectureRegistry
from .w2vbert_config import w2vbert_archs, w2vbert_arch, wav2vec2_arch, wav2vec2_archs

from .w2vbert_builder import Wav2Vec2EncoderBuilder, Wav2Vec2EncoderConfig, Wav2Vec2Config, Wav2Vec2Builder
from .typing import DataType, Device
from .w2vbert_attention import SDPA, ShawRelativePositionSDPA, create_default_sdpa
from .w2vbert_config import ConformerShawEncoderConfig, conformer_shaw_archs




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


# TODO: run this function to create a Wav2Vec2Model

def create_conformer_shaw_model(
    config: Wav2Vec2Config,
    *,
    device = None,
    dtype = None,
):
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