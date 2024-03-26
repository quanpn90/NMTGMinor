# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

from onmt.data.vocabulary_info import VocabularyInfo
# from fairseq2.models.transformer import (
#     TransformerDecoderModel,
#     TransformerEmbeddingFrontend,
#     TransformerFrontend,
#     init_final_projection,
# )
from onmt.nn.transformer.model import init_final_projection

from onmt.models.registry_utils import ArchitectureRegistry
from onmt.nn.embedding import StandardEmbedding

from onmt.nn.normalization import LayerNorm, RMSNorm
from onmt.nn.position_encoder import RotaryEncoder
from onmt.nn.projection import Linear

from onmt.nn.transformer.ffn import FeedForwardNetwork, GLUFeedForwardNetwork
from onmt.nn.transformer.multihead_attention import MultiheadAttention, StandardMultiheadAttention, create_default_sdpa
from onmt.nn.transformer.decoder import TransformerDecoderLayer, StandardTransformerDecoderLayer
from onmt.nn.transformer.decoder import TransformerDecoder, StandardTransformerDecoder
from onmt.nn.transformer.norm_order import TransformerNormOrder
from onmt.nn.transformer.decoder_model import TransformerDecoderModel

from onmt.typing import DataType, Device
from onmt.nn.transformer.frontend import TransformerFrontend, TransformerEmbeddingFrontend


@dataclass
class LLaMAConfig:
    """Holds the configuration of a LLaMA model."""

    model_dim: int
    """The dimensionality of the model."""

    max_seq_len: int
    """The maximum allowed sequence length."""

    vocab_info: VocabularyInfo
    """The vocabulary information."""

    num_layers: int
    """The number of Transformer decoder layers."""

    num_attn_heads: int
    """The number of attention heads in Transformer decoder layers."""

    num_key_value_heads: int
    """The number of key/value heads for Grouped Query Attention."""

    ffn_inner_dim: int
    """The dimensionality of inner projection layers in Transformer feed-forward
    networks."""

    ffn_inner_dim_to_multiple: int
    """The dimensionality of inner projection layers in Transformer feed-forward
    networks is rounded up to the nearest multiple of this value."""

    dropout_p: float
    """The dropout probability in Transformer layers."""


llama_archs = ArchitectureRegistry[LLaMAConfig]("llama")

llama_arch = llama_archs.decorator


@llama_arch("7b")
def _7b() -> LLaMAConfig:
    return LLaMAConfig(
        model_dim=4096,
        max_seq_len=2048,
        vocab_info=VocabularyInfo(
            size=32000, unk_idx=0, bos_idx=1, eos_idx=2, pad_idx=None
        ),
        num_layers=32,
        num_attn_heads=32,
        num_key_value_heads=32,
        ffn_inner_dim=4096 * 4,
        ffn_inner_dim_to_multiple=256,
        dropout_p=0.1,
    )


@llama_arch("13b")
def _13b() -> LLaMAConfig:
    return LLaMAConfig(
        model_dim=5120,
        max_seq_len=2048,
        vocab_info=VocabularyInfo(
            size=32000, unk_idx=0, bos_idx=1, eos_idx=2, pad_idx=None
        ),
        num_layers=40,
        num_attn_heads=40,
        num_key_value_heads=40,
        ffn_inner_dim=5120 * 4,
        ffn_inner_dim_to_multiple=256,
        dropout_p=0.1,
    )


@llama_arch("33b")
def _33b() -> LLaMAConfig:
    return LLaMAConfig(
        model_dim=6656,
        max_seq_len=2048,
        vocab_info=VocabularyInfo(
            size=32000, unk_idx=0, bos_idx=1, eos_idx=2, pad_idx=None
        ),
        num_layers=60,
        num_attn_heads=52,
        num_key_value_heads=52,
        ffn_inner_dim=6656 * 4,
        ffn_inner_dim_to_multiple=256,
        dropout_p=0.1,
    )


@llama_arch("65b")
def _65b() -> LLaMAConfig:
    return LLaMAConfig(
        model_dim=8192,
        max_seq_len=2048,
        vocab_info=VocabularyInfo(
            size=32000, unk_idx=0, bos_idx=1, eos_idx=2, pad_idx=None
        ),
        num_layers=80,
        num_attn_heads=64,
        num_key_value_heads=64,
        ffn_inner_dim=8192 * 4,
        ffn_inner_dim_to_multiple=256,
        dropout_p=0.1,
    )


@llama_arch("llama2_7b")
def _llama2_7b() -> LLaMAConfig:
    return LLaMAConfig(
        model_dim=4096,
        max_seq_len=4096,
        vocab_info=VocabularyInfo(
            size=32000, unk_idx=0, bos_idx=1, eos_idx=2, pad_idx=None
        ),
        num_layers=32,
        num_attn_heads=32,
        num_key_value_heads=32,
        ffn_inner_dim=4096 * 4,
        ffn_inner_dim_to_multiple=256,
        dropout_p=0.1,
    )


@llama_arch("llama2_13b")
def _llama2_13b() -> LLaMAConfig:
    return LLaMAConfig(
        model_dim=5120,
        max_seq_len=4096,
        vocab_info=VocabularyInfo(
            size=32000, unk_idx=0, bos_idx=1, eos_idx=2, pad_idx=None
        ),
        num_layers=40,
        num_attn_heads=40,
        num_key_value_heads=40,
        ffn_inner_dim=5120 * 4,
        ffn_inner_dim_to_multiple=256,
        dropout_p=0.1,
    )


@llama_arch("llama2_70b")
def _llama2_70b() -> LLaMAConfig:
    return LLaMAConfig(
        model_dim=8192,
        max_seq_len=4096,
        vocab_info=VocabularyInfo(
            size=32000, unk_idx=0, bos_idx=1, eos_idx=2, pad_idx=None
        ),
        num_layers=80,
        num_attn_heads=64,
        num_key_value_heads=8,
        ffn_inner_dim=int(8192 * 4 * 1.3),  # See A.2.1 in LLaMA 2
        ffn_inner_dim_to_multiple=4096,
        dropout_p=0.1,
    )


class LLaMABuilder:
    """Builds modules of a LLaMA model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2302.13971` and
    :cite:t:`https://doi.org/10.48550/arXiv.2307.09288`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    _config: LLaMAConfig
    _device: Optional[Device]
    _dtype: Optional[DataType]
    _pos_encoder: Optional[RotaryEncoder]

    def __init__(
            self,
            config: LLaMAConfig,
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
        self._config = config

        self._device, self._dtype = device, dtype

        self._pos_encoder = None

    def build_model(self) -> TransformerDecoderModel:
        """Build a model."""
        decoder_frontend = self.build_decoder_frontend()

        decoder = self.build_decoder()

        final_proj = Linear(
            self._config.model_dim,
            self._config.vocab_info.size,
            bias=False,
            init_fn=init_final_projection,
            device=self._device,
            dtype=self._dtype,
        )

        return TransformerDecoderModel(
            decoder_frontend,
            decoder,
            final_proj,
            self._config.max_seq_len,
            self._config.vocab_info,
        )

    def build_decoder_frontend(self) -> TransformerFrontend:
        """Build a Transformer decoder front-end."""
        embed = StandardEmbedding(
            num_embeddings=self._config.vocab_info.size,
            embedding_dim=self._config.model_dim,
            device=self._device,
            dtype=self._dtype,
        )

        return TransformerEmbeddingFrontend(
            embed,
            pos_encoder=None,
            no_scale=True,  # LLaMA does not use embedding scaling.
            dropout_p=self._config.dropout_p,
            device=self._device,
            dtype=self._dtype,
        )

    def build_decoder(self) -> TransformerDecoder:
        """Build a Transformer decoder."""
        num_layers = self._config.num_layers

        layers = [self.build_decoder_layer() for _ in range(num_layers)]

        return StandardTransformerDecoder(
            layers,
            norm_order=TransformerNormOrder.PRE,
            layer_norm_factory=self.build_layer_norm,
            device=self._device,
            dtype=self._dtype,
        )

    def build_decoder_layer(self) -> TransformerDecoderLayer:
        """Build a Transformer decoder layer."""
        self_attn = self.build_attention(
            self._config.num_attn_heads, self._config.num_key_value_heads
        )

        ffn = self.build_ffn()

        return StandardTransformerDecoderLayer(
            self_attn,
            encoder_decoder_attn=None,
            ffn=ffn,
            dropout_p=self._config.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            layer_norm_factory=self.build_layer_norm,
            device=self._device,
            dtype=self._dtype,
        )

    def build_attention(
            self, num_heads: int, num_key_value_heads: int
    ) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer."""
        sdpa = create_default_sdpa(attn_dropout_p=self._config.dropout_p)

        if self._pos_encoder is None:
            self._pos_encoder = RotaryEncoder(
                self._config.model_dim // num_heads,
                self._config.max_seq_len,
                device=self._device,
            )

        return StandardMultiheadAttention(
            self._config.model_dim,
            num_heads,
            num_key_value_heads=num_key_value_heads,
            sdpa=sdpa,
            pos_encoder=self._pos_encoder,
            bias=False,
            device=self._device,
            dtype=self._dtype,
        )

    def build_ffn(self) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return GLUFeedForwardNetwork(
            self._config.model_dim,
            self._config.ffn_inner_dim,
            bias=False,
            inner_dim_to_multiple=self._config.ffn_inner_dim_to_multiple,
            device=self._device,
            dtype=self._dtype,
        )

    def build_layer_norm(
            self,
            model_dim: int,
            *,
            device: Optional[Device] = None,
            dtype: Optional[DataType] = None,
    ) -> LayerNorm:
        """Build a Layer Normalization module."""
        return RMSNorm(model_dim, bias=False, device=device, dtype=dtype)


def create_llama_model(
        config: LLaMAConfig,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
) -> TransformerDecoderModel:
    """Create a LLaMA model.

    :param config:
        The configuration.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    return LLaMABuilder(config, device=device, dtype=dtype).build_model()



def create_llama_7b_model(
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
):

    config = _llama2_7b()

    model = create_llama_model(config, device=device, dtype=dtype)

    return model


from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Union
import re

def convert_model_state_dict(
    state_dict: Dict[str, Any], key_map: Mapping[str, str]
) -> Dict[str, Any]:
    """Convert a model state dictionary to fairseq2.

    :param state_dict:
        The original model state dictionary.
    :param key_map:
        A map of regex patterns to fairseq2 model keys.

    :returns:
        A converted model state dictionary that is compatible with fairseq2.
    """
    new_state_dict = {}

    def get_new_key(old_key: str) -> str:
        for old_pattern, replacement in key_map.items():
            if (new_key := re.sub(old_pattern, replacement, old_key)) != old_key:
                return new_key

        return old_key

    # Convert module keys from fairseq to fairseq2.
    for old_key in state_dict.keys():
        new_key = get_new_key(old_key)

        new_state_dict[new_key] = state_dict[old_key]

    return new_state_dict



def convert_llama_checkpoint(
    checkpoint: Dict[str, Any] # , config: LLaMAConfig
) -> Dict[str, Any]:
    """Convert a reference LLaMA checkpoint to fairseq2 format."""
    # Check if we have a fairseq2 checkpoint.
    if "model" in checkpoint:
        return checkpoint

    key_map = {
        # fmt: off
        r"^layers\.([0-9]+)\.attention\.wq\.":    r"decoder.layers.\1.self_attn.q_proj.",
        r"^layers\.([0-9]+)\.attention\.wk\.":    r"decoder.layers.\1.self_attn.k_proj.",
        r"^layers\.([0-9]+)\.attention\.wv\.":    r"decoder.layers.\1.self_attn.v_proj.",
        r"^layers\.([0-9]+)\.attention\.wo\.":    r"decoder.layers.\1.self_attn.output_proj.",
        r"^layers\.([0-9]+)\.attention_norm\.":   r"decoder.layers.\1.self_attn_layer_norm.",
        r"^layers\.([0-9]+)\.feed_forward\.w1\.": r"decoder.layers.\1.ffn.gate_proj.",
        r"^layers\.([0-9]+)\.feed_forward\.w2\.": r"decoder.layers.\1.ffn.output_proj.",
        r"^layers\.([0-9]+)\.feed_forward\.w3\.": r"decoder.layers.\1.ffn.inner_proj.",
        r"^layers\.([0-9]+)\.ffn_norm\.":         r"decoder.layers.\1.ffn_layer_norm.",
        r"^norm\.":                               r"decoder.layer_norm.",
        r"^tok_embeddings\.":                     r"decoder_frontend.embed.",
        r"^output\.":                             r"final_proj.",
        # fmt: on
    }

    # We do not need the pre-computed 'rope.freqs' buffers.
    checkpoint = {k: v for (k, v) in checkpoint.items() if "rope.freqs" not in k}

    checkpoint = convert_model_state_dict(checkpoint, key_map)

    return {"model": checkpoint}

