# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple, final

from torch import Tensor
from torch.nn import Dropout, Module

from ..speech_recognizer.w2v_bert.frontend import TransformerFrontend
from ..speech_recognizer.w2v_bert.typing import DataType, Device, finaloverride
from onmt.modules.layer_norm import LayerNorm
from onmt.nn.position_encoder import PositionEncoder
from onmt.nn.embedding import Embedding


@final
class TransformerEmbeddingFrontend(TransformerFrontend):
    """Represents a Transformer encoder/decoder front-end as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    embed: Embedding
    scale: float
    pos_encoder: Optional[PositionEncoder]
    layer_norm: Optional[LayerNorm]
    dropout: Optional[Dropout]

    def __init__(
        self,
        embed: Embedding,
        pos_encoder: Optional[PositionEncoder],
        *,
        no_scale: bool = False,
        layer_norm: bool = False,
        dropout_p: float = 0.1,
        # layer_norm_factory: Optional[LayerNormFactory] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param embed:
            The token embedding table.
        :param pos_encoder:
            The position encoder.
        :param no_scale:
            If ``True``, does not scale embeddings by the square root of the
            embedding size.
        :param layer_norm:
            If ``True``, applies Layer Normalization to embeddings before
            dropout.
        :param dropout_p:
            The dropout probability on embeddings.
        :param layer_norm_factory:
            The factory to construct the Layer Normalization module.
        """
        model_dim = embed.embedding_dim

        super().__init__(model_dim)

        self.embed = embed

        self.scale = 1.0 if no_scale else math.sqrt(model_dim)

        if pos_encoder is not None:
            if pos_encoder.encoding_dim != model_dim:
                raise ValueError(
                    f"`encoding_dim` of `pos_encoder` and `embedding_dim` of `embed` must be equal, but are {pos_encoder.encoding_dim} and {model_dim} instead."
                )

            self.pos_encoder = pos_encoder
        else:
            self.register_module("pos_encoder", None)

        if layer_norm:
            self.layer_norm = LayerNorm(model_dim, device=device, dtype=dtype)
        else:
            self.register_module("layer_norm", None)

        if dropout_p > 0.0:
            self.dropout = Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        **kwargs
        # state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        embeds = self.embed(seqs)

        if self.scale != 1.0:
            embeds = embeds * self.scale

        if self.pos_encoder is not None:
            # embeds = self.pos_encoder(embeds, padding_mask, state_bag=state_bag)
            embeds = self.pos_encoder(embeds, padding_mask)

        if self.layer_norm is not None:
            embeds = self.layer_norm(embeds)

        if self.dropout is not None:
            embeds = self.dropout(embeds)

        return embeds, padding_mask

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        if self.scale != 1.0:
            s = f"{s}, no_scale=False"

        return s