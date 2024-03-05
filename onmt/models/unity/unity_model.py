# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import cross_entropy

from ..speech_recognizer.w2v_bert.frontend import Wav2Vec2Frontend
from onmt.nn.projection import Linear
from onmt.nn.transformer.encoder import TransformerEncoder
from ..speech_recognizer.w2v_bert.typing import DataType, Device


class Wav2Vec2ModelWithAdaptor(Module):
    """Represents a wav2vec 2.0 model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    model_dim: int
    encoder_frontend: Wav2Vec2Frontend
    encoder: TransformerEncoder

    # final_proj: Linear
    # final_target_proj: Linear
    # num_distractors: int
    # logit_temp: float
    # diversity_loss_weight: float

    def __init__(
        self,
        encoder_frontend: Wav2Vec2Frontend,
        encoder: TransformerEncoder,
        *,
        device = None,
        dtype = None,
    ) -> None:
        """
        :param encoder_frontend:
            The encoder frontend.
        :param encoder:
            The encoder (i.e. context network).
        """
        super().__init__()

        model_dim = encoder.model_dim

        self.model_dim = model_dim

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder

    def forward(self, audio_input, padding_mask):
        """
        :param batch:
            The batch of sequences to process.
        """
        # seqs, padding_mask, targets, temporal_mask = self.run_frontend(
        #     audio_input, padding_mask
        # )
        seqs, padding_mask = self.encoder_frontend(audio_input, padding_mask)

        # TODO: Should pad for fp16?
        seqs, padding_mask = self.encoder(seqs, padding_mask)

        return seqs, padding_mask

    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"model_dim={self.model_dim}, "
        )