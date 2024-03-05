# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import GELU, Conv1d, Dropout, GroupNorm, Module, Sequential
from torch.nn.functional import group_norm, layer_norm

from ..fairseq_wav2vec2.fairseq_modules import Fp32LayerNorm, Fp32GroupNorm
from onmt.nn.normalization import LayerNorm, create_standard_layer_norm
from .typing import final, finaloverride, override


class SequenceFeatureExtractor(Module, ABC):
    """Extracts features from sequences and embeds them in a latent space."""

    feature_dim: int

    def __init__(self, feature_dim: int) -> None:
        """
        :param feature_dim:
            The dimensionality of extracted features.
        """
        super().__init__()

        self.feature_dim = feature_dim

    @abstractmethod
    def forward(
            self, seqs: Tensor, padding_mask=None
    ):
        """
        :param seqs:
            The sequences from which to extract features. *Shape:*
            :math:`(N,S,*)`, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`*` is any number of sequence-specific
            dimensions including none.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.

        :returns:
            - The extracted features. *Shape:* :math:`(N,S_{out},F)`, where
              :math:`N` is the batch size, :math:`S_{out}` is the output
              sequence length, and :math:`F` is the dimensionality of the
              features.
            - The padding mask of the extracted features. *Shape:*
              :math:`(N,S_{out})`, where :math:`N` is the batch size and
              :math:`S_{out}` is the output sequence length.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"feature_dim={self.feature_dim}"


class Wav2Vec2FbankFeatureExtractor(SequenceFeatureExtractor):
    num_fbank_channels: int
    stride: int
    sample_every_k: int

    def __init__(
            self, num_fbank_channels: int, stride: int, *, sample_every_k: int = 1
    ):
        super().__init__(feature_dim=num_fbank_channels * stride)

        self.num_fbank_channels = num_fbank_channels
        self.stride = stride
        self.sample_every_k = sample_every_k

    @finaloverride
    def forward(
            self, seqs: Tensor, padding_mask
    ):
        """See the base :meth:`SequenceFeatureExtractor.forward`.

        :param seqs:
            The input log-mel filterbanks. *Shape:* :math:`(N,S,C)`, where
            :math:`N` is the batch size, :math:`S` is the number of frames, and
            :math:`C` is the number of channels.
        """

        batch_size, num_frames, num_channels = seqs.shape

        if padding_mask is None:
            seq_lens = None
        else:
            # padding mask dimension is B x T where elements of the masked positions = 1
            seq_lens = (1 - padding_mask.long()).sum(dim=1)

        if (r := num_frames % self.stride) != 0:
            num_frames -= r

            seqs = seqs[:, :num_frames, :]

            if seq_lens is not None:
                seq_lens = seq_lens.clone()

                seq_lens[seq_lens > num_frames] = num_frames

        seqs = seqs.view(
            batch_size, num_frames // self.stride, num_channels * self.stride
        )

        if self.sample_every_k > 1:
            indices = torch.arange(0, batch_size, device=seqs.device)

            seqs = seqs[indices % self.sample_every_k != 0]

        if seq_lens is not None:
            # Since we contracted the temporal dimension, we should re-compute
            # the sequence lengths.
            seq_lens = self._contract_seq_lens(seq_lens)

            batch_seq_len = seqs.size(1)
            indices = torch.arange(batch_seq_len, device=seq_lens.device).expand(batch_size, -1)
            lengths = seq_lens.unsqueeze(1).expand(-1, batch_seq_len)

            padding_mask = (indices > lengths)

        return seqs, padding_mask

    def _contract_seq_lens(self, num_frames: Tensor) -> Tensor:
        num_frames = num_frames // self.stride

        if self.sample_every_k > 1:
            num_frames //= self.sample_every_k + 1

        return num_frames

    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"num_fbank_channels={self.num_fbank_channels}, "
            f"stride={self.stride}, "
            f"sample_every_k={self.sample_every_k}"
        )


class Wav2Vec2FeatureExtractionLayer(Module):
    """Represents a feature extraction layer used in
    :class:`Wav2Vec2FeatureExtractor`."""

    conv: Conv1d
    dropout: Optional[Dropout]
    group_norm: Optional[GroupNorm]
    layer_norm: Optional[LayerNorm]
    activation: GELU

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            kernel_size: int,
            stride: int,
            bias: bool,
            *,
            dropout_p: float = 0.0,
            group_norm: Optional[GroupNorm] = None,
            layer_norm: Optional[LayerNorm] = None,
            device=None,
            dtype=None,
    ) -> None:
        super().__init__()

        self.conv = Wav2Vec2FeatureConv1d(
            input_dim,
            output_dim,
            kernel_size,
            stride=stride,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        if dropout_p > 0.0:
            self.dropout = Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

        if group_norm is not None:
            self.group_norm = group_norm
        else:
            self.register_module("group_norm", None)

        if layer_norm is not None:
            self.layer_norm = layer_norm
        else:
            self.register_module("layer_norm", None)

        self.activation = GELU()

    def forward(self, seqs: Tensor) -> Tensor:
        # (N, C_inp, S) -> (N, C_out, S)
        seqs = self.conv(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        if self.group_norm is not None:
            seqs = self.group_norm(seqs)

        if self.layer_norm is not None:
            seqs = seqs.transpose(1, 2)

            seqs = self.layer_norm(seqs)

            seqs = seqs.transpose(1, 2)

        seqs = self.activation(seqs)

        return seqs


class Wav2Vec2FeatureConv1d(Conv1d):
    """Represents the convolution used in
    :class:`Wav2Vec2FeatureExtractionLayer`."""

    @override
    def reset_parameters(self) -> None:
        if self.bias is not None:
            # Call the base since we want to initialize bias as in `Conv1d`.
            super().reset_parameters()

        nn.init.kaiming_normal_(self.weight)


class SequenceFeatureExtractor(Module, ABC):
    """Extracts features from sequences and embeds them in a latent space."""

    feature_dim: int

    def __init__(self, feature_dim: int) -> None:
        """
        :param feature_dim:
            The dimensionality of extracted features.
        """
        super().__init__()

        self.feature_dim = feature_dim

    @abstractmethod
    def forward(
            self, seqs: Tensor, padding_mask
    ):
        """
        :param seqs:
            The sequences from which to extract features. *Shape:*
            :math:`(N,S,*)`, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`*` is any number of sequence-specific
            dimensions including none.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.

        :returns:
            - The extracted features. *Shape:* :math:`(N,S_{out},F)`, where
              :math:`N` is the batch size, :math:`S_{out}` is the output
              sequence length, and :math:`F` is the dimensionality of the
              features.
            - The padding mask of the extracted features. *Shape:*
              :math:`(N,S_{out})`, where :math:`N` is the batch size and
              :math:`S_{out}` is the output sequence length.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"feature_dim={self.feature_dim}"


@final
class Wav2Vec2FeatureExtractor(SequenceFeatureExtractor):
    """Extracts features from raw audio waveforms and embeds them in a latent
    space as described in Section 2 of
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    layers: Sequential
    layer_descs: List[Tuple[int, int, int]]
    grad_scale: float

    def __init__(
            self,
            layer_descs: Sequence[Tuple[int, int, int]],
            bias: bool,
            *,
            dropout_p: float = 0.0,
            layer_norm: bool = False,
            grad_scale: float = 1.0,
            device=None,
            dtype=None,
    ) -> None:
        """
        :param layer_descs:
            A tuple of output dimension, kernel size, and stride for each
            feature extraction layer.
        :param bias:
            If ``True``, convolutions learn an additive bias.
        :param dropout_p:
            The dropout probability on outputs of convolutions.
        :param layer_norm:
            If ``True``, applies Layer Normalization to outputs of convolutions
            after dropout.
        :param grad_scale:
            The scale factor for gradients of extracted features. Setting to a
            value less than 1.0 allows the feature extractor to learn at a lower
            rate than the rest of the model.
        """

        # The output dimensionality of the last feature extraction layer.
        feature_dim = layer_descs[-1][0]

        super().__init__(feature_dim)

        self.layers = Sequential()

        # We expect the input waveforms to be one dimensional.
        input_dim = 1

        for i, layer_desc in enumerate(layer_descs):
            output_dim, kernel_size, stride = layer_desc

            # If Layer Normalization is requested, apply it in all layers.
            if layer_norm:
                layer_norm_ = Fp32LayerNorm(
                    output_dim, bias=True, device=device, dtype=dtype
                )

                group_norm_ = None

            # Otherwise, apply Group Normalization in the first layer, and do
            # not apply any normalization in other layers.
            elif i == 0:
                group_norm_ = Float32GroupNorm(
                    output_dim, output_dim, device=device, dtype=dtype
                )

                layer_norm_ = None
            else:
                group_norm_ = None
                layer_norm_ = None

            layer = Wav2Vec2FeatureExtractionLayer(
                input_dim,
                output_dim,
                kernel_size,
                stride,
                bias,
                dropout_p=dropout_p,
                group_norm=group_norm_,
                layer_norm=layer_norm_,
                device=device,
                dtype=dtype,
            )
