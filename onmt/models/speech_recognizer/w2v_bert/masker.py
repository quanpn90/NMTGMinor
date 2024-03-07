# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from onmt.typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter

# from fairseq2.nn.padding import PaddingMask
# from fairseq2.nn.utils.mask import compute_row_mask
from onmt.typing import DataType, Device


def repeat_interleave(x: Tensor, dim: int, repeat: int) -> Tensor:
    """Repeat elements of a tensor.

    :param x:
        The input tensor.
    :param dim:
        The dimension along which to repeat values.
    :param repeat:
        The number of repetitions.

    :returns:
        The repeated tensor which has the same shape as input, except along the
        given axis.

    .. note::
        This is a lightweight version of :func:`torch.repeat_interleave` that
        is faster for repetitions along a single dimension.
    """
    if repeat == 1:
        return x

    shape = [-1] * (x.ndim + 1)

    if dim < 0:
        dim += x.ndim

    shape[dim + 1] = repeat

    return x.unsqueeze(dim + 1).expand(shape).flatten(dim, dim + 1)


def compute_row_mask(
        shape: Tuple[int, int],
        span_len: int,
        max_mask_prob: float,
        row_lens: Optional[Tensor] = None,
        min_num_spans: int = 0,
        device: Optional[Device] = None,
) -> Optional[Tensor]:
    """Compute a random row mask of the specified shape.

    :param shape:
        The shape of the mask.
    :param span_len:
        The length of each mask span.
    :param max_mask_prob:
        The maximum probability of masking an element among all elements in a
        row. Note that, due to mask span overlap, the effective probability
        might be smaller. The implementation also guarantees that there is
        always at least one unmasked element in each row.
    :param row_lens:
        The length of each row. *Shape:* :math:`(R)`, where :math:`R` is the
        number of rows.
    :param min_num_spans:
        The minimum number of mask spans per row.
    :param device:
        The device on which to initialize the mask.

    :returns:
        The boolean row mask. *:Shape:* ``shape``.
    """
    num_rows, max_row_len = shape

    if row_lens is None:
        # We only mask rows that are longer than the mask span length.
        if span_len >= max_row_len:
            raise ValueError(
                f"The size of the second dimension of `shape` must be greater than `span_len` ({span_len}), but is {max_row_len} instead."
            )

        row_lens = torch.full(
            (num_rows,), max_row_len, device=device, dtype=torch.int64
        )
    else:
        row_lens = row_lens.view(num_rows)

        # We only mask rows that are longer than the mask span length.
        if (span_len >= row_lens).any():
            raise ValueError(
                f"All lengths in `row_lens` must be greater than `span_len` ({span_len}), but at least one length is smaller. row_lens: {row_lens}"
            )

    indices = _compute_mask_spans(row_lens, span_len, max_mask_prob, min_num_spans)
    if indices is None:
        return row_lens.new_empty((0, 0))

    return _generate_mask(indices, max_row_len).to(device)


def _compute_mask_spans(
    row_lens: Tensor, span_len: int, max_mask_prob: float, min_num_spans: int
) -> Optional[Tensor]:
    """Compute random mask spans of the specified shape."""
    device, dtype = row_lens.device, row_lens.dtype

    num_rows = len(row_lens)
    if num_rows == 0:
        return None

    # Compute the number of mask spans per row. We should always have at least
    # one unmasked element; this is why we substract 1 from `row_lens`.
    num_spans_per_row = (max_mask_prob / span_len) * (row_lens - 1)

    # Require the same number of mask spans for all rows.
    num_spans = int(num_spans_per_row.to(dtype).min())

    if min_num_spans > num_spans:
        raise ValueError(
            f"`min_num_spans` is {min_num_spans}, but with the given `span_len` and `max_mask_prob` only {num_spans} mask span(s) can be generated."
        )

    if num_spans == 0:
        return None

    # The range of possible start indices for mask spans in form [0, max + 1).
    span_start_range = row_lens - span_len + 1

    # (R) -> (R x N)
    span_start_range = repeat_interleave(span_start_range, dim=0, repeat=num_spans)

    # Unlike the fairseq implementation, we do sample with replacement, which is
    # more consistent with the overlap strategy.
    # (R x N)
    rand_scales = torch.rand(num_rows * num_spans, device=device)

    # By random scaling we effectively pick a random start index for each mask
    # span.
    span_offsets = span_start_range * rand_scales

    # The following ops convert the mask span offsets (i.e. start indices) to
    # mask spans (i.e. index ranges).
    # (R x N) -> (R, N)
    span_offsets = span_offsets.to(dtype).view(num_rows, -1)

    # (R, N) -> (R, N x L)
    span_offsets = repeat_interleave(span_offsets, dim=-1, repeat=span_len)

    # (L)
    indices = torch.arange(span_len, device=device, dtype=dtype)

    # (L) -> (R, N x L)
    indices = indices.repeat(num_spans).unsqueeze(0).expand(num_rows, -1)

    return span_offsets + indices


class Wav2Vec2Masker(Module):
    """Masks extracted features as described in Section 3.1 of
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    temporal_span_len: int
    max_temporal_mask_prob: float
    temporal_mask_embed: Parameter
    spatial_span_len: int
    max_spatial_mask_prob: float

    def __init__(
            self,
            model_dim: int,
            temporal_span_len: int = 10,
            max_temporal_mask_prob: float = 0.65,
            spatial_span_len: int = 10,
            max_spatial_mask_prob: float = 0.0,
            *,
            device: Optional[Device] = None,
            dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param temporal_span_len:
            The length of each temporal mask span that is applied over time
            steps.
        :param max_temporal_mask_prob:
            The maximum probability of masking a time step. Note that, due to
            mask span overlap, the effective probability might be smaller.
        :param spatial_span_len:
            The length of each spatial mask span that is applied over features.
        :param max_spatial_mask_prob:
            The maximum probability of masking a feature. Note that, due to mask
            span overlap, the effective probability might be smaller.
        """
        super().__init__()

        if max_temporal_mask_prob == 0.0:
            raise ValueError("`max_temporal_mask_prob` must be greater than 0.")

        self.temporal_span_len = temporal_span_len
        self.max_temporal_mask_prob = max_temporal_mask_prob

        self.temporal_mask_embed = Parameter(
            torch.empty((model_dim,), device=device, dtype=dtype)
        )

        self.spatial_span_len = spatial_span_len
        self.max_spatial_mask_prob = max_spatial_mask_prob

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        nn.init.uniform_(self.temporal_mask_embed)

    def forward(
            self, seqs: Tensor, padding_mask
    ) -> Tuple[Tensor, Tensor]:
        """
        :param seqs:
            The sequences to mask. *Shape:* :math:`(N,S,M)`, where :math:`N` is
            the batch size, :math:`S` is the sequence length, and :math:`M` is
            the dimensionality of the model.
        :param seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``seqs``. *Shape:* :math:`(N)`, where :math:`N` is
            the batch size.

        :returns:
            - The input sequences with mask applied. *Shape:* Same as ``seqs``.
            - The temporal mask that has been applied to ``seqs``. *Shape:*
              :math:`(N,S)`, where :math:`N` is the batch size and :math`S` is
              the sequence length.
        """
        batch_size, seq_len, model_dim = seqs.shape

        # Temporal mask over time steps.
        temporal_mask = compute_row_mask(
            shape=(batch_size, seq_len),
            span_len=self.temporal_span_len,
            max_mask_prob=self.max_temporal_mask_prob,
            row_lens=padding_mask.seq_lens if padding_mask is not None else None,
            min_num_spans=2,
            device=seqs.device,
        )

        assert temporal_mask is not None

        seqs[temporal_mask] = self.temporal_mask_embed

        if self.max_spatial_mask_prob > 0.0:
            # Spatial mask over features.
            # (N, M)
            spatial_mask = compute_row_mask(
                shape=(batch_size, model_dim),
                span_len=self.spatial_span_len,
                max_mask_prob=self.max_spatial_mask_prob,
                min_num_spans=2,
                device=seqs.device,
            )

            assert spatial_mask is not None

            # (N, M) -> (N, S, M)
            spatial_mask = spatial_mask.unsqueeze(1).expand(-1, seq_len, -1)

            seqs[spatial_mask] = 0.0

        return seqs, temporal_mask

    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"temporal_span_len={self.temporal_span_len}, "
            f"max_temporal_mask_prob={self.max_temporal_mask_prob}, "
            f"spatial_span_len={self.spatial_span_len}, "
            f"max_spatial_mask_prob={self.max_spatial_mask_prob}"
        )


def extract_masked_elements(seqs: Tensor, temporal_mask: Tensor) -> Tensor:
    """Extract masked elements from ``seqs``.

    :param seqs:
        The sequences. *Shape:* :math:`(N,S,M)`, where :math:`N` is the batch
        size, :math:`S` is the sequence length, and :math:`M` is the
        dimensionality of the model.
    :param temporal_mask:
        The temporal mask. *Shape:* :math:`(N,S)`, where :math:`N` is the batch
        size and :math`S` is the sequence length.
    """
    batch_size = seqs.size(0)

    # (N, S, M) -> (N x T, M)
    seqs = seqs[temporal_mask]

    # (N x T, M) -> (N, T, M)
    return seqs.unflatten(0, (batch_size, -1))  # type: ignore[no-any-return]
