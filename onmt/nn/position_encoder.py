# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, final
from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor
from torch.nn import GELU, Conv1d, Module, Sequential
from torch.nn.utils.weight_norm import remove_weight_norm, weight_norm
import overrides
from overrides import override as override
from onmt.typing import Device, DataType, META

finaloverride = overrides.final

from onmt.nn.normalization import LayerNorm, create_standard_layer_norm


class PositionEncoder(Module, ABC):
    """Encodes sequences with positional information."""

    encoding_dim: int
    max_seq_len: Optional[int]

    def __init__(self, encoding_dim: int, max_seq_len: Optional[int]) -> None:
        """
        :param encoding_dim:
            The dimensionality of positional encodings.
        :param max_seq_len:
            The expected maximum sequence length.
        """
        super().__init__()

        self.encoding_dim = encoding_dim
        self.max_seq_len = max_seq_len

    def forward(
            self,
            seqs: Tensor,
            padding_mask,
            *,
            state_bag=None,
    ) -> Tensor:
        """
        :param seqs:
            The sequences to encode with positional information. *Shape:*
            :math:`(*,S,E)`, where :math:`*` is any number of batch dimensions
            including none, :math:`S` is the sequence length, and :math:`E` is
            the dimensionality of the positional encodings.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(*,S)`, where :math:`*`
            is any number of batch dimensions including none and :math:`S` is
            the sequence length.
        :param state_bag:
            The state bag to use for incremental decoding.

        :returns:
            The input sequences with positional information encoded. *Shape:*
            Same as ``seqs``.
        """
        if self.max_seq_len is not None:
            if self.training or state_bag is None:
                start_step = 0
            else:
                start_step = state_bag.step_nr

            if (seq_len := start_step + seqs.size(-2)) > self.max_seq_len:
                raise ValueError(
                    f"The input sequence length must be less than or equal to the maximum sequence length ({self.max_seq_len}), but is {seq_len} instead."
                )

        return self._do_forward(seqs, padding_mask, state_bag)

    @abstractmethod
    def _do_forward(
            self,
            seqs: Tensor,
            padding_mask,
            state_bag,
    ) -> Tensor:
        """
        :param seqs:
            The sequences to encode with positional information. *Shape:*
            :math:`(*,S,E)`, where :math:`*` is any number of batch dimensions
            including none, :math:`S` is the sequence length, and :math:`E` is
            the dimensionality of the positional encodings.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(*,S)`, where :math:`*`
            is any number of batch dimensions including none and :math:`S` is
            the sequence length.
        :param state_bag:
            The state bag to use for incremental decoding.

        :returns:
            The input sequences with positional information encoded. *Shape:*
            Same as ``seqs``.

        :meta public:
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        s = f"encoding_dim={self.encoding_dim}"

        if self.max_seq_len is not None:
            s = f"{s}, max_seq_len={self.max_seq_len}"

        return s


@final
class Wav2Vec2PositionEncoder(PositionEncoder):
    """Encodes sequences with relative positional information as described in
    Section 2 of :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    conv: Conv1d
    remove_pad: bool
    activation: GELU

    def __init__(
            self,
            model_dim: int,
            kernel_size: int,
            num_groups: int,
            *,
            device=None,
            dtype=None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param kernel_size:
            The kernel size of the 1D convolution.
        :param num_groups:
            The number of convolution groups.
        """
        super().__init__(model_dim, max_seq_len=None)

        self.conv = Wav2Vec2PositionalConv1d(
            model_dim,
            model_dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=num_groups,
            device=device,
            dtype=dtype,
        )

        self.remove_pad = kernel_size % 2 == 0

        self.activation = GELU()

    @finaloverride
    def _do_forward(
            self,
            seqs: Tensor,
            padding_mask,
            state_bag,
    ) -> Tensor:
        """:meta private:"""
        if state_bag is not None:
            raise ValueError(
                "`Wav2Vec2PositionEncoder` does not support incremental decoding."
            )

        # We have to ensure that the padded elements are correctly set to
        # zero; otherwise, noise will leak into the feature maps.
        # seqs = apply_padding_mask(seqs, padding_mask)
        if padding_mask is not None:
            seqs = seq.masked_fill_(padding_mask, 0)

        # (N, S, E) -> (N, E, S)
        encodings = seqs.transpose(1, 2)

        # (N, E, S) -> (N, E, S)
        encodings = self.conv(encodings)

        if self.remove_pad:
            encodings = encodings[:, :, :-1]

        encodings = self.activation(encodings)

        # (N, E, S) -> (N, S, E)
        encodings = encodings.transpose(1, 2)

        return seqs + encodings


class Wav2Vec2PositionalConv1d(Conv1d):
    """Represents the convolution used in :class:`Wav2Vec2PositionEncoder`."""
    """
    Note: this module uses weight normalization
    """

    @override
    def reset_parameters(self) -> None:
        model_dim, kernel_size = self.in_channels, self.kernel_size[0]

        try:
            remove_weight_norm(self)
        except ValueError:
            # Raised during the `__init__` call since we don't have the weight
            # norm hook registered yet. Safe to ignore.
            pass

        nn.init.normal_(
            self.weight, mean=0.0, std=(4.0 / (kernel_size * model_dim)) ** 0.5
        )

        weight_norm(self, dim=2)

        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


@final
class Wav2Vec2StackedPositionEncoder(PositionEncoder):
    """Encodes sequences with relative positional information using a stack
    of 1D convolutions.

    This position encoder is not mentioned in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`, but exists in the
    reference implementation.
    """

    layers: Sequential

    def __init__(
            self,
            model_dim: int,
            kernel_size: int,
            num_groups: int,
            num_layers: int,
            *,
            device=None,
            dtype=None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param kernel_size:
            The total kernel size of the 1D convolutions. Each convolution uses
            a kernel size of ``max(3, kernel_size // num_layers)``.
        :param num_groups:
            The number of convolution groups.
        :param num_layers:
            The number of convolution layers.
        """
        super().__init__(model_dim, max_seq_len=None)

        k = max(3, kernel_size // num_layers)

        self.layers = Sequential()

        for _ in range(num_layers):
            layer = Wav2Vec2PositionEncoderLayer(
                model_dim,
                k,
                num_groups,
                device=device,
                dtype=dtype,
            )

            self.layers.append(layer)

    @finaloverride
    def _do_forward(
            self,
            seqs: Tensor,
            padding_mask,
            state_bag=None,
    ) -> Tensor:
        """:meta private:"""
        if state_bag is not None:
            raise ValueError(
                "`Wav2Vec2StackedPositionEncoder` does not support incremental decoding."
            )

        # We have to ensure that the padded elements are correctly set to
        # zero; otherwise, noise will leak into the feature maps.
        # seqs = apply_padding_mask(seqs, padding_mask)
        if padding_mask is not None:
            seqs.masked_fill_(padding_mask, 0)

        # (N, S, E) -> (N, E, S)
        encodings = seqs.transpose(1, 2)

        # (N, E, S) -> (N, E, S)
        encodings = self.layers(encodings)

        # (N, E, S) -> (N, S, E)
        encodings = encodings.transpose(1, 2)

        return seqs + encodings


class Wav2Vec2PositionEncoderLayer(Module):
    """Represents a layer used in :class:`Wav2Vec2StackedPositionEncoder`."""

    conv: Conv1d
    layer_norm: LayerNorm
    activation: GELU

    def __init__(
            self,
            model_dim: int,
            kernel_size: int,
            num_groups: int,
            *,
            device=None,
            dtype=None,
    ) -> None:
        super().__init__()

        self.conv = Conv1d(
            model_dim,
            model_dim,
            kernel_size,
            padding="same",
            groups=num_groups,
            device=device,
            dtype=dtype,
        )

        self.layer_norm = create_standard_layer_norm(
            model_dim, bias=True, elementwise_affine=False, device=device, dtype=dtype
        )

        self.activation = GELU()

    def forward(self, encodings: Tensor) -> Tensor:
        # (N, E, S) -> (N, E, S)
        encodings = self.conv(encodings)

        # (N, E, S) -> (N, S, E)
        encodings = encodings.transpose(1, 2)

        encodings = self.layer_norm(encodings)

        # (N, S, E) -> (N, E, S)
        encodings = encodings.transpose(1, 2)

        encodings = self.activation(encodings)

        return encodings


@final
class SinusoidalPositionEncoder(PositionEncoder):
    """Encodes sequences with fixed sinusoidal positional information.

    The positional encodings are initialized as in tensor2tensor which differs
    slightly from the description in section 3.5 of
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`. This means instead of

    .. math::
        PE_{(pos, 2i)}   = \\text{sin}(pos/10000^{2i/d_{model}})

        PE_{(pos, 2i+1)} = \\text{cos}(pos/10000^{2i/d_{model}})

    we use

    .. math::
        PE_{(pos, i)} = \\text{sin}(pos/10000^{i/d_{model}})\\;\\text{for}\\;i\\;    <\\frac{d_{model}}{2}

        PE_{(pos, i)} = \\text{cos}(pos/10000^{i/d_{model}})\\;\\text{for}\\;i\\;\\geq\\frac{d_{model}}{2}

    See `here <https://github.com/tensorflow/tensor2tensor/pull/177>`_ for more
    information.

    Usage:

    >>> import torch
    >>>
    >>> from fairseq2.nn.position_encoder import SinusoidalPositionEncoder
    >>>
    >>> m = SinusoidalPositionEncoder(encoding_dim=4, max_seq_len=16)
    >>>
    >>> seqs = torch.ones((3, 4))
    >>>
    >>> m(seqs)
    tensor([[ 1.0000e+00,  1.0000e+00,  2.0000e+00,  2.0000e+00],  # pos 0
            [ 9.4147e-01,  2.0000e-04,  6.4030e-01,  2.0000e+00],  # pos 1
            [ 1.0930e-02,  3.0000e-04, -5.1615e-01,  2.0000e+00]]) # pos 2
    """

    freqs: Tensor

    def __init__(
            self,
            encoding_dim: int,
            max_seq_len: int,
            *,
            _legacy_pad_idx: Optional[int] = None,
            device: Optional[Device] = None,
    ) -> None:
        super().__init__(encoding_dim, max_seq_len)

        if encoding_dim % 2 != 0:
            raise ValueError(
                f"`encoding_dim` must be even, but is {encoding_dim} instead."
            )

        # This is a legacy parameter that should only be set when the encodings
        # must be compatible with fairseq.
        if _legacy_pad_idx is None:
            self._sin_offset = 0
        else:
            self._sin_offset = 1 + _legacy_pad_idx

        freqs = torch.empty(
            (max_seq_len, encoding_dim), device=device, dtype=torch.float32
        )

        self.register_buffer("freqs", freqs, persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        self.reset_non_persistent_buffers()

    def reset_non_persistent_buffers(self) -> None:
        """Reset the non-persistent buffers of the module."""
        num_sin = self.encoding_dim // 2

        device, dtype = self.freqs.device, self.freqs.dtype

        l_half = self.freqs[:, :num_sin]
        r_half = self.freqs[:, num_sin:]

        start_step = self._sin_offset

        assert self.max_seq_len is not None

        # (S)
        steps = torch.arange(
            start_step, start_step + self.max_seq_len, device=device, dtype=dtype
        )

        # (E)
        indices = torch.arange(num_sin, device=device, dtype=dtype)

        # This is identical to tensor2tensor's implementation.
        freqs = torch.exp(indices * -math.log(10000.0) / (num_sin - 1))

        # (S) x (E) -> (S, E)
        torch.outer(steps, freqs, out=l_half)

        r_half.copy_(l_half)

        l_half.sin_()
        r_half.cos_()

    @finaloverride
    def _do_forward(
            self,
            seqs: Tensor,
            padding_mask,
            step_number=0
            # state_bag: Optional[IncrementalStateBag],
    ) -> Tensor:
        """
        Args:
            seqs: torch Tensor with size [B x T x H] (word embeddings)
            padding_mask:
            step_number:

        Returns:

        """
        seq_len = seqs.size(-2)

        if self.training or step_number == 0:
            start_step = 0
        else:
            start_step = step_number

        fp32_seqs = seqs.float() + self.freqs[start_step: start_step + seq_len]

        return fp32_seqs.type_as(seqs)


@final
class RotaryEncoder(PositionEncoder):
    """Encodes sequences with relative positional information as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2104.09864`."""

    freqs: Tensor

    def __init__(
        self,
        encoding_dim: int,
        max_seq_len: int,
        *,
        device: Optional[Device] = None,
    ) -> None:
        super().__init__(encoding_dim, max_seq_len)

        if encoding_dim % 2 != 0:
            raise ValueError(
                f"`encoding_dim` must be even, but is {encoding_dim} instead."
            )

        freqs = torch.empty(
            (max_seq_len, encoding_dim // 2), device=device, dtype=torch.complex64
        )

        self.register_buffer("freqs", freqs, persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        self.reset_non_persistent_buffers()

    def reset_non_persistent_buffers(self) -> None:
        """Reset the non-persistent buffers of the module."""
        device = self.freqs.device

        assert self.max_seq_len is not None

        # As of PyTorch 2.0, `torch.polar` does not support meta device, but we
        # do not want to lose benefit of lazy initialization.
        if device == META:
            return

        # (S)
        steps = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)

        # (E / 2)
        indices = torch.arange(
            0, self.encoding_dim, step=2, device=device, dtype=torch.float32
        )

        freqs = 1.0 / (10000.0 ** (indices / self.encoding_dim))

        # (S) x (E / 2) -> (S, E / 2)
        freqs = torch.outer(steps, freqs)

        # (S, E / 2)
        torch.polar(torch.ones_like(freqs), freqs, out=self.freqs)

    @override
    def _do_forward(
        self,
        seqs: Tensor,
        padding_mask
    ) -> Tensor:
        """:meta private:"""
        seq_len = seqs.size(-2)

        # if self.training or state_bag is None:
        #     start_step = 0
        # else:
        #     start_stp = state_bag.step_nr
        # TODO: change the starting step based on the incremental states
        start_step = 0

        # (*, S, E) -> (*, S, E / 2, 2)
        seqs = seqs.unflatten(-1, (-1, 2))

        complex_seqs = torch.view_as_complex(seqs.float())

        complex_seqs = complex_seqs * self.freqs[start_step : start_step + seq_len]

        # (*, S, E / 2, 2) -> (*, S, E)
        fp32_seqs = torch.view_as_real(complex_seqs).flatten(-2)

        return fp32_seqs.type_as(seqs)