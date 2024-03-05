# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from abc import ABC, abstractmethod
from typing import Optional, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import embedding
from torch.nn.parameter import Parameter

from ..speech_recognizer.w2v_bert.typing import META, DataType, Device, finaloverride
from onmt.nn.position_encoder import PositionEncoder


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
