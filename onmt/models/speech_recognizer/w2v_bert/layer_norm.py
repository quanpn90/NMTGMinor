# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Protocol

from onmt.modules.layer_norm import LayerNorm
from .typing import DataType, Device


class LayerNormFactory(Protocol):
    """Constructs instances of :class:`LayerNorm`."""

    def __call__(
            self,
            model_dim: int,
            *,
            device: Optional[Device] = None,
            dtype: Optional[DataType] = None,
    ):
        """
        :param model_dim:
            The dimensionality of the model.
        :param device:
            The device on which to initialize the module.
        :param dtype:
            The data type of the module.
        """


def create_standard_layer_norm(
        model_dim: int, *, bias = True, device: Optional[Device] = None, dtype: Optional[DataType] = None
):
    """Create an instance of :class:`StandardLayerNorm`."""
    return LayerNorm(model_dim, bias=bias, device=device, dtype=dtype)
