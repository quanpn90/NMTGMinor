# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional, Iterator, Protocol, Tuple, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Dropout, Module
from torch.nn.parameter import Parameter

from onmt.nn.normalization import LayerNormFactory, LayerNorm
from onmt.models.speech_recognizer.w2v_bert.w2vbert_multihead_attention import MultiheadAttention
from onmt.models.speech_recognizer.w2v_bert.w2vbert_ffn import FeedForwardNetwork
from onmt.models.speech_recognizer.w2v_bert.norm_order import TransformerNormOrder
from onmt.models.speech_recognizer.w2v_bert.typing import DataType, Device, finaloverride, CPU
from onmt.nn.normalization import LayerNormFactory, create_standard_layer_norm


@final
class ModuleList(torch.nn.ModuleList):
    """Holds submodules in a list.

    This class extends :class:`torch.nn.ModuleList` with an extra feature that
    optionally drops a random number of submodules at every iteration during
    training.

    Usage:

    >>> from torch.nn import Module
    >>>
    >>> from fairseq2.nn import ModuleList
    >>>
    >>> layer1 = Module()
    >>> layer2 = Module()
    >>> layer3 = Module()
    >>>
    >>> layers = ModuleList([layer1, layer2, layer3], drop_p=0.5)
    >>>
    >>> for layer in layers.drop_iter():  # This might iterate over layers 1 and 3.
    ...    x = layer(x)
    >>> for layer in layers.drop_iter():  # This might iterate over all layers.
    ...    x = layer(x)
    >>> for layer in layers.drop_iter():  # This might not iterate over any layers.
    ...    x = layer(x)
    """

    drop_p: float

    def __init__(
        self, modules: Optional[Iterable[Module]] = None, *, drop_p: float = 0.0
    ) -> None:
        """
        :param modules:
            An iterable of modules to add.
        :param drop_p:
            The probability of dropping a submodule during training.
        """
        super().__init__(modules)

        self.drop_p = drop_p

    def drop_iter(self) -> Iterator[Module]:
        """Return an iterator that drops a random set of submodules."""
        if self.drop_p > 0.0 and self.training:
            prob_dist = torch.rand(len(self), device=CPU, dtype=torch.float32)
        else:
            prob_dist = None

        for idx, m in enumerate(super().__iter__()):
            if prob_dist is None or prob_dist[idx] > self.drop_p:
                yield m

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        if self.drop_p > 0.0:
            s = f"{s}, drop_p={self.drop_p}"

        return s


class TransformerEncoderLayer(Module, ABC):
    """Represents a Transformer encoder layer."""

    model_dim: int

    def __init__(self, model_dim: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        """
        super().__init__()

        self.model_dim = model_dim

    @abstractmethod
    def forward(
            self,
            seqs: Tensor,
            padding_mask,
            self_attn_mask=None,
    ):
        """
        :param seqs:
            The sequences to process. *Shape:* :math:`(N,S,M)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`M`
            is the dimensionality of the model.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.
        :param self_attn_mask:
            The mask that will be added to attention weights before computing
            the self attention. *Shape:* :math:`([H],S,S)`, where :math:`H` is
            the number of attention heads and :math:`S` is the sequence length.

        :returns:
            - The encoder layer output. *Shape:* Same as ``seqs``.
            - The padding mask of the encoder layer output. *Shape:* Same as
              ``padding_mask``.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


@final
class StandardTransformerEncoderLayer(TransformerEncoderLayer):
    """Represents a Transformer encoder layer as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`.
    """

    self_attn: MultiheadAttention
    self_attn_norm: Optional[LayerNorm]
    self_attn_dropout: Optional[Dropout]
    self_attn_layer_norm: LayerNorm
    ffn: FeedForwardNetwork
    ffn_dropout: Optional[Dropout]
    residual_scale: Optional[Parameter]
    ffn_layer_norm: LayerNorm
    norm_order: TransformerNormOrder

    def __init__(
            self,
            self_attn: MultiheadAttention,
            ffn: FeedForwardNetwork,
            *,
            scale_residual: bool = False,
            dropout_p: float = 0.1,
            norm_order: TransformerNormOrder = TransformerNormOrder.POST,
            layer_norm_factory: Optional[LayerNormFactory] = None,
            device: Optional[Device] = None,
            dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param self_attn:
            The self attention layer.
        :param ffn:
            The feed-forward network.
        :param scale_residual:
            If ``True``, scales residuals before adding them to the output of
            the feed-forward network as described in
            :cite:t:`https://doi.org/10.48550/arxiv.2110.09456`.
        :param dropout_p:
            The dropout probability on outputs of the self attention layer and
            the feed-forward network.
        :param norm_order:
            The Layer Normalization order.
        :param layer_norm_factory:
            The factory to construct the Layer Normalization modules.
        """
        model_dim = self_attn.model_dim

        super().__init__(model_dim)

        if layer_norm_factory is None:
            layer_norm_factory = create_standard_layer_norm

        self_attn_layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)

        if norm_order != TransformerNormOrder.POST:
            self.self_attn_layer_norm = self_attn_layer_norm

        self.self_attn = self_attn

        if norm_order == TransformerNormOrder.PRE_WITH_NORMFORMER:
            self.self_attn_norm = layer_norm_factory(
                model_dim, device=device, dtype=dtype
            )
        else:
            self.register_module("self_attn_norm", None)

        if dropout_p > 0.0:
            self.self_attn_dropout = Dropout(dropout_p)
        else:
            self.register_module("self_attn_dropout", None)

        if norm_order == TransformerNormOrder.POST:
            self.self_attn_layer_norm = self_attn_layer_norm

        ffn_layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)

        if norm_order != TransformerNormOrder.POST:
            self.ffn_layer_norm = ffn_layer_norm

        self.ffn = ffn

        if dropout_p > 0.0:
            self.ffn_dropout = Dropout(dropout_p)
        else:
            self.register_module("ffn_dropout", None)

        if scale_residual:
            self.residual_scale = Parameter(
                torch.empty((model_dim,), device=device, dtype=dtype)
            )
        else:
            self.register_parameter("residual_scale", None)

        if norm_order == TransformerNormOrder.POST:
            self.ffn_layer_norm = ffn_layer_norm

        self.norm_order = norm_order

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        if self.residual_scale is not None:
            nn.init.ones_(self.residual_scale)

    @finaloverride
    def forward(
            self,
            seqs: Tensor,
            padding_mask=None,
            self_attn_mask=None,
    ):
        seqs = self._forward_self_attn(seqs, padding_mask, self_attn_mask)

        seqs = self._forward_ffn(seqs)

        return seqs, padding_mask

    def _forward_self_attn(
            self,
            seqs: Tensor,
            padding_mask=None,
            self_attn_mask=None,
    ) -> Tensor:
        residual = seqs

        if self.norm_order != TransformerNormOrder.POST:
            seqs = self.self_attn_layer_norm(seqs)

        seqs = self.self_attn(
            seqs,
            padding_mask,
            keys=seqs,
            key_padding_mask=padding_mask,
            values=seqs,
            attn_mask=self_attn_mask,
        )

        if self.self_attn_norm is not None:
            seqs = self.self_attn_norm(seqs)

        if self.self_attn_dropout is not None:
            seqs = self.self_attn_dropout(seqs)

        seqs = seqs + residual

        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.self_attn_layer_norm(seqs)

        return seqs

    def _forward_ffn(self, seqs: Tensor) -> Tensor:
        residual = seqs

        if self.norm_order != TransformerNormOrder.POST:
            seqs = self.ffn_layer_norm(seqs)

        seqs = self.ffn(seqs)

        if self.ffn_dropout is not None:
            seqs = self.ffn_dropout(seqs)

        if self.residual_scale is not None:
            residual = self.residual_scale * residual

        seqs = seqs + residual

        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.ffn_layer_norm(seqs)

        return seqs

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return f"{s}, norm_order={self.norm_order}"


class TransformerEncoder(Module, ABC):
    """Represents a Transformer encoder."""

    model_dim: int
    layers: ModuleList

    # _layer_output_hooks: Dict[int, EncoderLayerOutputHook]

    def __init__(self, model_dim: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        """
        super().__init__()

        self.model_dim = model_dim

        # self._layer_output_hooks = OrderedDict()

    @abstractmethod
    def forward(
            self, seqs: Tensor, padding_mask=None, self_attn_mask=None
    ):
        """
        :param seqs:
            The sequences to encode. *Shape:* :math:`(N,S,M)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`M`
            is the dimensionality of the model.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.

        :param self_attn_mask:
            Specific mask for self-attention, for example causal or pattern-ed attention

        :returns:
            - The encoder output. *Shape:* Same as ``seqs``.
            - The padding mask of the encoder output. *Shape:* Same as
              ``padding_mask``.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


@final
class StandardTransformerEncoder(TransformerEncoder):
    """Represents a Transformer encoder as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    # self_attn_mask_factory: Optional[AttentionMaskFactory]
    layer_norm: Optional[LayerNorm]
    norm_order: TransformerNormOrder

    def __init__(
            self,
            layers: Iterable[TransformerEncoderLayer],
            *,
            frontend = None,
            # self_attn_mask_factory: Optional[AttentionMaskFactory] = None,
            layer_drop_p: float = 0.0,
            norm_order: TransformerNormOrder = TransformerNormOrder.POST,
            layer_norm_factory: Optional[LayerNormFactory] = None,
            device: Optional[Device] = None,
            dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param layers:
            The encoder layers.
        :param frontend:
            The FrontEnd (for example positional embedding layers)
        :param self_attn_mask_factory:
            The self attention mask factory.
        :param layer_drop_p:
            If greater than zero, applies LayerDrop to the encoder layers as
            described in :cite:t:`https://doi.org/10.48550/arxiv.1909.11556`.
        :param norm_order:
            The Layer Normalization order.
        :param layer_norm_factory:
            The factory to construct the Layer Normalization module.
        """
        layer_list = ModuleList(layers, drop_p=layer_drop_p)
        if not layer_list:
            raise ValueError("`layers` must be non-empty.")

        model_dim = layer_list[0].model_dim

        super().__init__(model_dim)

        if layer_norm_factory is None:
            layer_norm_factory = create_standard_layer_norm

        self.layers = layer_list

        if norm_order != TransformerNormOrder.POST:
            self.layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)
        else:
            self.register_module("layer_norm", None)

        self.norm_order = norm_order

        self.frontend = frontend

    @finaloverride
    def forward(
            self, seqs: Tensor, padding_mask=None, self_attn_mask=None,
    ):
        # if self._layer_output_hooks and self.layers.drop_p > 0.0:
        #     raise RuntimeError(
        #         "The layer output hooks cannot be run when LayerDrop is enabled."
        #     )
        if self.frontend is not None:
            seqs, padding_mask = self.frontend(seqs, padding_mask)

        num_layers = len(self.layers)

        # if self.self_attn_mask_factory is None:
        #     self_attn_mask = None
        # else:
        #     self_attn_mask = self.self_attn_mask_factory(
        #         seqs, keys=seqs, training=self.training
        #     )
        self_attn_mask = self_attn_mask

        for layer_idx, layer in enumerate(self.layers.drop_iter()):
            seqs, padding_mask = layer(seqs, padding_mask, self_attn_mask)

            # for hook in self._layer_output_hooks.values():
            #     if not hook(layer_idx, seqs, padding_mask, num_layers):
            #         break

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        return seqs, padding_mask

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return f"{s}, norm_order={self.norm_order}"
