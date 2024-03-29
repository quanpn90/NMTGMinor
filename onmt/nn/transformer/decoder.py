# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Iterable, Optional, Protocol, Tuple, final, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Dropout, Module
from torch.nn.parameter import Parameter

# from fairseq2.nn.normalization import LayerNorm
from onmt.nn.normalization import LayerNorm, LayerNormFactory, StandardLayerNorm
# from fairseq2.nn.padding import PaddingMask
from onmt.nn.transformer.attention_mask import (
    AttentionMask,
    AttentionMaskFactory,
    CausalAttentionMask,
    CausalAttentionMaskFactory)

# from fairseq2.nn.transformer.attention_mask import AttentionMask
from onmt.nn.transformer.ffn import FeedForwardNetwork
from onmt.nn.normalization import LayerNormFactory, create_standard_layer_norm
from onmt.nn.transformer.multihead_attention import MultiheadAttention
from onmt.nn.transformer.norm_order import TransformerNormOrder
from onmt.nn.module_list import ModuleList

from onmt.typing import DataType, Device, finaloverride, override


class TransformerDecoderLayer(Module, ABC):
    """Represents a Transformer decoder layer."""

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
            padding_mask=None,
            self_attn_mask: Optional[AttentionMask] = None,
            encoder_output: Optional[Tensor] = None,
            encoder_padding_mask=None,
            # *,
            # state_bag: Optional[IncrementalStateBag] = None,
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
        :param encoder_output:
            The encoder output to use in encoder-decoder attention. *Shape:*
            :math:`(N,S_{enc},M_{enc})`, where :math:`N` is the batch size,
            :math:`S_{enc}` is the encoder output sequence length, and
            :math:`M_{enc}` is the dimensionality of the encoder.
        :param encoder_padding_mask:
            The padding mask of ``encoder_output``. *Shape:* :math:`(N,S_{enc})`,
            where :math:`N` is the batch size and :math:`S_{enc}` is the encoder
            output sequence length.
        :param state_bag:
            The state bag to use for incremental decoding.

        :returns:
            - The decoder layer output. *Shape:* Same as ``seqs``.
            - The padding mask of the decoder layer output. *Shape:* Same as
              ``padding_mask``.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


@final
class StandardTransformerDecoderLayer(TransformerDecoderLayer):
    """Represents a Transformer decoder layer as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    self_attn: MultiheadAttention
    self_attn_norm: Optional[LayerNorm]
    self_attn_dropout: Optional[Dropout]
    self_attn_layer_norm: LayerNorm
    encoder_decoder_attn: Optional[MultiheadAttention]
    encoder_decoder_attn_dropout: Optional[Dropout]
    encoder_decoder_attn_layer_norm: Optional[LayerNorm]
    ffn: FeedForwardNetwork
    ffn_dropout: Optional[Dropout]
    residual_scale: Optional[Parameter]
    ffn_layer_norm: LayerNorm
    norm_order: TransformerNormOrder

    def __init__(
            self,
            self_attn: MultiheadAttention,
            encoder_decoder_attn: Optional[MultiheadAttention],
            ffn: FeedForwardNetwork,
            *,
            frontend=None,
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
        :param encoder_decoder_attn:
            The encoder-decoder attention layer.
        :param ffn:
            The feed-forward network.
        :param scale_residual:
            If ``True``, scales residuals before adding them to the output of
            the feed-forward network as described in
            :cite:t:`https://doi.org/10.48550/arxiv.2110.09456`.
        :param dropout_p:
            The dropout probability on outputs of the attention layers and the
            feed-forward network.
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

        if encoder_decoder_attn is None:
            self.register_module("encoder_decoder_attn", None)
            self.register_module("encoder_decoder_attn_layer_norm", None)
        else:
            encoder_decoder_attn_layer_norm = layer_norm_factory(
                model_dim, device=device, dtype=dtype
            )

            if norm_order != TransformerNormOrder.POST:
                self.encoder_decoder_attn_layer_norm = encoder_decoder_attn_layer_norm

            self.encoder_decoder_attn = encoder_decoder_attn

            if dropout_p > 0.0:
                self.encoder_decoder_attn_dropout = Dropout(dropout_p)
            else:
                self.register_module("encoder_decoder_attn_dropout", None)

            if norm_order == TransformerNormOrder.POST:
                self.encoder_decoder_attn_layer_norm = encoder_decoder_attn_layer_norm

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

        # TODO: Quan: the frontend is either here or the upper layer
        if frontend is not None:
            self.frontend = frontend

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        if self.residual_scale is not None:
            nn.init.ones_(self.residual_scale)

    @finaloverride
    def forward(
            self,
            seqs: Tensor,
            padding_mask: Optional[Tensor],
            self_attn_mask: Optional[AttentionMask] = None,
            encoder_output: Optional[Tensor] = None,
            encoder_padding_mask: Optional[Tensor] = None,
            *args, **kwargs
            # *,
            # state_bag: Optional[IncrementalStateBag] = None,
    ): # -> Tuple[Tensor, Optional[PaddingMask]]:
        seqs = self._forward_self_attn(seqs, padding_mask, self_attn_mask)

        seqs = self._forward_encoder_decoder_attn(
            seqs, padding_mask, encoder_output, encoder_padding_mask)

        seqs = self._forward_ffn(seqs)

        return seqs, padding_mask

    def _forward_self_attn(
            self,
            seqs: Tensor,
            padding_mask,
            self_attn_mask=None
            # state_bag: Optional[IncrementalStateBag],
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
            attn_mask=self_attn_mask
        )

        if self.self_attn_norm is not None:
            seqs = self.self_attn_norm(seqs)

        if self.self_attn_dropout is not None:
            seqs = self.self_attn_dropout(seqs)

        seqs = seqs + residual

        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.self_attn_layer_norm(seqs)

        return seqs

    def _forward_encoder_decoder_attn(
            self,
            seqs: Tensor,
            padding_mask: Optional[Tensor],
            encoder_output: Optional[Tensor],
            encoder_padding_mask: Optional[Tensor],
            # state_bag: Optional[IncrementalStateBag],
    ) -> Tensor:
        if self.encoder_decoder_attn is None:
            if encoder_output is not None:
                raise ValueError(
                    "`encoder_output` must be `None` for decoder-only attention."
                )

            return seqs

        if encoder_output is None:
            raise ValueError(
                "`encoder_output` must not be `None` for encoder-decoder attention."
            )

        residual = seqs

        if self.norm_order != TransformerNormOrder.POST:
            seqs = cast(LayerNorm, self.encoder_decoder_attn_layer_norm)(seqs)

        seqs = self.encoder_decoder_attn(
            seqs,
            padding_mask,
            keys=encoder_output,
            key_padding_mask=encoder_padding_mask,
            values=encoder_output,
        )

        if self.encoder_decoder_attn_dropout is not None:
            seqs = self.encoder_decoder_attn_dropout(seqs)

        seqs = seqs + residual

        if self.norm_order == TransformerNormOrder.POST:
            seqs = cast(LayerNorm, self.encoder_decoder_attn_layer_norm)(seqs)

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


class TransformerDecoder(Module, ABC):
    """Represents a Transformer decoder."""

    model_dim: int
    layers: ModuleList

    _layer_output_hooks: Dict[int, DecoderLayerOutputHook]

    def __init__(self, model_dim: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        """
        super().__init__()

        self.model_dim = model_dim

        self._layer_output_hooks = OrderedDict()

    @abstractmethod
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        encoder_output: Optional[Tensor] = None,
        encoder_padding_mask: Optional[PaddingMask] = None,
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """
        :param seqs:
            The sequences to decode. *Shape:* :math:`(N,S,M)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`M`
            is the dimensionality of the model.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.
        :param encoder_output:
            The encoder output to use in encoder-decoder attention. *Shape:*
            :math:`(N,S_{enc},M_{enc})`, where :math:`N` is the batch size,
            :math:`S_{enc}` is the encoder output sequence length, and
            :math:`M_{enc}` is the dimensionality of the encoder.
        :param encoder_padding_mask:
            The padding mask of ``encoder_output``. *Shape:* :math:`(N,S_{enc})`,
            where :math:`N` is the batch size and :math:`S_{enc}` is the encoder
            output sequence length.
        :param state_bag:
            The state bag to use for incremental decoding.

        :returns:
            - The decoder output. *Shape:* Same as ``seqs``.
            - The padding mask of the decoder output. *Shape:* Same as
              ``padding_mask``.
        """

    # def register_layer_output_hook(
    #     self, hook: DecoderLayerOutputHook
    # ) -> RemovableHandle:
    #     """Register a layer output hook on the module.
    #
    #     The hook will be called every time after a layer in the decoder stack
    #     has computed an output.
    #
    #     :param hook:
    #         The hook to register.
    #
    #     :returns:
    #         A handle that can be used to remove the added hook by calling
    #         ``handle.remove()``.
    #     """
    #     handle = RemovableHandle(self._layer_output_hooks)
    #
    #     self._layer_output_hooks[handle.id] = hook
    #
    #     return handle

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


@final
class StandardTransformerDecoder(TransformerDecoder):
    """Represents a Transformer decoder as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    self_attn_mask_factory: Optional[AttentionMaskFactory]
    layer_norm: Optional[LayerNorm]
    norm_order: TransformerNormOrder

    def __init__(
            self,
            layers: Iterable[TransformerDecoderLayer],
            *,
            self_attn_mask_factory: Optional[AttentionMaskFactory] = None,
            use_causal_attn_mask: bool = True,
            layer_drop_p: float = 0.0,
            norm_order: TransformerNormOrder = TransformerNormOrder.POST,
            layer_norm_factory: Optional[LayerNormFactory] = None,
            device: Optional[Device] = None,
            dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param layers:
            The decoder layers.
        :param self_attn_mask_factory:
            The self attention mask factory.
        :param use_causal_attn_mask:
            If ``True``, passes a full :class:`CausalAttentionMask` to the
            decoder layers; otherwise, passes ``None``. Ignored if
            ``self_attn_mask_factory`` is specified.
        :param layer_drop_p:
            If greater than zero, applies LayerDrop to the decoder layers as
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

        if self_attn_mask_factory is not None:
            self.self_attn_mask_factory = self_attn_mask_factory
        elif use_causal_attn_mask:
            # TODO: write custom causal attention mask factory
            self.self_attn_mask_factory = CausalAttentionMaskFactory()
        else:
            self.self_attn_mask_factory = None

        self.layers = layer_list

        if norm_order != TransformerNormOrder.POST:
            self.layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)
        else:
            self.register_module("layer_norm", None)

        self.norm_order = norm_order

    @override
    def forward(
            self,
            seqs: Tensor,
            padding_mask: Optional[Tensor],
            encoder_output: Optional[Tensor] = None,
            encoder_padding_mask: Optional[Tensor] = None,
            *args, **kwargs
    ): # -> Tuple[Tensor, Optional[PaddingMask]]:
        # if self._layer_output_hooks and self.layers.drop_p > 0.0:
        #     raise RuntimeError(
        #         "The layer output hooks cannot be run when LayerDrop is enabled."
        #     )

        num_layers = len(self.layers)

        # generate the self attn mask (causal/alibi etc)
        if self.self_attn_mask_factory is None:
            self_attn_mask = None
        else:
            self_attn_mask = self.self_attn_mask_factory(
                seqs, keys=seqs, training=self.training
            )

        for layer_idx, layer in enumerate(self.layers.drop_iter()):
            seqs, padding_mask = layer(
                seqs,
                padding_mask,
                self_attn_mask,
                encoder_output,
                encoder_padding_mask,
                # state_bag=state_bag,
            )

            # for hook in self._layer_output_hooks.values():
            #     if not hook(layer_idx, seqs, padding_mask, num_layers):
            #         break

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        return seqs, padding_mask

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        if self.self_attn_mask_factory is not None:
            self_attn_mask_factory = getattr(
                self.self_attn_mask_factory, "__name__", self.self_attn_mask_factory
            )

            s = f"{s}, self_attn_mask_factory={self_attn_mask_factory}"

        return f"{s}, norm_order={self.norm_order}"