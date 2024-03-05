# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, MutableSequence, Optional, Protocol, Tuple, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle

from .w2vbert_attention import TorchSDPA, SDPA, create_default_sdpa
from onmt.nn.position_encoder import PositionEncoder
from onmt.nn.projection import Linear

from onmt.modules.optimized.self_attention_func import self_attn_func, self_attn_compact_func
from onmt.modules.optimized.relative_self_attention_func import relative_self_attn_func

import overrides

finaloverride = overrides.final



class MultiheadAttention(Module, ABC):
    """Represents a Transformer multi-head attention layer."""

    num_heads: int
    model_dim: int

    # _attn_weight_hooks: Dict[int, AttentionWeightHook]

    def __init__(self, model_dim: int, num_heads: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param num_heads:
            The number of attention heads.
        """
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads

        # self._attn_weight_hooks = OrderedDict()

    @abstractmethod
    def forward(
            self,
            seqs: Tensor,
            padding_mask,
            keys: Tensor,
            key_padding_mask,
            values: Tensor,
            *,
            attn_mask=None
    ) -> Tensor:
        """
        :param seqs:
            The sequences to query. *Shape:* :math:`(N,S,M)`, where :math:`N` is
            the batch size, :math:`S` is the sequence length, and :math:`M` is
            the dimensionality of the model.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.
        :param keys:
            The keys. *Shape:* :math:`(N,S_{kv},K)`, where :math:`N` is the
            batch size, :math:`S_{kv}` is the key/value sequence length, and
            :math:`K` is the key size.
        :param key_padding_mask:
            The padding mask indicating which key positions to ignore for the
            purpose of attention. *Shape:* :math:`(N,S_{kv})`, where :math:`N`
            is the batch size and :math:`S_{kv}` is the key/value sequence
            length.
        :param values:
            The values. *Shape:* :math:`(N,S_{kv},V)`, where :math:`N` is the
            batch size, :math:`S_{kv}` is the key/value sequence length, and
            :math:`V` is the value size.
        :param attn_mask:
            The mask that will be added to attention weights before computing
            the attention. *Shape:* :math:`([H],S,S_{kv})`, where :math:`H` is
            the number of attention heads, :math:`S` is the sequence length, and
            :math:`S_{kv}` is the key/value sequence length.
        :param state_bag:
            The state bag to use for incremental decoding.

        :returns:
            The attention values for ``seqs``. *Shape:* :math:`(N,S,M)`, where
            :math:`N` is the batch size, :math:`S` is the sequence length, and
            :math:`M` is the dimensionality of the model.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"num_heads={self.num_heads}, model_dim={self.model_dim}"

    @abstractmethod
    def convert_fast_attention(
            self,
    ) -> None:
        """

        Args:

        Returns:

        """

class AttentionWeightHook(Protocol):
    """Represents a hook to pass to
    :meth:`~MultiheadAttention.register_attn_weight_hook`."""

    def __call__(
            self, m: MultiheadAttention, attn: Tensor, attn_weights: Tensor
    ) -> None:
        """
        :param m:
            The module that has computed the attention weights.
        :param attn:
            The computed attention values. *Shape:* :math:`(N,S,V)`, where
            :math:`N` is the batch size, :math:`S` is the sequence length, and
            :math:`V` is the value size.
        :param attn_weights:
            The computed attention weights. *Shape:* :math:`(N,S,S_{kv})`, where
            :math:`N` is the batch size, :math:`S` is the sequence length, and
            :math:`S_{kv}` is the key/value sequence length.
        """


class AttentionWeightStoreHook:
    """Stores attention weights in a provided storage.

    .. note::
        This class follows the :class:`AttentionWeightHook` protocol.
    """

    _storage: MutableSequence[Tuple[Tensor, Tensor]]

    def __init__(self, storage: MutableSequence[Tuple[Tensor, Tensor]]) -> None:
        """
        :param storage:
            The storage in which to store attention weights.
        """
        self._storage = storage

    def __call__(
            self, m: MultiheadAttention, attn: Tensor, attn_weights: Tensor
    ) -> None:
        self._storage.append((attn, attn_weights))


@final
class StandardMultiheadAttention(MultiheadAttention):
    """Represents a Transformer multi-head attention as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    num_key_value_heads: int
    q_proj: Projection
    k_proj: Projection
    v_proj: Projection
    # attn_mask_factory: Optional[AttentionMaskFactory]
    pos_encoder: Optional[PositionEncoder]
    bias_k: Optional[Parameter]
    bias_v: Optional[Parameter]
    add_zero_attn: bool
    sdpa: SDPA
    head_scale_weight: Optional[Parameter]
    output_proj: Projection

    # state_factory: Optional[AttentionStateFactory]

    def __init__(
            self,
            model_dim: int,
            num_heads: int,
            *,
            num_key_value_heads: Optional[int] = None,
            q_proj=None,
            k_proj=None,
            v_proj=None,
            # attn_mask_factory: Optional[AttentionMaskFactory] = None,
            pos_encoder: Optional[PositionEncoder] = None,
            sdpa: Optional[SDPA] = None,
            scale_heads: bool = False,
            output_proj=None,
            bias: bool = True,
            # state_factory: Optional[AttentionStateFactory] = None,
            device=None,
            dtype=None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param num_heads:
            The number of attention heads.
        :param num_key_value_heads:
            The number of key/value heads for Grouped Query Attention as
            described in :cite:t:`https://doi.org/10.48550/arXiv.2305.13245`.
            If ``None`` or set to ``num_heads``, it is equivalent to standard
            Multi Head Attention (MHA); if set to 1, it is equivalent to Multi
            Query Attention (MQA).
        :param q_proj:
            The projection to apply to sequences before computing attention. If
            ``None``, a default projection will be used.
        :param k_proj:
            The projection to apply to keys before computing attention. If
            ``None``, a default projection will be used.
        :param v_proj:
            The projection to apply to values before computing attention. If
            ``None``, a default projection will be used.
        :param attn_mask_factory:
            The attention mask factory.
        :param pos_encoder:
            The position encoder to apply to sequences and keys after projection.
        :param sdpa:
            The :class:`SDPA` module to compute head attentions. If ``None``, a
            default implementation will be used.
        :param scale_heads:
            If ``True``, applies head scaling as described in
            :cite:t:`https://doi.org/10.48550/arxiv.2110.09456`
        :param output_proj:
            The projection to produce final attentions. If ``None``, a default
            projection will be used.
        :param bias:
            If ``True``, query, key, value, and output projections learn an
            additive bias. Ignored for explicitly specified projections.
        :param state_factory:
            The factory to construct :class:`AttentionState` instances for
            incremental decoding.
        """
        super().__init__(model_dim, num_heads)

        if num_key_value_heads is None:
            self.num_key_value_heads = num_heads
        else:
            if num_heads < num_key_value_heads:
                raise ValueError(
                    f"`num_heads` must be greater than or equal to `num_key_value_heads` ({num_key_value_heads}), but is {num_heads} instead."
                )

            if num_heads % num_key_value_heads != 0:
                raise ValueError(
                    f"`num_heads` must be a multiple of `num_key_value_heads` ({num_key_value_heads}), but is {num_heads} instead."
                )

            self.num_key_value_heads = num_key_value_heads

        head_dim = model_dim // num_heads

        num_query_groups = num_heads // self.num_key_value_heads

        if q_proj is None and k_proj is None and v_proj is None:
            q_proj = Linear(
                model_dim,
                model_dim,
                bias,
                init_fn=init_qkv_projection,
                device=device,
                dtype=dtype,
            )
            k_proj = Linear(
                model_dim,
                head_dim * self.num_key_value_heads,
                bias,
                init_fn=init_qkv_projection,
                device=device,
                dtype=dtype,
            )
            v_proj = Linear(
                model_dim,
                head_dim * self.num_key_value_heads,
                bias,
                init_fn=init_qkv_projection,
                device=device,
                dtype=dtype,
            )
        else:
            if q_proj is None or k_proj is None or v_proj is None:
                raise ValueError(
                    "`q_proj`, `k_proj`, and `v_proj` must be all specified."
                )

            if q_proj.input_dim != model_dim:
                raise ValueError(
                    f"`input_dim` of `q_proj` must be equal to `model_dim` ({model_dim}), but is {q_proj.input_dim} instead."
                )

            if (k_dim := k_proj.output_dim * num_query_groups) != q_proj.output_dim:
                raise ValueError(
                    f"`output_dim` of `q_proj` and `output_dim` of `k_proj` (times the number of query groups when GQA) must be equal, but are {q_proj.output_dim} and {k_dim} instead."
                )

            if k_proj.output_dim % self.num_key_value_heads != 0:
                raise ValueError(
                    f"`output_dim` of `k_proj` must be a multiple of `num_key_value_heads` ({self.num_key_value_heads}), but is {k_proj.output_dim} instead."
                )

            if v_proj.output_dim % self.num_key_value_heads != 0:
                raise ValueError(
                    f"`output_dim` of `v_proj` must be a multiple of `num_key_value_heads` ({self.num_key_value_heads}), but is {v_proj.output_dim} instead."
                )

        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj

        # self.attn_mask_factory = attn_mask_factory

        if pos_encoder is not None:
            if head_dim != pos_encoder.encoding_dim:
                raise ValueError(
                    f"`encoding_dim` of `pos_encoder` must be equal to the size of the header dimension ({head_dim}), but is {pos_encoder.encoding_dim} instead."
                )

            self.pos_encoder = pos_encoder
        else:
            self.register_module("pos_encoder", None)

        if sdpa is not None:
            self.sdpa = sdpa
        else:
            self.sdpa = create_default_sdpa()

        if scale_heads:
            self.head_scale_weight = Parameter(
                torch.empty(num_heads, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("head_scale_weight", None)

        v_dim = v_proj.output_dim * num_query_groups

        if output_proj is None:
            self.output_proj = Linear(
                v_dim,
                model_dim,
                bias,
                init_fn=init_output_projection,
                device=device,
                dtype=dtype,
            )
        else:
            if v_dim != output_proj.input_dim:
                raise ValueError(
                    f"`output_dim` of `v_proj` (times the number of query groups when GQA) and `input_dim` of `output_proj` must be equal, but are {v_dim} and {output_proj.input_dim} instead."
                )

            if output_proj.output_dim != model_dim:
                raise ValueError(
                    f"`output_dim` of `output_proj` must be equal to `model_dim` ({model_dim}), but is {output_proj.output_dim} instead."
                )

            self.output_proj = output_proj

        # self.state_factory = state_factory

        self.reset_parameters()
        self.fast = False

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        if self.head_scale_weight is not None:
            nn.init.ones_(self.head_scale_weight)

    def convert_fast_attention(
            self,
    ) -> None:

        if self.sdpa.name in ['nothing_works']:
            if self.fast:
                return

            print('[INFO] Converting Shaw Attention to Fast ..')

            self.fast = True
            w_q = self.q_proj.weight.clone()
            w_k = self.k_proj.weight.clone()
            w_v = self.v_proj.weight.clone()
            weights = [w_q, w_k, w_v]
            weight_ = torch.cat(weights, dim=0).contiguous()

            b_q = self.q_proj.bias.clone()
            b_k = self.k_proj.bias.clone()
            b_v = self.v_proj.bias.clone()
            biases = [b_q, b_k, b_v]
            bias_ = torch.cat(biases, dim=0).contiguous()

            head_dim = self.model_dim // self.num_heads
            heads = self.num_heads
            input_dim = self.model_dim

            # when we concatenate the weights, the output has the size 3 * D (3 -> heads -> head_dim)
            # the fast attention module requires (heads -> 3 -> head_dim)
            weight_ = weight_.reshape(3 * head_dim * heads, input_dim).view(3, heads, head_dim, input_dim).transpose(0,
                                                                                                                     1). \
                reshape(-1, input_dim)

            bias_ = bias_.reshape(3 * head_dim * heads).view(3, heads, head_dim).transpose(0, 1).reshape(-1)

            weight_t = torch.Tensor(3 * input_dim, input_dim).to(weight_.device)
            bias_t = torch.Tensor(3 * input_dim).to(weight_.device)
            weight_t.copy_(weight_)
            bias_t.copy_(bias_)
            self.proj_weight = Parameter(weight_t)
            self.proj_bias = Parameter(bias_t)

            self.proj_weight.requires_grad = self.q_proj.weight.requires_grad
            self.proj_bias.requires_grad = self.q_proj.bias.requires_grad
            del self.q_proj, self.k_proj, self.v_proj
        else:
            self.fast = False
            pass

    @finaloverride
    def forward(
            self,
            seqs: Tensor,  # why don't you call this "query"
            padding_mask,
            keys: Tensor,
            key_padding_mask,
            values: Tensor,
            *,
            attn_mask=None
    ) -> Tensor:

        if self.fast:
            in_proj_weight = self.proj_weight
            out_proj_weight = self.output_proj.weight

            rel_indices = self.sdpa.get_relative_indices_three_dim(keys)

            # (S_kv, S_kv, K_h)
            positions = self.sdpa.rel_k_embed(rel_indices)

            inputs = seqs.transpose(0, 1).contiguous()
            is_training = self.training
            recompute = False

            outputs, coverage = relative_self_attn_func(inputs, positions, False,
                                                        is_training, self.num_heads,
                                                        in_proj_weight, out_proj_weight, None,
                                                        self.proj_bias, self.output_proj.bias, None,
                                                        None, None,
                                                        key_padding_mask, 0.0,
                                                        False, None, False,
                                                        # incremental and state and double precision
                                                        True, True, recompute)  # learnable_pos + return-coverage

            outputs = outputs.transpose(0, 1).contiguous()

            return outputs

        else:

            state_bag = None  # Note: encoder doesn't need state bag

            # (N, S, M) -> (N, H, S, K_h)
            q = self._project_q(seqs, padding_mask, state_bag)
            # k: (N, S_kv, M) -> (N, H_kv, S_kv, K_h)
            # v: (N, S_kv, M) -> (N, H_kv, S_kv, V_h)
            k, v = self._project_kv(keys, key_padding_mask, values)

            # With Grouped Query Attention, each key/value head is repeated.
            if (num_query_groups := self.num_heads // self.num_key_value_heads) > 1:
                # (N, H_kv, S_kv, K_h) -> (N, H, S_kv, K_h)
                k = repeat_interleave(k, dim=1, repeat=num_query_groups)
                # (N, H_kv, S_kv, K_h) -> (N, H, S_kv, V_h)
                v = repeat_interleave(v, dim=1, repeat=num_query_groups)

            # if self.attn_mask_factory is not None:
            #     attn_mask = self.attn_mask_factory(
            #         seqs, keys=keys, training=self.training, state_bag=state_bag
            #     )
            # TODO: check attn_mask_factory later

            attn_mask = None
            needs_weights = False  # len(self._attn_weight_hooks) > 0

            # attn:         (N, H, S, V_h)
            # attn_weights: (N, H, S, S_kv)
            attn, attn_weights = self.sdpa(
                q,
                k,
                key_padding_mask,
                v,
                attn_mask=attn_mask,
                needs_weights=needs_weights,
            )

            # if attn_weights is not None:
            #     for hook in self._attn_weight_hooks.values():
            #         hook(self, attn, attn_weights)

            # (N, H, S, V_h) -> (N, S, H, V_h)
            attn = attn.transpose(1, 2)

            if self.head_scale_weight is not None:
                attn = torch.einsum("nshv,h->nshv", attn, self.head_scale_weight)

            # (N, S, H, V_h) -> (N, S, V_proj)
            attn = attn.flatten(2, 3)

            # (N, S, V_proj) -> (N, S, M)
            attn = self.output_proj(attn)

            return attn  # type: ignore[no-any-return]

    def _project_q(
            self,
            seqs: Tensor,
            padding_mask,
            state_bag=None,
    ) -> Tensor:
        # (N, S, M) -> (N, S, K_proj)
        q = self.q_proj(seqs)

        # (N, S, K_proj) -> (N, H, S, K_h)
        q = q.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)

        if self.pos_encoder is not None:
            q = self.pos_encoder(q, padding_mask, state_bag=state_bag)

        return q  # type: ignore[no-any-return]

    def _project_kv(
            self,
            keys: Tensor,
            key_padding_mask,
            values: Tensor,
            state_bag=None,
    ) -> Tuple[Tensor, Tensor]:
        # (N, S, K) -> (N, S, K_proj)
        k = self.k_proj(keys)
        # (N, S, V) -> (N, S, V_proj)
        v = self.v_proj(values)

        # (N, S, K_proj) -> (N, H, S, K_h)
        k = k.unflatten(-1, (self.num_key_value_heads, -1)).transpose(1, 2)
        # (N, S, V_proj) -> (N, H, S, V_h)
        v = v.unflatten(-1, (self.num_key_value_heads, -1)).transpose(1, 2)

        # note: for Rotary Position Encoding
        if self.pos_encoder is not None:
            k = self.pos_encoder(k, key_padding_mask, state_bag=state_bag)

        return k, v

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        if self.num_key_value_heads != self.num_heads:
            s = f"{s}, num_key_value_heads={self.num_key_value_heads}"

        # if self.state_factory is not None:
        #     state_factory = getattr(self.state_factory, "__name__", self.state_factory)
        #
        #     s = f"{s}, state_factory={state_factory}"

        return s


def init_qkv_projection(proj: Linear) -> None:
    """Initialize ``proj`` as a multi-head attention input projection."""
    # Empirically observed the convergence to be much better with the scaled
    # initialization.
    nn.init.xavier_uniform_(proj.weight, gain=2 ** -0.5)

    if proj.bias is not None:
        nn.init.zeros_(proj.bias)


def init_output_projection(proj: Linear) -> None:
    """Initialize ``proj`` as a multi-head attention output projection."""
    nn.init.xavier_uniform_(proj.weight)

    if proj.bias is not None:
        nn.init.zeros_(proj.bias)


# class AttentionState(IncrementalState):
#     """Holds the projected keys and values of a :class:`MultiheadAttention`
#     module during incremental decoding."""
#
#     @abstractmethod
#     def append(self, k: Tensor, v: Tensor) -> None:
#         """Update the state with ``k`` and ``v``.
#
#         :param k:
#             The projected keys of the current step. *Shape:*
#             :math:`(N,H,1,K_{proj})`, where :math:`N` is the batch size,
#             :math:`H` is the number of heads, :math:`1` is the step length, and
#             :math:`K_{proj}` is the projected key size.
#         :param v:
#             The projected values of the current step. *Shape:*
#             :math:`(N,H,1,V_{proj})`, where :math:`N` is the batch size,
#             :math:`H` is the number of heads, :math:`1` is the step length, and
#             :math:`V_{proj}` is the projected value size.
#         """
#
#     @abstractmethod
#     def get(self) -> Tuple[Tensor, Tensor]:
#         """Return the state that should be used to compute the attention.
#
#         :returns:
#             - The projected keys.
#             - The projected values.
#         """


# class AttentionStateFactory(Protocol):
#     """Constructs instances of :class:`AttentionState`."""
#
#     def __call__(
#             self, k: Tensor, v: Tensor, max_seq_len: int, capacity_increment: Optional[int]
#     ) -> AttentionState:
#         """
#         :param k:
#             The initial projected keys.
#         :param v:
#             The initial projected values.
#         :param max_seq_len:
#             The expected maximum sequence length.
#         :param capacity_increment:
#             The sequence length capacity of state tensors will be incremented by
#             multiples of this value. If ``None``, state tensors will be
#             preallocated with a capacity of ``max_seq_len``.
#         """
#
#
# @final
# class FullAttentionState(AttentionState):
#     """Holds the past projected keys and values of a :class:`MultiheadAttention`
#     module during incremental decoding."""
#
#     seq_len: int
#     """The current sequence length of :attr:`k` and :attr:`v`."""
#
#     k: Tensor
#     """The projected keys accumulated from the past decoding steps. *Shape:*
#     :math:`(N,H,S_{rsv},K_{proj})`, where :math:`N` is the batch size, :math:`H`
#     is the number of heads, :math:`S_{rsv}` is the reserved sequence length
#     capacity, and :math:`K_{proj}` is the projected key size."""
#
#     v: Tensor
#     """The projected values accumulated from the past decoding steps. *Shape:*
#     :math:`(N,H,S_{rsv},V_{proj})`, where :math:`N` is the batch size, :math:`H`
#     is the number of heads, :math:`S_{rsv}` is the reserved sequence length
#     capacity, and :math:`V_{proj}` is the projected value size."""
#
#     capacity_increment: Optional[int]
#     """The sequence length capacity of :attr:`k` and :attr:`v` is incremented by
#     multiples of this value."""
#
#     def __init__(
#             self, k: Tensor, v: Tensor, max_seq_len: int, capacity_increment: Optional[int]
#     ) -> None:
#         if capacity_increment is not None and capacity_increment < 1:
#             raise ValueError(
#                 f"`capacity_increment` must be greater than or equal to 1, but is {capacity_increment} instead."
#             )
#
#         batch_size, num_heads, seq_len, head_dim = k.shape
#
#         init_capacity = 0 if capacity_increment else max_seq_len
#
#         self.k = k.new_empty((batch_size, num_heads, init_capacity, head_dim))
#         self.v = v.new_empty((batch_size, num_heads, init_capacity, head_dim))
#
#         self.seq_len = 0
#
#         self.capacity_increment = capacity_increment
#
#         self._expand_kv(seq_len)
#
#         self.k[:, :, :seq_len] = k
#         self.v[:, :, :seq_len] = v
#
#         self.seq_len = seq_len
#
#     @finaloverride
#     def append(self, k: Tensor, v: Tensor) -> None:
#         input_seq_len = k.size(2)
#
#         self._expand_kv(input_seq_len)
#
#         pos = self.seq_len
#
#         self.k[:, :, pos: pos + input_seq_len] = k
#         self.v[:, :, pos: pos + input_seq_len] = v
#
#         self.seq_len += input_seq_len
#
#     def _expand_kv(self, input_seq_len: int) -> None:
#         if self.capacity_increment is None:
#             return
#
#         batch_size, num_heads, capacity, head_dim = self.k.shape
#
#         new_seq_len = self.seq_len + input_seq_len
#
#         if new_seq_len <= capacity:
#             return
#
#         inc = self.capacity_increment
#
#         capacity = ((new_seq_len + inc - 1) // inc) * inc
#
#         k = self.k.new_empty((batch_size, num_heads, capacity, head_dim))
#         v = self.v.new_empty((batch_size, num_heads, capacity, head_dim))
#
#         if self.seq_len > 0:
#             k[:, :, : self.seq_len] = self.k[:, :, : self.seq_len]
#             v[:, :, : self.seq_len] = self.v[:, :, : self.seq_len]
#
#         self.k = k
#         self.v = v
#
#     @finaloverride
#     def get(self) -> Tuple[Tensor, Tensor]:
#         k = self.k[:, :, : self.seq_len]
#         v = self.v[:, :, : self.seq_len]
#
#         return k, v
#
#     @finaloverride
#     def reorder(self, new_order: Tensor) -> None:
#         self.k = self.k.index_select(0, new_order)
#         self.v = self.v.index_select(0, new_order)
#
#
# @final
# class LocalAttentionState(AttentionState):
#     """Holds the past :attr:`attn_window_len` projected keys and values of a
#     :class:`MultiheadAttention` module during incremental decoding.
#
#     The intended use of this class is with Sliding Window Attention as described
#     in :cite:t:`https://doi.org/10.48550/arxiv.2004.05150`.
#     """
#
#     seq_len: int
#     """The current sequence length of :attr:`k` and :attr:`v`."""
#
#     attn_window_len: int
#     """The attention window length."""
#
#     k: Tensor
#     """The projected keys accumulated from the past :attr:`attn_window_len`
#     decoding steps. *Shape:* :math:`(N,H,S_{rsv},K_{proj})`, where :math:`N` is
#     the batch size, :math:`H` is the number of heads, :math:`S_{rsv}` is the
#     reserved sequence length capacity, and :math:`K_{proj}` is the projected key
#     size."""
#
#     v: Tensor
#     """The projected values accumulated from the past :attr:`attn_window_len`
#     decoding steps. *Shape:* :math:`(N,H,S_{rsv},V_{proj})`, where :math:`N` is
#     the batch size, :math:`H` is the number of heads, :math:`S_{rsv}` is the
#     reserved sequence length capacity, and :math:`V_{proj}` is the projected
#     value size."""
#
#     capacity_increment: Optional[int]
#     """The sequence length capacity of :attr:`k` and :attr:`v` is incremented by
#     multiples of this value."""
#
#     def __init__(
#             self,
#             k: Tensor,
#             v: Tensor,
#             max_seq_len: int,
#             attn_window_len: int,
#             capacity_increment: Optional[int],
#     ) -> None:
#         if capacity_increment is not None and capacity_increment < 1:
#             raise ValueError(
#                 f"`capacity_increment` must be greater than or equal to 1, but is {capacity_increment} instead."
#             )
#
#         self.attn_window_len = min(max_seq_len, attn_window_len)
#
#         batch_size, num_heads, seq_len, head_dim = k.shape
#
#         init_capacity = 0 if capacity_increment else self.attn_window_len
#
#         self.k = k.new_empty((batch_size, num_heads, init_capacity, head_dim))
#         self.v = v.new_empty((batch_size, num_heads, init_capacity, head_dim))
#
#         self.seq_len = 0
#
#         self.capacity_increment = capacity_increment
#
#         self._expand_kv(seq_len)
#
#         copy_len = min(seq_len, self.attn_window_len)
#
#         self.k[:, :, :copy_len] = k[:, :, -copy_len:]
#         self.v[:, :, :copy_len] = v[:, :, -copy_len:]
#
#         self.seq_len = seq_len
#
#     @finaloverride
#     def append(self, k: Tensor, v: Tensor) -> None:
#         input_seq_len = k.size(2)
#
#         self._expand_kv(input_seq_len)
#
#         if self.seq_len + input_seq_len > self.attn_window_len:
#             copy_len = min(input_seq_len, self.attn_window_len)
#
#             if copy_len == self.attn_window_len:
#                 pos = 0
#             else:
#                 self.k = torch.roll(self.k, shifts=-copy_len, dims=2)
#                 self.v = torch.roll(self.v, shifts=-copy_len, dims=2)
#
#                 pos = self.attn_window_len - copy_len
#         else:
#             pos = self.seq_len
#
#             copy_len = input_seq_len
#
#         self.k[:, :, pos: pos + copy_len] = k[:, :, -copy_len:]
#         self.v[:, :, pos: pos + copy_len] = v[:, :, -copy_len:]
#
#         self.seq_len += input_seq_len
#
#     def _expand_kv(self, input_seq_len: int) -> None:
#         if self.capacity_increment is None:
#             return
#
#         batch_size, num_heads, capacity, head_dim = self.k.shape
#
#         new_seq_len = self.seq_len + input_seq_len
#
#         if new_seq_len <= capacity or capacity == self.attn_window_len:
#             return
#
#         inc = self.capacity_increment
#
#         capacity = min(((new_seq_len + inc - 1) // inc) * inc, self.attn_window_len)
#
#         k = self.k.new_empty((batch_size, num_heads, capacity, head_dim))
#         v = self.v.new_empty((batch_size, num_heads, capacity, head_dim))
#
#         if self.seq_len > 0:
#             k[:, :, : self.seq_len] = self.k[:, :, : self.seq_len]
#             v[:, :, : self.seq_len] = self.v[:, :, : self.seq_len]
#
#         self.k = k
#         self.v = v
#
#     @finaloverride
#     def get(self) -> Tuple[Tensor, Tensor]:
#         k = self.k[:, :, : self.seq_len]
#         v = self.v[:, :, : self.seq_len]
#
#         return k, v
#
#     @finaloverride
#     def reorder(self, new_order: Tensor) -> None:
#         self.k = self.k.index_select(0, new_order)
#         self.v = self.v.index_select(0, new_order)
#
#
# class LocalAttentionStateFactory:
#     """Constructs instances of :class:`LocalAttentionState`."""
#
#     def __init__(self, attn_window_len: int) -> None:
#         """
#         :param attn_window_len:
#             The attention window length.
#         """
#         self.attn_window_len = attn_window_len
#
#     def __call__(
#             self,
#             k: Tensor,
#             v: Tensor,
#             max_seq_len: int,
#             capacity_increment: Optional[int],
#     ) -> LocalAttentionState:
#         return LocalAttentionState(
#             k, v, max_seq_len, self.attn_window_len, capacity_increment
#         )
#
#     def __repr__(self) -> str:
#         return f"LocalAttentionStateFactory(attn_window_len={self.attn_window_len})"
#
#
# @final
# class StaticAttentionState(AttentionState):
#     """Holds the static projected keys and values (e.g. encoder-decoder) of a
#     :class:`MultiheadAttention` module during incremental decoding."""
#
#     k: Tensor
#     v: Tensor
#
#     def __init__(
#             self, k: Tensor, v: Tensor, max_seq_len: int, capacity_increment: Optional[int]
#     ) -> None:
#         self.k = k
#         self.v = v
#
#     @finaloverride
#     def append(self, k: Tensor, v: Tensor) -> None:
#         raise ValueError(" `StaticAttentionState` does not support `append()`.")
#
#     @finaloverride
#     def get(self) -> Tuple[Tensor, Tensor]:
#         return self.k, self.v
#
#     @finaloverride
#     def reorder(self, new_order: Tensor) -> None:
#         if new_order.size(0) != self.k.size(0):
#             self.k = self.k.index_select(0, new_order)
#             self.v = self.v.index_select(0, new_order)
