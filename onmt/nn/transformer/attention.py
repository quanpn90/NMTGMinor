# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generator, Optional, Protocol, Tuple, final
import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import dropout, softmax
import torch.nn as nn
from torch.nn.functional import pad

from onmt.nn.embedding import StandardEmbedding


from onmt.nn.projection import Linear

import overrides
finaloverrides = overrides.final


class SDPA(Module, ABC):
    """Computes scaled dot-product attention."""

    def __init__(self):
        super().__init__()
        self.name = "generic_sdpa"

    @abstractmethod
    def forward(
            self,
            seqs: Tensor,
            keys: Tensor,
            key_padding_mask,
            values: Tensor,
            *,
            attn_mask=None,
            needs_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param seqs:
            The sequences to query. *Shape:* :math:`(N,H,S,K)`, where :math:`N`
            is the batch size, :math:`H` is the number of heads, :math:`S` is
            the sequence length, and :math:`K` is the key size.
        :param keys:
            The keys. *Shape:* :math:`(N,H,S_{kv},K)`, where :math:`N` is the
            batch size, :math:`H` is the number of heads, :math:`S_{kv}` is the
            key/value sequence length, and :math:`K` is the key size.
        :param key_padding_mask:
            The padding mask indicating which key positions to ignore for the
            purpose of attention. *Shape:* :math:`(N,S_{kv})`, where :math:`N`
            is the batch size and :math:`S_{kv}` is the key/value sequence
            length.
        :param values:
            The values. *Shape:* :math:`(N,H,S_{kv},V)`, where :math:`N` is the
            batch size, :math:`H` is the number of heads, :math:`S_{kv}` is the
            key/value sequence length, and :math:`V` is the value size.
        :param attn_mask:
            The mask that will be added to attention weights before computing
            the attention. *Shape:* :math:`([H],S,S_{kv})`, where :math:`H` is
            the number of heads, :math:`S` is the sequence length, and
            :math:`S_{kv}` is the key/value sequence length.
        :param needs_weights:
            If ``True``, returns the attention weights.

        :returns:
            - The attention values. *Shape:* :math:`(N,H,S,V)`, where :math:`N`
              is the batch size, :math:`H` is the number of heads, :math:`S` is
              the sequence length, and :math:`V` is the value size.
            - The attention weights. *Shape:* :math:`(N,H,S,S_{kv})`, where
              :math:`N` is the batch size, :math:`H` is the number of heads,
              :math:`S` is the sequence length, and :math:`S_{kv}` is the
              key/value sequence length.
        """


@final
class TorchSDPA(SDPA):
    """Computes scaled dot-product attention using PyTorch SDPA v2."""

    attn_dropout_p: float

    def __init__(self, *, attn_dropout_p: float = 0.0) -> None:
        """
        :param attn_dropout_p:
            The dropout probability on attention weights.
        """
        super().__init__()

        self._has_warned = False
        self.name = "torch_sdpa"

        self.attn_dropout_p = attn_dropout_p

    @finaloverrides
    def forward(
            self,
            seqs: Tensor,
            keys: Tensor,
            key_padding_mask,
            values: Tensor,
            *,
            attn_mask=None,
            needs_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        # note: key_padding_mask is a binary mask (the masked positions are 1)
        # attn_mask is a float mask (additive mask  )

        # note2: torch and fairseq implementation uses binary masks in which masked positions are 0
        # while my implementation uses 1 for masked positions

        if needs_weights:
            if not self._has_warned:
                logger.warning(
                    "`TorchSDPA` has to fall back to the naive SDPA implementation "
                    "because of `needs_weights` set to `True`.")  # fmt: skip

                self._has_warned = True

            return _naive_scaled_dot_product_attention(
                seqs,
                keys,
                key_padding_mask,
                values,
                attn_mask,
                self.attn_dropout_p,
                needs_weights,
                self.training,
            )

        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.attn_dropout_p

        is_causal = False

        if key_padding_mask is not None:
            mask = key_padding_mask

            # (N, S_kv) -> (N, 1, 1, S_kv)
            mask = mask[:, None, None, :]

            # (N, 1, 1, S_kv) -> (N, H, S, S_kv)
            mask = mask.expand(-1, seqs.size(1), seqs.size(2), -1)

            if attn_mask is not None:

                attn_mask.masked_fill_(mask, -torch.inf)
                mask = attn_mask

        # elif isinstance(attn_mask, CausalAttentionMask):
        #     # PyTorch SDPA supports only full causal attention.
        #     if attn_mask.attn_len is None and attn_mask.attn_window_len is None:
        #         mask = None
        #
        #         is_causal = True
        #     else:
        #         # ([H], S, S_kv)
        #         mask = attn_mask.materialize()
        elif attn_mask is not None:
            # ([H], S, S_kv)
            mask = attn_mask
        else:
            mask = None

        if mask.dtype == torch.bool:
            mask = mask.logical_not()

        attn = F.scaled_dot_product_attention(  # type: ignore[attr-defined]
            seqs,
            keys,
            values,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        return attn, None

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"attn_dropout_p={self.attn_dropout_p}"


@final
class NaiveSDPA(SDPA):
    """Computes scaled dot-product attention using a Python implementation."""

    attn_dropout_p: float

    def __init__(self, *, attn_dropout_p: float = 0.0) -> None:
        """
        :param attn_dropout_p:
            The dropout probability on attention weights.
        """
        super().__init__()

        self.name = "naive_sdpa"
        self.attn_dropout_p = attn_dropout_p

    @finaloverrides
    def forward(
            self,
            seqs: Tensor,
            keys: Tensor,
            key_padding_mask,
            values: Tensor,
            *,
            attn_mask=None,
            needs_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return _naive_scaled_dot_product_attention(
            seqs,
            keys,
            key_padding_mask,
            values,
            attn_mask,
            self.attn_dropout_p,
            needs_weights,
            self.training,
        )

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"attn_dropout_p={self.attn_dropout_p}"


def _naive_scaled_dot_product_attention(
        seqs: Tensor,
        keys: Tensor,
        key_padding_mask,
        values: Tensor,
        attn_mask,
        dropout_p: float,
        needs_weights: bool,
        training: bool,
) -> Tuple[Tensor, Optional[Tensor]]:
    # (N, H, S, K) @ (N, H, K, S_kv) = (N, H, S, S_kv)
    attn_weights = torch.matmul(seqs, keys.transpose(-1, -2))

    attn_weights = attn_weights * (seqs.size(-1) ** -0.5)

    if attn_mask is not None:
        # (S, S_kv)
        m = attn_mask

        # (N, H, S, S_kv) + (S, S_kv) -> (N, H, S, S_kv)
        attn_weights = attn_weights + m

    if key_padding_mask is not None:
        # (N, S_kv)
        m = key_padding_mask

        m = m[:, None, None, :]

        # (N, H, S, S_kv) + (N, 1, 1, S_kv) -> (N. H, S, S_kv)
        attn_weights = torch.where(m, attn_weights, -torch.inf)

    # For numerical stability run in single precision.
    attn_weights = softmax(attn_weights, dim=-1, dtype=torch.float32)

    attn_weights = attn_weights.type_as(seqs)

    if training and dropout_p > 0.0:
        attn_weights = dropout(attn_weights, dropout_p)

    # (N, H, S, S_kv) @ (N, H, S_kv, V) = (N, H, S, V)
    attn = torch.matmul(attn_weights, values)

    return attn, attn_weights if needs_weights else None


@final
class RelativePositionSDPA(SDPA):
    """Computes scaled dot-product attention as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1901.02860`."""

    model_dim: int
    num_heads: int
    u_bias: Parameter
    v_bias: Parameter
    r_proj: Linear
    inner_sdpa: SDPA

    def __init__(
            self,
            model_dim: int,
            num_heads: int,
            pos_encoding,
            *,
            inner_sdpa: Optional[SDPA] = None,
            device=None,
            dtype=None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param: num_heads:
            The number of attention heads.
        :param: pos_encoding:
            The relative positional encoding table.
        :param inner_sdpa:
            The actual :class:`SDPA` module to compute head attentions.
        """
        super().__init__()

        self.name = "relative_sdpa"

        if model_dim % num_heads != 0:
            raise ValueError(
                f"`model_dim` must be a multiple of `num_heads` ({num_heads}), but is {model_dim} instead."
            )

        self.model_dim = model_dim
        self.num_heads = num_heads

        if pos_encoding.encoding_dim != model_dim:
            raise ValueError(
                f"`encoding_dim` of `pos_encoding` must be equal to `model_dim` ({model_dim}), but is {pos_encoding.encoding_dim} instead."
            )

        self.pos_encoding = pos_encoding

        head_dim = model_dim // num_heads

        self.u_bias = Parameter(
            torch.empty((num_heads, head_dim), device=device, dtype=dtype)
        )
        self.v_bias = Parameter(
            torch.empty((num_heads, head_dim), device=device, dtype=dtype)
        )

        self.r_proj = Linear(
            model_dim, model_dim, bias=False, device=device, dtype=dtype
        )

        if inner_sdpa is not None:
            self.inner_sdpa = inner_sdpa
        else:
            self.inner_sdpa = create_default_sdpa()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        nn.init.xavier_normal_(self.u_bias)
        nn.init.xavier_normal_(self.v_bias)

    @finaloverrides
    def forward(
            self,
            seqs: Tensor,
            keys: Tensor,
            key_padding_mask,
            values: Tensor,
            *,
            attn_mask=None,
            needs_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        q = seqs
        k = keys

        # (H, K_h) -> (H, 1, K_h)
        u_bias = self.u_bias.unsqueeze(1)
        v_bias = self.v_bias.unsqueeze(1)

        # (N, H, S, K_h) + (H, 1, K_h) -> (N, H, S, K_h)
        q_with_u_bias = q + u_bias
        q_with_v_bias = q + v_bias

        # (N, H, 2 x S - 1, K_h)
        r = self._compute_r(k, batch_size=q.size(0))

        # (N, H, S, K_h) @ (N, H, K_h, 2 x S - 1) = (N, H, S, 2 x S - 1)
        bd = torch.matmul(q_with_v_bias, r.transpose(-1, -2))

        # (N, H, S, 2 x S - 1) -> (N, H, S, S)
        bd = self._shift_bd(bd)


        # We treat `bd` as an attention mask to take advantage of efficient SDPA
        # implementations.
        bd = bd * (q.size(-1) ** -0.5)

        if attn_mask is None:
            mask = bd
        else:
            mask = bd + attn_mask

        attn_mask = mask

        output = self.inner_sdpa(  # type: ignore[no-any-return]
            q_with_u_bias,
            k,
            key_padding_mask,
            values,
            attn_mask=attn_mask,
            needs_weights=needs_weights,
        )

        return output

    def _compute_r(self, k: Tensor, batch_size: int) -> Tensor:
        # (2 x S - 1, K)
        r = self.pos_encoding(k)

        # (2 x S - 1, K) -> (2 x S - 1, K)
        r = self.r_proj(r)

        # (2 x S - 1, K) -> (1, 2 x S - 1, H, K_h)
        r = r.view(1, -1, self.num_heads, k.size(-1))

        # (1, 2 x S - 1, H, K_h) -> (N, H, 2 x S - 1, K_h)
        r = r.transpose(1, 2).expand(batch_size, -1, -1, -1)

        return r  # type: ignore[no-any-return]

    def _shift_bd(self, bd: Tensor) -> Tensor:
        # (N, H, S, 2 x S - 1) -> (N, H, S, 2 x S)
        x = pad(bd, (1, 0))

        # (N, H, S, 2 x S) -> (N, H, 2 x S, S)
        x = x.view(x.size(0), x.size(1), x.size(3), x.size(2))

        # Discard the first set of positive positions.
        # (N, H, 2 x S, S) -> (N, H, 2 x S - 1, S)
        x = x[:, :, 1:, :]

        # This op effectively shifts each row by an extra step.
        # (N, H, 2 x S - 1, S) -> (N, H, S, 2 x S - 1)
        x = x.view_as(bd)

        # Discard positions used for shift.
        # (N, H, S, 2 x S - 1) -> (N, H, S, S)
        x = x[..., : bd.size(2)]

        return x

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}, num_heads={self.num_heads}"


class RelativePositionalEncoding(Module):
    """Produces relative positional encodings as described in Appendix B of
    :cite:t:`dai2019transformerxl`."""

    encoding_dim: int
    max_seq_len: int
    freqs: Tensor

    def __init__(
            self,
            encoding_dim: int,
            max_seq_len: int,
            *,
            device=None,
            dtype=None,
    ) -> None:
        """
        :param encoding_dim:
            The dimensionality of positional encodings.
        :param max_seq_len:
            The expected maximum sequence length.
        """
        super().__init__()

        if encoding_dim % 2 != 0:
            raise ValueError(
                f"`encoding_dim` must be even, but is {encoding_dim} instead."
            )

        self.encoding_dim = encoding_dim
        self.max_seq_len = max_seq_len

        freqs = torch.empty(
            ((max_seq_len * 2) - 1, encoding_dim), device=device, dtype=dtype
        )

        self.register_buffer("freqs", freqs, persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        self.reset_non_persistent_buffers()

    def reset_non_persistent_buffers(self) -> None:
        """Reset the non-persistent buffers of the module."""
        fp32_freqs = self.freqs.float()

        device, dtype = fp32_freqs.device, fp32_freqs.dtype

        positive_half = fp32_freqs[: self.max_seq_len]
        negative_half = fp32_freqs[self.max_seq_len:]

        # (S)
        steps = torch.arange(self.max_seq_len, device=device, dtype=dtype)

        # (E / 2)
        indices = torch.arange(0, self.encoding_dim, step=2, device=device, dtype=dtype)

        freqs = torch.exp(indices * -math.log(10000.0) / self.encoding_dim)

        # (S) x (E / 2) -> (S, E / 2)
        freqs = torch.outer(steps, freqs)

        flipped_freqs = freqs.flip([0])

        # A mirrored matrix of sinusoidal positive and negative positional
        # encodings to use in shift trick.
        #
        # [max, ...,  3,  2,  1,  0, -1, -2, -3, ..., min]
        torch.sin(flipped_freqs, out=positive_half[:, 0::2])
        torch.cos(flipped_freqs, out=positive_half[:, 1::2])

        torch.sin(-1 * freqs[1:], out=negative_half[:, 0::2])
        torch.cos(-1 * freqs[1:], out=negative_half[:, 1::2])

        self.freqs.copy_(fp32_freqs)

    def forward(self, seqs: Tensor) -> Tensor:
        """
        :param seqs:
            The sequences for which to return positional encodings. *Shape:*
            :math:`(*,S,E)`, where :math:`*` is any number of batch dimensions
            including none, :math:`S` is the sequence length, and :math:`E` is
            the dimensionality of the positional encodings.

        :returns:
            The positional encodings to use in shift trick in
            :class:`RelativePositionSDPA`. *Shape:* :math:`(2 x S - 1, E)`,
            where :math:`S` is the sequence length and :math:`E` is the
            dimensionality of the positional encodings.
        """
        seq_len = seqs.size(-2)

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"The input sequence length must be less than or equal to the maximum sequence length ({self.max_seq_len}), but is {seq_len} instead."
            )

        return self.freqs[self.max_seq_len - seq_len: self.max_seq_len + seq_len - 1]

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"encoding_dim={self.encoding_dim}, max_seq_len={self.max_seq_len}"


@final
class ShawRelativePositionSDPA(SDPA):
    """Computes scaled dot-product attention as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1803.02155`."""

    model_dim: int
    num_heads: int
    max_left_rel_pos: int
    max_right_rel_pos: int
    rel_k_embed: StandardEmbedding
    rel_v_embed: Optional[StandardEmbedding]
    inner_sdpa: SDPA

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        max_left_rel_pos: int,
        *,
        max_right_rel_pos: Optional[int] = None,
        use_rel_pos_values: bool = False,
        inner_sdpa: Optional[SDPA] = None,
        device=None,
        dtype=None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param: num_heads:
            The number of attention heads.
        :param: max_left_rel_pos:
            The left clipping value for relative positions.
        :param: max_right_rel_pos:
            The right clipping value for relative positions.
        :param: use_rel_pos_values:
            If ``True``, uses relative position values to compute attention.
        :param inner_sdpa:
            The actual :class:`SDPA` module to compute head attentions.
        """
        super().__init__()

        self.name = "relative_shaw_sdpa"

        if model_dim % num_heads != 0:
            raise ValueError(
                f"`model_dim` must be a multiple of `num_heads` ({num_heads}), but is {model_dim} instead."
            )

        self.model_dim = model_dim
        self.num_heads = num_heads

        head_dim = model_dim // num_heads

        self.max_left_rel_pos = max_left_rel_pos

        self.max_right_rel_pos = (
            max_left_rel_pos if max_right_rel_pos is None else max_right_rel_pos
        )

        num_pos = self.max_left_rel_pos + 1 + self.max_right_rel_pos

        self.rel_k_embed = StandardEmbedding(
            num_pos, head_dim, init_fn=init_shaw_embedding, device=device, dtype=dtype
        )

        if use_rel_pos_values:
            self.rel_v_embed = StandardEmbedding(
                num_pos,
                head_dim,
                init_fn=init_shaw_embedding,
                device=device,
                dtype=dtype,
            )
        else:
            self.register_module("rel_v_embed", None)

        if inner_sdpa is not None:
            self.inner_sdpa = inner_sdpa
        else:
            self.inner_sdpa = create_default_sdpa()

    @finaloverrides
    def forward(
        self,
        seqs: Tensor,
        keys: Tensor,
        key_padding_mask,
        values: Tensor,
        *,
        attn_mask = None,
        needs_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        q_len = seqs.size(2)

        # (N, H, S, K_h) @ (N, H, K_h, S_kv) = (N, H, S, S_kv)
        attn_weights = torch.matmul(seqs, keys.transpose(-1, -2))

        # (S_kv, S_kv)
        rel_indices = self._get_relative_indices(keys)

        # (S_kv, S_kv, K_h)
        rel_keys = self.rel_k_embed(rel_indices)

        # (S_kv, S_kv, K_h) -> (S, S_kv, K_h)
        rel_keys = rel_keys[-q_len:]

        # (N, H, S, K_h) @ (S, S_kv, K_h) = (N, H, S, S_kv)
        rel_attn_weights = torch.einsum("nhsk,stk->nhst", seqs, rel_keys)

        # We treat `rel_attn_weights` as an attention mask to take advantage of
        # efficient SDPA implementations.
        rel_attn_weights = rel_attn_weights * (seqs.size(-1) ** -0.5)

        if attn_mask is None:
            mask = rel_attn_weights
        else:
            mask = rel_attn_weights + attn_mask

        attn_mask = mask

        attn, attn_weights = self.inner_sdpa(  # type: ignore[no-any-return]
            seqs,
            keys,
            key_padding_mask,
            values,
            attn_mask=attn_mask,
            needs_weights=needs_weights or self.rel_v_embed is not None,
        )

        if self.rel_v_embed is not None:
            assert attn_weights is not None

            # (S_kv, S_kv, V_h)
            rel_pos_values = self.rel_v_embed(rel_indices)

            # (S_kv, S_kv, V_h) -> (S, S_kv, V_h)
            rel_pos_values = rel_pos_values[-q_len:]

            # (N, H, S, S_kv) @ (S, S_kv, V_h) = (N, H, S, V_h)
            rel_attn = torch.einsum("nhst,stv->nhsv", attn_weights, rel_pos_values)

            attn = attn + rel_attn

        return attn, attn_weights if needs_weights else None

    def _get_relative_indices(self, keys: Tensor) -> Tensor:
        # (S, 1)
        indices = torch.arange(keys.size(2), device=keys.device).unsqueeze(0)

        # (S, S)
        rel_indices = indices - indices.transpose(0, 1)

        rel_indices = torch.clamp(
            rel_indices, -self.max_left_rel_pos, self.max_right_rel_pos
        )

        return rel_indices + self.max_left_rel_pos

    def get_relative_indices_three_dim(self, keys: Tensor) -> Tensor:
        # (S, 1)
        indices = torch.arange(keys.size(1), device=keys.device).unsqueeze(0)

        # (S, S)
        rel_indices = indices - indices.transpose(0, 1)

        rel_indices = torch.clamp(
            rel_indices, -self.max_left_rel_pos, self.max_right_rel_pos
        )

        return rel_indices + self.max_left_rel_pos

    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"model_dim={self.model_dim}, "
            f"num_heads={self.num_heads}, "
            f"max_left_rel_pos={self.max_left_rel_pos}, "
            f"max_right_rel_pos={self.max_right_rel_pos}"
        )


def init_shaw_embedding(embed: StandardEmbedding) -> None:
    """Initialize ``embed`` for use in :class:`ShawRelativePositionSDPA`."""
    nn.init.xavier_uniform_(embed.weight)



# utility functions to create default attention layers

class SDPAFactory(Protocol):
    """Constructs instances of :class:`SDPA`."""

    def __call__(self, *, attn_dropout_p: float = 0.0) -> SDPA:
        """
        :param attn_dropout_p:
            The dropout probability on attention weights.
        """


def _get_fallback_sdpa_factory() -> SDPAFactory:
    return TorchSDPA


_sdpa_factory: SDPAFactory = _get_fallback_sdpa_factory()


def set_default_sdpa_factory(factory: Optional[SDPAFactory]) -> None:
    """Set the default :class:`SDPA` factory."""
    global _sdpa_factory

    if factory is not None:
        _sdpa_factory = factory
    else:
        _sdpa_factory = _get_fallback_sdpa_factory()


def create_default_sdpa(*, attn_dropout_p: float = 0.0) -> SDPA:
    """Create an instance of the default :class:`SDPA`.

    :param attn_dropout_p:
        The dropout probability on attention weights.
    """
    return _sdpa_factory(attn_dropout_p=attn_dropout_p)


@contextmanager
def default_sdpa_factory(factory: Optional[SDPAFactory]) -> Generator[None, None, None]:
    """Set a temporary default :class:`SDPA` factory."""
    original_factory = _sdpa_factory

    set_default_sdpa_factory(factory)

    try:
        yield
    finally:
        set_default_sdpa_factory(original_factory)