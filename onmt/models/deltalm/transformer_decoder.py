# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional
from collections import defaultdict

import torch
import torch.nn as nn

from .modules.positional_embeddings import PositionalEmbedding, SinusoidalPositionalEmbedding
from .modules.layer_drop import LayerDropModuleList
from onmt.modules.layer_norm import LayerNorm
from .modules.transformer_layer import TransformerDecoderLayerBase
from torch import Tensor

class TransformerDecoderBase(nn.Module):
    """
        Transformer decoder consisting of *cfg.decoder_layers* layers. Each layer
        is a :class:`TransformerDecoderLayer`.
        Args:
            args (argparse.Namespace): parsed command-line arguments
            dictionary (~fairseq.data.Dictionary): decoding dictionary
            embed_tokens (torch.nn.Embedding): output embedding
            no_encoder_attn (bool, optional): whether to attend to encoder outputs
                (default: False).
    """

    def __init__(
            self,
            cfg,
            embed_tokens,
            no_encoder_attn=False,
            output_projection=None
    ):
        self.cfg = cfg
        super(TransformerDecoderBase, self).__init__()

        self.register_buffer("version", torch.Tensor([3]))
        self.dropout_module = nn.Dropout(
            cfg.dropout
        )
        self.decoder_layerdrop = cfg.decoder_layerdrop
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder_embed_dim
        self.embed_dim = embed_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        self.project_in_dim = (
            torch.nn.Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.decoder_learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )

        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = cfg.cross_self_attention

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])

        self.layers.extend(
            [
                self.build_decoder_layer(cfg, no_encoder_attn)
                for _ in range(cfg.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if cfg.decoder_normalize_before and not cfg.no_decoder_final_norm:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = transformer_layer.TransformerDecoderLayerBase(cfg, no_encoder_attn)

        # removed checkpoint and fsdp
        return layer

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            **kwargs,
            # prev_output_tokens,
            # encoder_hidden_states,
            # encoder_attn_mask,
            # encoder_out: Optional[Dict[str, List[Tensor]]] = None,

    ):

        bs, slen = input_ids.size()

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                input_ids, incremental_state=None
            )

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(input_ids)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        # if prev_output_tokens.eq(self.padding_idx).any():
        #     self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        qlen = x.size(0)
        self_attn_mask = torch.triu(
            x.new_ones(qlen, qlen), diagonal=1).bool()

        # decoder layers
        attns = list()
        for idx, layer in enumerate(self.layers):

            x, layer_attn, _ = layer(
                x,
                encoder_hidden_states,
                encoder_attention_mask,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask
            )
            attns.append(layer_attn)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x, attns

    def step(self, input, decoder_state, **kwargs):

        # context is stored in the decoder state in [T B H] format
        encoder_hidden_states = decoder_state.context
        encoder_attention_mask = decoder_state.src_mask

        buffers = decoder_state.attention_buffers
        lang = decoder_state.tgt_lang
        atb = decoder_state.tgt_atb
        src_lang = decoder_state.src_lang
        buffering = decoder_state.buffering

        input_ids = input
        input_shape = input_ids.size()
        time_step = input.size(1)

        input_ = input

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                input_ids, incremental_state=None
            )

        x = self.embed_scale * self.embed_tokens(input_ids)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        qlen = x.size(0)
        self_attn_mask = torch.triu(
            x.new_ones(qlen, qlen), diagonal=1).bool()

        # decoder layers
        attns = list()
        for idx, layer in enumerate(self.layers):
            x, layer_attn, _ = layer(
                x,
                encoder_hidden_states,
                encoder_attention_mask,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=None
            )
            attns.append(layer_attn)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        output = x[-1].unsqueeze(0)
        coverage = attns[-1]

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = output
        output_dict['coverage'] = coverage
        output_dict['context'] = encoder_hidden_states
        return output_dict

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
                self._future_mask.size(0) == 0
                or (not self._future_mask.device == tensor.device)
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]