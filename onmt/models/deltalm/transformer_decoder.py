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
from pretrain_module.modeling_mbart import index_copy
import numpy as np

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
            output_projection=None,
            opt=None
    ):
        self.adapter = None
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
        print("Decoder padding idx:", self.padding_idx)
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
        self.checkpoint_activations = cfg.checkpoint_activations

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

        from onmt.modules.optimized.flash_mha import flash_bert_mha
        self.fast_bert_mha = flash_bert_mha

        self.n_languages = 0

            # for layer in self.layers:
            #     layer.add_adapters(opt.n_languages, adapter_location=opt.decoder_adapter)

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = transformer_layer.TransformerDecoderLayerBase(cfg, no_encoder_attn)

        # removed checkpoint and fsdp
        return layer

    def add_adapters(self, n_languages):

        from .modules.efficient_adapters import EfficientAdapter
        self.adapter = EfficientAdapter(n_languages * self.num_layers,
                                        self.embed_dim, self.embed_dim // 4)
        self.n_languages = n_languages

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            checkpointing_ffn=False,
            checkpointing_self_attn=False,
            checkpointing_cross_attn=False,
            lang=None,
            **kwargs,
    ):
        bsz, qlen = input_ids.size()
        klen = encoder_hidden_states.size(0)

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
        can_run_fast_bert_mha = False

        if self.fast_bert_mha is not None and torch.is_autocast_enabled():
            can_run_fast_bert_mha = True

            # unpadding x
            if attention_mask is None:
                padding_mask = input_ids.new_zeros(bsz, qlen)
            else:
                padding_mask = attention_mask
            padding_mask = padding_mask.contiguous().long()
            lengths = (1 - padding_mask).sum(dim=1)
            lengths = lengths.cpu().tolist()  # list of lengths for B seqs

            x = x.view(-1, x.size(-1))
            non_pad_indices = torch.nonzero(padding_mask.view(-1).ne(1)).squeeze(1)
            x = x.index_select(0, non_pad_indices)
            max_len = max(lengths)
            # cumulative sequence lengths (required input for fmha)
            a = torch.tensor(np.array([0] + lengths), dtype=torch.int32)
            cu_seqlens = torch.cumsum(a, 0).to(dtype=torch.int32, device=x.device)

            non_pad_indices_q = non_pad_indices

            # unpadding context
            # transposing from [T x B x H] to [B x T x H]
            encoder_hidden_states = encoder_hidden_states.transpose(0, 1).contiguous()
            padding_mask = encoder_attention_mask
            if padding_mask is None:
                context_len = encoder_hidden_states.size(1)
                padding_mask = input_ids.new_zeros(bsz, context_len)
            padding_mask = padding_mask.long()
            lengths = (1 - padding_mask).sum(dim=1)
            lengths = lengths.cpu().tolist()  # list of lengths for B seqs
            encoder_hidden_states = encoder_hidden_states.view(-1, encoder_hidden_states.size(-1))
            non_pad_indices_kv = torch.nonzero(padding_mask.view(-1).ne(1)).squeeze(1)
            encoder_hidden_states = encoder_hidden_states.index_select(0, non_pad_indices_kv)

            max_len_kv = max(lengths)
            # cumulative sequence lengths (required input for fmha)
            a = torch.tensor(np.array([0] + lengths), dtype=torch.int32)
            cu_seqlens_kv = torch.cumsum(a, 0).to(dtype=torch.int32, device=encoder_hidden_states.device)

            self_attn_mask = None
        else:
            x = x.transpose(0, 1).contiguous()
            max_len, cu_seqlens = None, None
            max_len_kv, cu_seqlens_kv = None, None

            # causal masking.
            self_attn_mask = torch.triu(
                x.new_ones(qlen, qlen), diagonal=1).bool()
            non_pad_indices_q, non_pad_indices_kv = None, None

        self_attn_padding_mask: Optional[Tensor] = None

        # decoder layers
        attns = list()
        for idx, layer in enumerate(self.layers):

            x, layer_attn, _ = layer(
                x,
                encoder_hidden_states,
                encoder_attention_mask,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                checkpointing_ffn=checkpointing_ffn,
                checkpointing_self_attn=checkpointing_self_attn,
                checkpointing_cross_attn=checkpointing_cross_attn,
                max_len=max_len, cu_seqlens=cu_seqlens,
                max_len_kv=max_len_kv, cu_seqlens_kv=cu_seqlens_kv,
            )

            # run through the adapter
            if self.adapter is not None:
                assert lang is not None
                adapter_id = self.adapter.num_modules // self.num_layers * idx + lang
                x = self.adapter(x, adapter_id)

            attns.append(layer_attn)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if can_run_fast_bert_mha:
            seq_len = qlen
            x = index_copy(x, non_pad_indices_q, bsz * seq_len)
            x = x.view(bsz, seq_len, -1).transpose(0, 1).contiguous()

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

        bsz, qlen = x.size(0), x.size(1)

        using_buffer = (x.size(1) > 1 and len(buffers) > 0)

        if buffering:
            # use the last value of input to continue decoding
            if using_buffer:
                # if buffers has not been initilized and we have > 1 input length data
                # then its a prefix decoding step
                x = x[:, -1:, :]

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        can_run_fast_bert_mha = False

        if self.fast_bert_mha is not None and (torch.is_autocast_enabled() or x.dtype == torch.half) and not buffering:
            can_run_fast_bert_mha = True

            # unpadding x
            padding_mask = input_ids.new_zeros(bsz, qlen)
            padding_mask = padding_mask.contiguous().long()
            lengths = (1 - padding_mask).sum(dim=1)
            lengths = lengths.cpu().tolist()  # list of lengths for B seqs

            x = x.view(-1, x.size(-1))
            non_pad_indices = torch.nonzero(padding_mask.view(-1).ne(1)).squeeze(1)
            x = x.index_select(0, non_pad_indices)
            max_len = max(lengths)
            # cumulative sequence lengths (required input for fmha)
            a = torch.tensor(np.array([0] + lengths), dtype=torch.int32)
            cu_seqlens = torch.cumsum(a, 0).to(dtype=torch.int32, device=x.device)

            non_pad_indices_q = non_pad_indices

            # unpadding context
            # transposing from [T x B x H] to [B x T x H]
            encoder_hidden_states = encoder_hidden_states.transpose(0, 1).contiguous()
            padding_mask = encoder_attention_mask
            if padding_mask is None:
                context_len = encoder_hidden_states.size(1)
                padding_mask = input_ids.new_zeros(bsz, context_len)
            padding_mask = padding_mask.long()
            lengths = (1 - padding_mask).sum(dim=1)
            lengths = lengths.cpu().tolist()  # list of lengths for B seqs
            encoder_hidden_states = encoder_hidden_states.view(-1, encoder_hidden_states.size(-1))
            non_pad_indices_kv = torch.nonzero(padding_mask.view(-1).ne(1)).squeeze(1)
            encoder_hidden_states = encoder_hidden_states.index_select(0, non_pad_indices_kv)

            max_len_kv = max(lengths)
            # cumulative sequence lengths (required input for fmha)
            a = torch.tensor(np.array([0] + lengths), dtype=torch.int32)
            cu_seqlens_kv = torch.cumsum(a, 0).to(dtype=torch.int32, device=encoder_hidden_states.device)

            self_attn_mask = None
        else:

            non_pad_indices_q, non_pad_indices_kv = None, None
            # B x T x C -> T x B x C
            x = x.transpose(0, 1).contiguous()
            max_len = None
            cu_seqlens = None
            max_len_kv = None
            cu_seqlens_kv = None

            # causal masking.
            self_attn_mask = torch.triu(
                x.new_ones(qlen, qlen), diagonal=1).bool()

            if buffering and using_buffer:
                self_attn_mask = self_attn_mask[-1:, :]

        # decoder layers
        attns = list()
        for idx, layer in enumerate(self.layers):

            if buffering:
                buffer = buffers[idx] if idx in buffers else None
            else:
                buffer = None

            x, layer_attn, buffer = layer(
                x,
                encoder_hidden_states,
                encoder_attention_mask,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=None,
                max_len = max_len, cu_seqlens = cu_seqlens,
                max_len_kv = max_len_kv, cu_seqlens_kv = cu_seqlens_kv,
                incremental=buffering, incremental_cache=buffer,
            )

            if buffering:
                decoder_state.update_attention_buffer(buffer, idx)

            attns.append(layer_attn)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if can_run_fast_bert_mha:
            seq_len = qlen
            x = index_copy(x, non_pad_indices_q, bsz * seq_len)
            x = x.view(bsz, seq_len, -1).transpose(0, 1).contiguous()

        output = x[-1].unsqueeze(0)
        coverage = attns[-1]

        if coverage is None:
            coverage = output.new_zeros(bsz, seq_len, seq_len)

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