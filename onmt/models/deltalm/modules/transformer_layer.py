# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
# from fairseq import utils
# from onmt.models.speech_recognizer.fairseq_wav2vec2.fairseq_modules import MultiheadAttention
from onmt.modules.layer_norm import LayerNorm
# from fairseq.modules import LayerNorm, MultiheadAttention
# from fairseq.modules.fairseq_dropout import FairseqDropout
# from fairseq.modules.quant_noise import quant_noise
from torch import Tensor
from .utils import get_activation_fn
from .multihead_attention import MultiHeadAttention


def dropout_residual_connection(x, residual, dropout_module, is_training):

    return dropout_add_jit(x, residual, dropout_module.p, is_training)

@torch.jit.script
def dropout_add_jit(x, residual, prob, is_training) :
    # type: (Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x, p=prob, training=is_training)
    out = residual + out
    return out

def linear_act_linear(x, fc1, fc2, prob, is_training, activation_func):

    out = fc1(x)
    out = activation_func(out)
    out = torch.nn.functional.dropout(out, p=prob, training=is_training)
    out = fc2(out)

    return out

class TransformerEncoderLayerBase(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.encoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.encoder_embed_dim
        # self.quant_noise = cfg.quant_noise.pq
        # self.quant_noise_block_size = cfg.quant_noise.pq_block_size
        self.self_attn = self.build_self_attention(self.embed_dim, cfg)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = nn.Dropout(cfg.dropout)
        self.activation_fn = get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        self.activation_dropout_module = nn.Dropout(activation_dropout_p)
        self.normalize_before = cfg.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            cfg.encoder_ffn_embed_dim
        )
        self.fc2 = self.build_fc2(
            cfg.encoder_ffn_embed_dim,
            self.embed_dim
        )

        self.checkpoint_activations = cfg.checkpoint_activations

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.activation_fn_name = cfg.activation_fn

        # Fused MLP config
        self.fused = False
        self.fused_function = None
        if self.activation_fn_name == 'relu':
            from onmt.modules.mlp.mlp import mlp_relu_function
            if mlp_relu_function is not None:
                self.fused_function = mlp_relu_function
                self.fused = True
        elif self.activation_fn_name == 'gelu':
            from onmt.modules.mlp.mlp import mlp_gelu_function
            if mlp_gelu_function is not None:
                self.fused_function = mlp_gelu_function
                self.fused = True

        # Adapter config
        self.n_languages = -1
        self.has_adapter = False

    def build_fc1(self, input_dim, output_dim, *args):
        return nn.Linear(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim, *args):
        return nn.Linear(input_dim, output_dim)

    def build_self_attention(self, embed_dim, cfg):
        from pretrain_module.modeling_mbart import MBartAttention

        return MBartAttention(
            embed_dim=embed_dim,
            num_heads=cfg.encoder_attention_heads,
            dropout=cfg.attention_dropout,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
        max_len=None, cu_seqlens=None,
        **kwargs
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x, _, _ = self.self_attn(
            hidden_states=x,
            attention_mask=encoder_padding_mask,
            output_attentions=False,
            max_len=max_len, cu_seqlens=cu_seqlens
        )

        # x = self.dropout_module(x)
        # x = self.residual_connection(x, residual)
        x = dropout_residual_connection(x, residual, self.dropout_module, self.training)

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        if self.fused and x.is_cuda:
            dropout_p = self.activation_dropout_module.p if self.training else 0.0

            weights = [self.fc1.weight, self.fc2.weight]
            biases = [self.fc1.bias, self.fc2.bias]

            x = self.fused_function(dropout_p, False, x, *weights, *biases)

        else:
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)

        x = dropout_residual_connection(x, residual, self.dropout_module, self.training)

        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class TransformerDecoderLayerBase(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = cfg.decoder_embed_dim
        self.dropout_module = nn.Dropout(cfg.dropout)

        self.cross_self_attention = cfg.cross_self_attention

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            cfg,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        self.activation_dropout_module = nn.Dropout(float(activation_dropout_p))
        self.normalize_before = cfg.decoder_normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, cfg)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            cfg.decoder_ffn_embed_dim
        )
        self.fc2 = self.build_fc2(
            cfg.decoder_ffn_embed_dim,
            self.embed_dim
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

        elf.checkpoint_activations

        # self.activation_fn_name = cfg.activation_fn
        # self.fused = False
        # self.fused_function = None
        # if self.activation_fn_name == 'relu':
        #     from onmt.modules.mlp.mlp import mlp_relu_function
        #     if mlp_relu_function is not None:
        #         self.fused_function = mlp_relu_function
        #         self.fused = True
        # elif self.activation_fn_name == 'gelu':
        #     from onmt.modules.mlp.mlp import mlp_gelu_function
        #     if mlp_gelu_function is not None:
        #         self.fused_function = mlp_gelu_function
        #         self.fused = True

    def build_fc1(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_self_attention(
        self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False
    ):
        from pretrain_module.modeling_mbart import MBartAttention

        return MBartAttention(  # MBartAutoRegressiveSelfAttentionSLow(
            embed_dim=embed_dim,
            num_heads=cfg.decoder_attention_heads,
            dropout=cfg.attention_dropout,
            is_decoder=True,
        )
        # return MultiHeadAttention(
        #     embed_dim,
        #     cfg.decoder_attention_heads,
        #     dropout=cfg.attention_dropout,
        #     add_bias_kv=add_bias_kv,
        #     add_zero_attn=add_zero_attn,
        #     self_attention=not cfg.cross_self_attention
        # )

    def build_encoder_attention(self, embed_dim, cfg):

        from pretrain_module.modeling_mbart import MBartCrossAttention

        return MBartCrossAttention(
            embed_dim,
            cfg.decoder_attention_heads,
            dropout=cfg.attention_dropout,
        )

        # return MultiHeadAttention(
        #     embed_dim,
        #     cfg.decoder_attention_heads,
        #     kdim=cfg.encoder_embed_dim,
        #     vdim=cfg.encoder_embed_dim,
        #     dropout=cfg.attention_dropout,
        #     encoder_decoder_attention=True
        # )

    def residual_connection(self, x, residual):

        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        **kwargs
    ):
        """
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x, attn, _ = self.self_attn(
            hidden_states=x,
            attention_mask=self_attn_mask,
            output_attentions=False
        )

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            x, attn, _ = self.encoder_attn(
                hidden_states=x,
                key_value_states=encoder_out,
                attention_mask=encoder_padding_mask,
                output_attentions=False,
                # incremental=incremental, incremental_cache=incremental_cache,
                # checkpointing=checkpointing_cross_attn,
                # lang=lang, atb=atb
            )

            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, attn, None
