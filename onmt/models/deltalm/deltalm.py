import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .transformer_encoder import TransformerEncoderBase
from .transformer_decoder import TransformerDecoderBase

from .modules.transformer_layer import TransformerDecoderLayerBase
from .modules.utils import get_activation_fn
from onmt.modules.layer_norm import LayerNorm
from .modules.multihead_attention import MultiHeadAttention

from onmt.modules.optimized.dropout_add import fused_dropout_add
from pretrain_module.modeling_mbart import index_copy


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



def upgrade_state_dict_for_deltalm(
        state_dict: Dict[str, Any], pretrained_deltalm_checkpoint: str, is_encoder=True,
) -> Dict[str, Any]:
    if not os.path.exists(pretrained_deltalm_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_deltalm_checkpoint))

    with open(pretrained_deltalm_checkpoint, "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))
    if 'weights' in state:

        deltalm_state_dict = state['weights']
    elif 'model' in state:
        deltalm_state_dict = state['model']
    else:
        deltalm_state_dict = state

    new_deltalm_state_dict = {}

    for key in deltalm_state_dict.keys():
        if is_encoder:
            if key.startswith('encoder.') or key.startswith('src_embedding.'):
                new_key = key.replace('encoder.', '')
                new_key = new_key.replace('src_embedding.', '')
                new_deltalm_state_dict[new_key] = deltalm_state_dict[key]
        else:
            if key.startswith('decoder.') or key.startswith('tgt_embedding.'):
                new_key = key.replace('decoder.', '')
                new_key = new_key.replace('tgt_embedding.', '')
                new_deltalm_state_dict[new_key] = deltalm_state_dict[key]

    deltalm_state_dict = new_deltalm_state_dict

    # print(deltalm_state_dict.keys())

    for key in deltalm_state_dict.keys():

        if "output_projection" in key:
            continue

        map_key = key
        map_key = map_key.replace('.ffn_1.fc1', '.fc3')
        map_key = map_key.replace('.ffn_1.fc2', '.fc4')
        map_key = map_key.replace('.ffn_2', '')
        map_key = map_key.replace('.ffn.', '.')
        map_key = map_key.replace('emb_layer_norm', 'layernorm_embedding')

        # print(key, state_dict[map_key].size(), deltalm_state_dict[key].size())
        assert map_key in state_dict, map_key
        if 'embed_positions' in key or 'embed_tokens' in key:
            left_size = state_dict[map_key].size(0)
            right_size = deltalm_state_dict[key].size(0)
            if left_size <= right_size:
                state_dict[map_key] = deltalm_state_dict[key][:left_size]
            else:
                state_dict[map_key][:right_size] = deltalm_state_dict[key]
        else:
            state_dict[map_key] = deltalm_state_dict[key]

    return state_dict


class DeltaLMEncoder(TransformerEncoderBase):
    def __init__(self, args, embed_tokens, n_adapters=0):
        super().__init__(args, embed_tokens)
        self.use_adapter = False

        # only add adapters if there > 1 languages
        if n_adapters > 1:
            self.use_adapter = True
            self.add_adapters(n_adapters)

        if getattr(args, "pretrained_deltalm_checkpoint", "") != "":
            self_state_dict = self.state_dict()

            deltalm_loaded_state_dict = upgrade_state_dict_for_deltalm(
                state_dict=self_state_dict,
                pretrained_deltalm_checkpoint=args.pretrained_deltalm_checkpoint,
                is_encoder=True,
            )

            for key in self_state_dict:
                if key not in deltalm_loaded_state_dict:
                    print("Warning: key %s not found in pretrained dictionary." % key)

            for key in deltalm_loaded_state_dict:
                if key not in self_state_dict:
                    print("Warning: key %s in pretrained dictionary not found in current model." % key)

            self.load_state_dict(deltalm_loaded_state_dict, strict=True)
            print("Load DeltaLM's encoder from {0}".format(args.pretrained_deltalm_checkpoint))



class DeltaLMDecoder(TransformerDecoderBase):
    def __init__(self, args, embed_tokens, no_encoder_attn=False, opt=None, n_adapters=0):
        if opt is not None:
            args.decoder_layerdrop = opt.death_rate_decoder
            args.activation_dropout = opt.ffn_dropout

        super().__init__(args, embed_tokens, no_encoder_attn)

        # only add adapters if there > 1 languages
        if n_adapters > 1:
            self.use_adapter = True
            self.add_adapters(n_adapters)

        if getattr(args, "pretrained_deltalm_checkpoint", "") != "":
            deltalm_loaded_state_dict = upgrade_state_dict_for_deltalm(
                state_dict=self.state_dict(),
                pretrained_deltalm_checkpoint=args.pretrained_deltalm_checkpoint,
                is_encoder=False,
            )
            self.load_state_dict(deltalm_loaded_state_dict, strict=False)
            print("Load DeltaLM's decoder from {0}".format(args.pretrained_deltalm_checkpoint))

        self.model_size = args.decoder_embed_dim
        self.switchout = 0.0

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = DeltaLMDecoderLayer(args, no_encoder_attn)
        return layer



class DeltaLMDecoderLayer(TransformerDecoderLayerBase):

    def __init__(
            self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super(TransformerDecoderLayerBase, self).__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = nn.Dropout(
            args.dropout
        )

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = nn.Dropout(
            float(activation_dropout_p)
        )
        self.normalize_before = args.decoder_normalize_before
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim
        )

        self.fc3 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim
        )
        self.fc4 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim
        )

        self.ffn_layer_norm = LayerNorm(self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

        self.checkpoint_activations = args.checkpoint_activations

        self.activation_fn_name = args.activation_fn
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

        from onmt.modules.optimized.flash_mha import flash_bert_mha
        self.fast_bert_mha = flash_bert_mha

    # TODO: add incremental states
    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
            checkpointing_ffn=False,
            checkpointing_self_attn=False,
            checkpointing_cross_attn=False,
            max_len=None, cu_seqlens=None,
            max_len_kv=None, cu_seqlens_kv=None,
            **kwargs
    ):
        """
        """
        if need_head_weights:
            need_attn = True

        ###############################################

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x, attn, _ = self.self_attn(
            hidden_states=x,
            attention_mask=self_attn_mask,
            output_attentions=False,
            checkpointing=checkpointing_self_attn,
            cu_seqlens = cu_seqlens, max_len=max_len
        )

        x = dropout_residual_connection(x, residual, self.dropout_module, self.training)

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        ###############################################

        residual = x
        if self.normalize_before:
            x = self.ffn_layer_norm(x)

        if self.fused and x.is_cuda:
            dropout_p = self.activation_dropout_module.p if self.training else 0.0

            weights = [self.fc3.weight, self.fc4.weight]
            biases = [self.fc3.bias, self.fc4.bias]

            x = self.fused_function(dropout_p, checkpointing_ffn, x, *weights, *biases)

        else:
            x = self.activation_fn(self.fc3(x))
            x = self.activation_dropout_module(x)
            x = self.fc4(x)

        x = dropout_residual_connection(x, residual, self.dropout_module, self.training)
        if not self.normalize_before:
            x = self.ffn_layer_norm(x)

        ###############################################

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            x, attn, _ = self.encoder_attn(
                hidden_states=x,
                key_value_states=encoder_out,
                attention_mask=encoder_padding_mask,
                output_attentions=False,
                checkpointing=checkpointing_cross_attn,
                cu_seqlens=cu_seqlens, max_len=max_len,
                cu_seqlens_kv=cu_seqlens_kv, max_len_kv=max_len_kv
            )

            x = dropout_residual_connection(x, residual, self.dropout_module, self.training)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        ###############################################
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        if self.fused and x.is_cuda:
            dropout_p = self.activation_dropout_module.p if self.training else 0.0

            weights = [self.fc1.weight, self.fc2.weight]
            biases = [self.fc1.bias, self.fc2.bias]

            x = self.fused_function(dropout_p, checkpointing_ffn, x, *weights, *biases)

        else:
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)

        x = dropout_residual_connection(x, residual, self.dropout_module, self.training)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, attn, None


class OmniDeltaLMDecoderLayer(DeltaLMDecoderLayer):

    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
            checkpointing_ffn=False,
            checkpointing_self_attn=False,
            checkpointing_cross_attn=False,
            stack=None,
            **kwargs
    ):
        """
        Args:
            x: [T x B x D]
            encoder_out: [T x B x D]
            encoder_padding_mask: [B x T]
            self_attn_mask: [B x T] or [T x T]?
            self_attn_padding_mask: [B x T]
            need_attn:
            need_head_weights:
            checkpointing_ffn:
            checkpointing_self_attn:
            checkpointing_cross_attn:
            stack: a list of previously used inputs (used for all-attention)
            **kwargs:

        Returns:

        """
        if need_head_weights:
            need_attn = True

        ###############################################

        residual = x

        # should we need layer norm anymore? (probably)
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x, attn, _ = self.self_attn(
            hidden_states=x,
            attention_mask=self_attn_mask,
            output_attentions=False,
            checkpointing=checkpointing_self_attn
        )

        # x = self.dropout_module(x)
        # x = self.residual_connection(x, residual)
        # x = dropout_residual_connection(x, residual, self.dropout_module, self.training)

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        ###############################################

        residual = x
        if self.normalize_before:
            x = self.ffn_layer_norm(x)

        if self.fused and x.is_cuda:
            dropout_p = self.activation_dropout_module.p if self.training else 0.0

            weights = [self.fc3.weight, self.fc4.weight]
            biases = [self.fc3.bias, self.fc4.bias]

            x = self.fused_function(dropout_p, checkpointing_ffn, x, *weights, *biases)

        else:
            x = self.activation_fn(self.fc3(x))
            x = self.activation_dropout_module(x)
            x = self.fc4(x)

        # x = self.dropout_module(x)
        # x = self.residual_connection(x, residual)
        x = dropout_residual_connection(x, residual, self.dropout_module, self.training)
        if not self.normalize_before:
            x = self.ffn_layer_norm(x)

        ###############################################

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            x, attn, _ = self.encoder_attn(
                hidden_states=x,
                key_value_states=encoder_out,
                attention_mask=encoder_padding_mask,
                output_attentions=False,
                checkpointing=checkpointing_cross_attn
            )

            # x = self.dropout_module(x)
            # x = self.residual_connection(x, residual)
            x = dropout_residual_connection(x, residual, self.dropout_module, self.training)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        ###############################################
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        if self.fused and x.is_cuda:
            dropout_p = self.activation_dropout_module.p if self.training else 0.0

            weights = [self.fc1.weight, self.fc2.weight]
            biases = [self.fc1.bias, self.fc2.bias]

            x = self.fused_function(dropout_p, checkpointing_ffn, x, *weights, *biases)

        else:
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)

        # x = self.dropout_module(x)
        # x = self.residual_connection(x, residual)
        x = dropout_residual_connection(x, residual, self.dropout_module, self.training)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, attn, None
