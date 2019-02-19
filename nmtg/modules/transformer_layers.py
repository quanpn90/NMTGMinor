import math

import torch
import torch.nn.functional as F
from torch import nn

from nmtg.models.encoder_decoder import IncrementalModule
from nmtg.modules.attention import MultiHeadAttention, AverageAttention
from nmtg.modules.linear import XavierLinear, MaxOut
from nmtg.modules.masking import MaskedFunction


class PrePostProcessing(nn.Module):
    """
    Applies processing to tensors
    Args:
        model_dim:          dimension of model
        dropout:            dropout probability
        elementwise_affine: Passed to LayerNorm
        gated_residuals:    Use gated residuals with a parameter
    sequence of processing steps:
        n = normalization
        d = dropout
        a = adding previous input to output (residual)
    """

    def __init__(self, model_dim, sequence='nda', dropout=0.0,
                 elementwise_affine=True, gated_residuals=False):
        super(PrePostProcessing, self).__init__()
        self.d_model = model_dim
        self.dropout = dropout
        self.gated_residuals = gated_residuals

        self.steps = sequence

        if self.gated_residuals:
            self.k = nn.Parameter(torch.ones(1))

        if 'n' in self.steps:
            layer_norm = nn.LayerNorm([self.d_model], elementwise_affine=elementwise_affine)
            self.layer_norm = MaskedFunction(layer_norm)
        if 'd' in self.steps:
            self.dropout = nn.Dropout(self.dropout, inplace=False)

    def forward(self, tensor, input_tensor=None, mask=None):
        output = tensor
        for step in self.steps:
            if step == 'n':
                output = self.layer_norm(output, mask=mask)
            if step == 'd':
                output = self.dropout(output)
            if step == 'a':
                if input_tensor is not None:
                    if self.gated_residuals:
                        output = F.relu(self.k) * output + input_tensor
                    else:
                        output = output + input_tensor

        return output


def get_feed_forward(feed_forward_type, model_dim, feed_forward_dim, feed_forward_dropout,
                     weight_norm):
    if feed_forward_type == 'linear_relu_linear':
        feed_forward = FeedForward(model_dim, feed_forward_dim, feed_forward_dropout,
                                   weight_norm=weight_norm)
    elif feed_forward_type == 'maxout':
        pool_size = int(math.ceil(feed_forward_dim / model_dim))
        feed_forward = MaxOut(model_dim, model_dim, pool_size)
    else:
        raise ValueError('Unrecognized feed forward type "{}"'.format(feed_forward_type))
    return feed_forward


class FeedForward(nn.Module):
    """
    Applies position-wise feed forward to inputs

    Args:
        model_dim:      dimension of model
        hidden_dim:     dimension of feed forward
        dropout:        dropout probability
        weight_norm:    use weight normalization on the weights

    Params:
        layer_1: FC layer from model_dim to hidden_dim
        layer_2: FC layer from hidden_dim to model_dim

    Input Shapes:
        input: batch_size x len x model_dim

    Output Shapes:
        out: batch_size x len x model_dim
    """

    def __init__(self, model_dim, hidden_dim, dropout, weight_norm=False):
        super().__init__()
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.layer_1 = XavierLinear(model_dim, hidden_dim, weight_norm=weight_norm)
        self.layer_2 = XavierLinear(hidden_dim, model_dim, weight_norm=weight_norm)

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        hidden = F.relu(self.layer_1(inputs), inplace=True)
        hidden = self.dropout(hidden)
        out = self.layer_2(hidden)
        return out


class TransformerEncoderLayer(nn.Module):
    """
    Wraps multi-head attentions and position-wise feed forward into one encoder layer.

    Layers:
        (1)
         Layer norm
         Multi-head self-attention
         Dropout
         Residual with (1)
         (2)
         Layer norm
         Feed-forward
         Dropout
         Residual with (2)

    Feed-Forward:
        Configurable between linear -> ReLU -> linear and Maxout

    Args:
        model_dim:            dimension of model
        num_heads:            number of heads
        feed_forward_dim:     dimension of feed forward
        feed_forward_dropout: dropout probability in the feed forward
        attention_dropout:    dropout probability in attention
        residual_dropout:     dropout probability for the residual layers
        weight_norm:          whether to use weight normalization on the feed forward layers
        masked_layers:        whether to use masking for layer norm and feed forward. Useful for sparse masks
        gated_residuals:      whether to use gated residuals
        batch_first:          whether input (and output) should be batch dimension first or sequence
                              length dimension first
        feed_forward_type:    Which type of feed forward to use. Currently supports 'linear_relu_linear'
                              and 'maxout'

    Params:
        attention:    multi-head self-attentions layer
        feed_forward:  feed forward layer

    Input Shapes:
        inputs:         batch_size x len_query x model_dim  or  len_query x batch_size x model_dim
        input_mask:     batch_size x len_query  or  len_query x batch_size (or broadcastable)
        attention_bias: batch_size x len_query x len_query or broadcastable, regardless of batch_first

    Output Shapes:
        out: batch_size x len_query x model_dim  or  len_query x batch_size x model_dim
    """

    def __init__(self, *, model_dim=512, num_heads=8, feed_forward_dim=2048,
                 feed_forward_dropout=0.1, attention_dropout=0.1, residual_dropout=0.1,
                 weight_norm=False, masked_layers=False, gated_residuals=False, batch_first=False,
                 feed_forward_type='linear_relu_linear'):
        super().__init__()
        self.masked_layers = masked_layers

        self.preprocess_attn = PrePostProcessing(model_dim, 'n')
        self.postprocess_attn = PrePostProcessing(model_dim, 'da', residual_dropout,
                                                  gated_residuals=gated_residuals)
        self.preprocess_ffn = PrePostProcessing(model_dim, 'n')
        self.postprocess_ffn = PrePostProcessing(model_dim, 'da', residual_dropout,
                                                 gated_residuals=gated_residuals)
        self.attention = MultiHeadAttention(model_dim, num_heads, attention_dropout,
                                            masked_layers=masked_layers,
                                            batch_first=batch_first)

        self.feed_forward = MaskedFunction(get_feed_forward(feed_forward_type,
                                                            model_dim,
                                                            feed_forward_dim,
                                                            feed_forward_dropout,
                                                            weight_norm))

    def forward(self, inputs, input_mask=None, attention_bias=None):
        if not self.masked_layers:
            input_mask = None
        # Self-Attention layer
        query = self.preprocess_attn(inputs, mask=input_mask)
        attention_out = self.attention(query, query, query, attention_bias, input_mask)
        attention_out = self.postprocess_attn(attention_out, inputs)

        # Feed-Forward layer
        out = self.preprocess_ffn(attention_out, mask=input_mask)
        out = self.feed_forward(out, mask=input_mask)
        out = self.postprocess_ffn(out, attention_out)
        return out


class TransformerDecoderLayer(IncrementalModule):
    """
    Wraps multi-head self-attention, encoder-decoder attention and position-wise
    feed forward into one layer of decoder

    Layers:
        (1)
         Layer norm
         Multi-head self-attention
         Dropout
         Residual with (1)
         (2)
         Layer norm
         Multi-head query-context attention
         Dropout
         Residual with (2)
         (3)
         Layer norm
         Feed-forward
         Dropout
         Residual with (3)

    Feed-Forward:
        Configurable between linear -> ReLU -> linear and Maxout

    Args:
        model_dim:            dimension of model
        num_heads:            number of heads
        feed_forward_dim:     dimension of feed forward
        feed_forward_dropout: dropout probability in the feed forward
        attention_dropout:    dropout probability in attention
        residual_dropout:     dropout probability for the residual layers
        weight_norm:          whether to use weight normalization on the feed forward layers
        masked_layers:        whether to use masking for layer norm and feed forward. Useful for sparse masks
        gated_residuals:      whether to use gated residuals
        batch_first:          whether input (and output) should be batch dimension first or sequence
                              length dimension first
        feed_forward_type:    Which type of feed forward to use. Currently supports 'linear_relu_linear'
                              and 'maxout'
        ignore_context:       If True, do not use the context input at all
        encoder_to_share:     Instance of TransformerEncoderLayer to share parameters with

    Input Shapes:
        inputs:              len_query x batch_size x model_dim  or  batch_size x len_query x model_dim
        context:             len_context x batch_size x model_dim  or  batch_size x len_context x model_dim
        input_mask:          batch_size x len_query  or  len_query x batch_size
        context_mask:        batch_size x len_context  or  len_context x batch_size
        self_attention_mask: batch_size x len_query x len_query or broadcastable, regardless of batch_first

    Output Shapes:
        out:      len_query x batch_size x model_dim  or  len_query x batch_size x model_dim
    """

    def __init__(self, *, model_dim=512, num_heads=8, feed_forward_dim=2048,
                 feed_forward_dropout=0.1, attention_dropout=0.1, residual_dropout=0.1,
                 weight_norm=False, masked_layers=False, gated_residuals=False, batch_first=False,
                 feed_forward_type='linear_relu_linear',
                 ignore_context=False, encoder_to_share=None):
        super().__init__()
        self.ignore_context = ignore_context
        self.masked_layers = masked_layers

        if encoder_to_share is None:
            self.preprocess_attn = PrePostProcessing(model_dim, 'n')
            self.postprocess_attn = PrePostProcessing(model_dim, 'da', residual_dropout,
                                                      gated_residuals=gated_residuals)

            self.preprocess_ffn = PrePostProcessing(model_dim, 'n')
            self.postprocess_ffn = PrePostProcessing(model_dim, 'da', residual_dropout,
                                                     gated_residuals=gated_residuals)

            self.attention_tgt = MultiHeadAttention(model_dim, num_heads, attention_dropout,
                                                    masked_layers=masked_layers,
                                                    batch_first=batch_first)

            self.feed_forward = MaskedFunction(get_feed_forward(feed_forward_type,
                                                                model_dim,
                                                                feed_forward_dim,
                                                                feed_forward_dropout,
                                                                weight_norm))

        else:
            # share the self-attention layers between encoder and decoder

            self.preprocess_attn = encoder_to_share.preprocess_attn
            self.postprocess_attn = encoder_to_share.postprocess_attn

            self.preprocess_ffn = encoder_to_share.preprocess_ffn
            self.postprocess_ffn = encoder_to_share.postprocess_ffn

            self.attention_tgt = encoder_to_share.attention
            self.feed_forward = encoder_to_share.feed_forward

        if not ignore_context:
            self.preprocess_src_attn = PrePostProcessing(model_dim, 'n')
            self.postprocess_src_attn = PrePostProcessing(model_dim, 'da', residual_dropout,
                                                          gated_residuals=gated_residuals)
            self.attention_src = MultiHeadAttention(model_dim, num_heads, attention_dropout,
                                                    masked_layers=masked_layers,
                                                    batch_first=batch_first)

    def forward(self, inputs, context, input_mask=None, context_mask=None, self_attention_bias=None,
                encoder_attention_bias=None):
        if not self.masked_layers:
            input_mask = None
            context_mask = None

        # Self-Attention layer
        query = self.preprocess_attn(inputs, mask=input_mask)
        self_attention_out = self.attention_tgt(query, query, query, self_attention_bias, input_mask)
        self_attention_out = self.postprocess_attn(self_attention_out, inputs)

        # Context-To-Query-Attention layer
        if not self.ignore_context:
            query = self.preprocess_src_attn(self_attention_out, mask=input_mask)
            src_attention_out = self.attention_src(query, context, context, encoder_attention_bias,
                                                   input_mask, context_mask)
            src_attention_out = self.postprocess_src_attn(src_attention_out, self_attention_out)
        else:
            src_attention_out = self_attention_out

        # Feed-Forward layer
        out = self.preprocess_ffn(src_attention_out, mask=input_mask)
        out = self.feed_forward(out, input_mask)
        out = self.postprocess_ffn(out, src_attention_out)
        return out

    def _step(self, inputs, context, incremental_state, input_mask=None, context_mask=None, self_attention_bias=None,
              encoder_attention_bias=None):

        if not self.masked_layers:
            input_mask = None
            context_mask = None

        # Self-Attention layer
        query = self.preprocess_attn(inputs, mask=input_mask)
        self_attention_out = self.attention_tgt.step(query, query, query, incremental_state,
                                                     self_attention_bias, input_mask)
        self_attention_out = self.postprocess_attn(self_attention_out, inputs)

        # Context-To-Query-Attention layer
        if not self.ignore_context:
            query = self.preprocess_src_attn(self_attention_out, mask=input_mask)
            src_attention_out = self.attention_src.step(query, context, context, incremental_state,
                                                        encoder_attention_bias,
                                                        input_mask, context_mask,
                                                        static_kv=True)
            src_attention_out = self.postprocess_src_attn(src_attention_out, self_attention_out)
        else:
            src_attention_out = self_attention_out

        # Feed-Forward layer
        out = self.preprocess_ffn(src_attention_out, mask=input_mask)
        out = self.feed_forward(out, mask=input_mask)
        out = self.postprocess_ffn(out, src_attention_out)
        return out


class AverageTransformerDecoderLayer(IncrementalModule):
    def __init__(self, *, model_dim=512, num_heads=8, feed_forward_dim=2048,
                 feed_forward_dropout=0.1, attention_dropout=0.1, residual_dropout=0.1,
                 weight_norm=False, masked_layers=False, gated_residuals=False, batch_first=False,
                 feed_forward_type='linear_relu_linear',
                 ignore_context=False, encoder_to_share=None):
        super().__init__()
        self.ignore_context = ignore_context
        self.masked_layers = masked_layers

        if encoder_to_share is None:
            self.preprocess_attn = PrePostProcessing(model_dim, 'n')
            self.postprocess_attn = PrePostProcessing(model_dim, 'da', residual_dropout,
                                                      gated_residuals=gated_residuals)

            self.preprocess_ffn = PrePostProcessing(model_dim, 'n')
            self.postprocess_ffn = PrePostProcessing(model_dim, 'da', residual_dropout,
                                                     gated_residuals=gated_residuals)

            self.attention_tgt = AverageAttention(model_dim, attention_dropout,
                                                  get_feed_forward(feed_forward_type,
                                                                   model_dim,
                                                                   feed_forward_dim,
                                                                   feed_forward_dropout,
                                                                   weight_norm),
                                                  batch_first=batch_first,
                                                  masked_layers=masked_layers)

            self.feed_forward = MaskedFunction(get_feed_forward(feed_forward_type,
                                                                model_dim,
                                                                feed_forward_dim,
                                                                feed_forward_dropout,
                                                                weight_norm))

        else:
            # share the self-attention layers between encoder and decoder

            self.preprocess_attn = encoder_to_share.preprocess_attn
            self.postprocess_attn = encoder_to_share.postprocess_attn

            self.preprocess_ffn = encoder_to_share.preprocess_ffn
            self.postprocess_ffn = encoder_to_share.postprocess_ffn

            self.feed_forward = encoder_to_share.feed_forward

        if not ignore_context:
            self.preprocess_src_attn = PrePostProcessing(model_dim, 'n')
            self.postprocess_src_attn = PrePostProcessing(model_dim, 'da', residual_dropout,
                                                          gated_residuals=gated_residuals)
            self.attention_src = MultiHeadAttention(model_dim, num_heads, attention_dropout,
                                                    masked_layers=masked_layers,
                                                    batch_first=batch_first)

    def forward(self, inputs, context, input_mask=None, context_mask=None, self_attention_bias=None,
                encoder_attention_bias=None):
        if not self.masked_layers:
            input_mask = None
            context_mask = None

        # Self-Attention layer
        query = self.preprocess_attn(inputs, mask=input_mask)
        self_attention_out = self.attention_tgt(query, self_attention_bias)
        self_attention_out = self.postprocess_attn(self_attention_out, inputs)

        # Context-To-Query-Attention layer
        if not self.ignore_context:
            query = self.preprocess_src_attn(self_attention_out, mask=input_mask)
            src_attention_out = self.attention_src(query, context, context, encoder_attention_bias,
                                                   input_mask, context_mask)
            src_attention_out = self.postprocess_src_attn(src_attention_out, self_attention_out)
        else:
            src_attention_out = self_attention_out

        # Feed-Forward layer
        out = self.preprocess_ffn(src_attention_out, mask=input_mask)
        out = self.feed_forward(out, input_mask)
        out = self.postprocess_ffn(out, src_attention_out)
        return out

    def _step(self, inputs, context, incremental_state, input_mask=None, context_mask=None,
              self_attention_bias=None, encoder_attention_bias=None):
        # Self-attention mask is here for compatibility, it is not used

        padding_mask = input_mask
        if not self.masked_layers:
            input_mask = None
            context_mask = None

        # Self-Attention layer
        query = self.preprocess_attn(inputs, mask=input_mask)
        self_attention_out = self.attention_tgt.step(query, incremental_state, padding_mask)
        self_attention_out = self.postprocess_attn(self_attention_out, inputs)

        # Context-To-Query-Attention layer
        if not self.ignore_context:
            query = self.preprocess_src_attn(self_attention_out, mask=input_mask)
            src_attention_out = self.attention_src.step(query, context, context, incremental_state,
                                                        encoder_attention_bias,
                                                        input_mask, context_mask,
                                                        static_kv=True)
            src_attention_out = self.postprocess_src_attn(src_attention_out, self_attention_out)
        else:
            src_attention_out = self_attention_out

        # Feed-Forward layer
        out = self.preprocess_ffn(src_attention_out, mask=input_mask)
        out = self.feed_forward(out, mask=input_mask)
        out = self.postprocess_ffn(out, src_attention_out)
        return out
