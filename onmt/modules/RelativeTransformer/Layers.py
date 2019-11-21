import torch
import torch.nn as nn
import onmt

from onmt.modules.Transformer.Layers import PrePostProcessing, MultiHeadAttention, FeedForward, Linear
from onmt.modules.RelativeAttention import RelPartialLearnableMultiHeadAttn
from onmt.utils import flip
from onmt.modules.Bottle import Bottle
from onmt.modules.Linear import XavierLinear as Linear
from onmt.modules.Linear import XavierLinear
from onmt.modules.Linear import group_linear, FeedForwardSwish
from onmt.modules.GlobalAttention import MultiHeadAttention
from onmt.modules.WordDrop import VariationalDropout


#  Positional Embedding with discrete inputs
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(SinusoidalPositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


# This is the only component in the Translation model
class RelativeTransformerDecoderLayer(nn.Module):

    def __init__(self, h, d_model, p, d_ff, attn_p=0.1, variational=False):
        super(RelativeTransformerDecoderLayer, self).__init__()
        self.variational = variational
        self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model, p, sequence='da', variational=self.variational)
        self.variational = variational

        self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, p, sequence='da', variational=self.variational)

        d_head = d_model // h
        self.multihead_tgt = RelPartialLearnableMultiHeadAttn(h, d_model, d_head, dropatt=attn_p)

        if onmt.Constants.activation_layer == 'linear_relu_linear':
            ff_p = p
            feedforward = FeedForward(d_model, d_ff, ff_p, variational=self.variational)
        elif onmt.Constants.activation_layer == 'maxout':
            k = int(math.ceil(d_ff / d_model))
            feedforward = MaxOut(d_model, d_model, k)
        elif onmt.Constants.activation_layer == 'linear_swish_linear':
            ff_p = p
            feedforward = FeedForwardSwish(d_model, d_ff, ff_p, variational=self.variational)
        else:
            raise NotImplementedError
        self.feedforward = Bottle(feedforward)

    def forward(self, input, pos, r_w_bias, r_r_bias, mask, mems=None):

        """
        :param mems: The hidden layers from the previous segments
        :param mask: The attention mask to avoid padding
        :param input: Embedding (from the last layer) T x B x H
        :param pos: Positional Encoding T x B x H
        :return:
        """

        # input and context should be time first ?

        if mems is not None:
            cat = torch.cat([mems, input], 0)
            query = self.preprocess_attn(cat)
        else:
            query = self.preprocess_attn(input)

        out, coverage = self.multihead_tgt(query, pos, r_w_bias, r_r_bias, mask)

        # dropout + residual
        input = self.postprocess_attn(out, input)

        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input))
        input = self.postprocess_ffn(out, input)

        return input, coverage

    # def step(self, input, pos, context, mask_tgt, mask_src, buffer=None):
    #     """ Self attention layer
    #         layernorm > attn > dropout > residual
    #     """
    #
    #     query = self.preprocess_attn(input)
    #
    #     out, _, buffer = self.multihead_tgt.step(query, pos, mask_tgt, buffer=buffer)
    #
    #     input = self.postprocess_attn(out, input)
    #
    #     """ Context Attention layer
    #         layernorm > attn > dropout > residual
    #     """
    #     if not self.ignore_source:
    #         query = self.preprocess_src_attn(input)
    #         out, coverage, buffer = self.multihead_src.step(query, context, context, mask_src, buffer=buffer)
    #         input = self.postprocess_src_attn(out, input)
    #     else:
    #         coverage = None
    #
    #     """ Feed forward layer
    #         layernorm > ffn > dropout > residual
    #     """
    #     out = self.feedforward(self.preprocess_ffn(input))
    #     input = self.postprocess_ffn(out, input)
    #
    #     return input, coverage, buffer


# class RelativeEncoderLayer(EncoderLayer):
