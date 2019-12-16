import torch
import torch.nn as nn
import onmt

from onmt.models.transformer_layers import PrePostProcessing, MultiHeadAttention, FeedForward, Linear
from onmt.modules.relative_attention import RelPartialLearnableMultiHeadAttn
from onmt.utils import flip
from onmt.modules.bottle import Bottle
from onmt.modules.linear import XavierLinear as Linear
from onmt.modules.linear import XavierLinear
from onmt.modules.linear import group_linear, FeedForwardSwish
from onmt.modules.attention import MultiHeadAttention
from onmt.modules.dropout import VariationalDropout
from onmt.modules.relative_attention import RelPartialLearnableMultiHeadAttn


class RelativeTransformerEncoderLayer(nn.Module):
    def __init__(self, h, d_model, p, d_ff, attn_p=0.1, variational=False, **kwargs):
        super(RelativeTransformerEncoderLayer, self).__init__()
        self.variational = variational

        self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model, p, sequence='da', variational=self.variational)
        self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, p, sequence='da', variational=self.variational)
        # self.multihead = MultiHeadAttention(h, d_model, attn_p=attn_p, share=2)
        d_head = d_model // h
        self.multihead = RelPartialLearnableMultiHeadAttn(h, d_model, d_head, dropatt=attn_p)

        if onmt.constants.activation_layer == 'linear_relu_linear':
            ff_p = p
            feedforward = FeedForward(d_model, d_ff, ff_p, variational=self.variational)
        elif onmt.constants.activation_layer == 'maxout':
            k = int(math.ceil(d_ff / d_model))
            feedforward = MaxOut(d_model, d_model, k)
        elif onmt.constants.activation_layer == 'linear_swish_linear':
            ff_p = p
            feedforward = FeedForwardSwish(d_model, d_ff, ff_p, variational=self.variational)
        else:
            raise NotImplementedError
        self.feedforward = Bottle(feedforward)

    def forward(self, input, pos_emb, r_w_bias, r_r_bias, attn_mask):
        query = self.preprocess_attn(input)
        out, _ = self.multihead(query, pos_emb, r_w_bias, r_r_bias, attn_mask=attn_mask)
        input = self.postprocess_attn(out, input)

        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input))
        input = self.postprocess_ffn(out, input)

        return input

# # This is the only component in the Translation model
# class RelativeTransformerDecoderLayer(nn.Module):
#
#     def __init__(self, h, d_model, p, d_ff, attn_p=0.1, variational=False):
#         super(RelativeTransformerDecoderLayer, self).__init__()
#         self.variational = variational
#         self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
#         self.postprocess_attn = PrePostProcessing(d_model, p, sequence='da', variational=self.variational)
#         self.variational = variational
#
#         self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
#         self.postprocess_ffn = PrePostProcessing(d_model, p, sequence='da', variational=self.variational)
#
#         d_head = d_model // h
#         self.multihead_tgt = RelPartialLearnableMultiHeadAttn(h, d_model, d_head, dropatt=attn_p)
#
#         if onmt.constants.activation_layer == 'linear_relu_linear':
#             ff_p = p
#             feedforward = FeedForward(d_model, d_ff, ff_p, variational=self.variational)
#         elif onmt.constants.activation_layer == 'maxout':
#             k = int(math.ceil(d_ff / d_model))
#             feedforward = MaxOut(d_model, d_model, k)
#         elif onmt.constants.activation_layer == 'linear_swish_linear':
#             ff_p = p
#             feedforward = FeedForwardSwish(d_model, d_ff, ff_p, variational=self.variational)
#         else:
#             raise NotImplementedError
#         self.feedforward = Bottle(feedforward)
#
#     def forward(self, input, pos, r_w_bias, r_r_bias, mask, mems=None):
#
#         """
#         :param mems: The hidden layers from the previous segments
#         :param mask: The attention mask to avoid padding
#         :param input: Embedding (from the last layer) T x B x H
#         :param pos: Positional Encoding T x B x H
#         :return:
#         """
#
#         # input and context should be time first ?
#
#         if mems is not None:
#             cat = torch.cat([mems, input], 0)
#             query = self.preprocess_attn(cat)
#         else:
#             query = self.preprocess_attn(input)
#
#         out, coverage = self.multihead_tgt(query, pos, r_w_bias, r_r_bias, mask)
#
#         # dropout + residual
#         input = self.postprocess_attn(out, input)
#
#         """ Feed forward layer
#             layernorm > ffn > dropout > residual
#         """
#         out = self.feedforward(self.preprocess_ffn(input))
#         input = self.postprocess_ffn(out, input)
#
#         return input, coverage

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
