import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.utils.weight_norm as WeightNorm
import onmt

from onmt.modules.Bottle import Bottle
from onmt.modules.StaticDropout import StaticDropout
from onmt.modules.Linear import XavierLinear, group_linear, FeedForward
from onmt.modules.MaxOut import MaxOut
from onmt.modules.GlobalAttention import MultiHeadAttention
from onmt.modules.PrePostProcessing import PrePostProcessing


class VariationalDecoderLayer(nn.Module):
    """Wraps multi-head attentions and position-wise feed forward into one layer of decoder

    Args:
        h:       number of heads
        d_model: dimension of model
        p:       dropout probabolity
        d_ff:    dimension of feed forward

    Params:
        multihead_tgt:  multi-head self attentions layer
        multihead_src:  multi-head encoder-decoder attentions layer
        feedforward:    feed forward layer

    Input Shapes:
        query:    batch_size x len_query x d_model
        key:      batch_size x len_key x d_model
        value:    batch_size x len_key x d_model
        context:  batch_size x len_src x d_model
        mask_tgt: batch_size x len_query x len_key or broadcastable
        mask_src: batch_size x len_query x len_src or broadcastable

    Output Shapes:
        out:      batch_size x len_query x d_model
        coverage: batch_size x len_query x len_key

    """

    def __init__(self, h, d_model, p, d_ff, attn_p=0.1, residual_p=0.1, version=1.0,
                 ignore_source=False, use_latent=False, d_latent=32, encoder_to_share=None):
        super().__init__()
        self.ignore_source = ignore_source

        if encoder_to_share is None:

            self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
            self.postprocess_attn = PrePostProcessing(d_model, residual_p, sequence='da', static=onmt.Constants.static)

            self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
            self.postprocess_ffn = PrePostProcessing(d_model, residual_p, sequence='da', static=onmt.Constants.static)

            self.multihead_tgt = MultiHeadAttention(h, d_model, attn_p=attn_p, static=onmt.Constants.static, share=1)

            if onmt.Constants.activation_layer == 'linear_relu_linear':
                ff_p = p
                feedforward = FeedForward(d_model, d_ff, ff_p, static=onmt.Constants.static)
            elif onmt.Constants.activation_layer == 'maxout':
                k = int(math.ceil(d_ff / d_model))
                feedforward = MaxOut(d_model, d_model, k)
            else:
                raise NotImplementedError
            self.feedforward = Bottle(feedforward)

        else:
            # share the self-attention layers between encoder and decoder

            self.preprocess_attn = encoder_to_share.preprocess_attn
            self.postprocess_attn = encoder_to_share.postprocess_attn

            self.preprocess_ffn = encoder_to_share.preprocess_ffn
            self.postprocess_ffn = encoder_to_share.postprocess_ffn

            self.multihead_tgt = encoder_to_share.multihead
            self.feedforward = encoder_to_share.feedforward

        if self.ignore_source == False:
            self.preprocess_src_attn = PrePostProcessing(d_model, p, sequence='n')
            self.postprocess_src_attn = PrePostProcessing(d_model, residual_p, sequence='da',
                                                          static=onmt.Constants.static)
            self.multihead_src = MultiHeadAttention(h, d_model, attn_p=attn_p, static=onmt.Constants.static, share=2)

        self.z_gate = XavierLinear(d_model * 2, d_model)
        self.i_gate = XavierLinear(d_model * 2, d_model)
        self.preprocess_latent = PrePostProcessing(d_model, p, sequence='n')
        # if use_latent == True:
        # self.z_transform = Linear(d_latent, d_model)
        # self.preprocess_latent = PrePostProcessing(d_model, p, sequence='n')
        # self.postprocess_latent = PrePostProcessing(d_model, residual_p, sequence='da', static=onmt.Constants.static)

    def forward(self, input, context, latent_z, mask_tgt, mask_src, pad_mask_tgt=None, pad_mask_src=None,
                residual_dropout=0.0):

        """ Self attention layer
            layernorm > attn > dropout > residual
        """

        # input and context should be time first ?

        query = self.preprocess_attn(input)

        self_context = query

        out, _ = self.multihead_tgt(query, self_context, self_context, mask_tgt)

        input = self.postprocess_attn(out, input)

        """ Context Attention layer 
            layernorm > attn > dropout > residual
        """
        if self.ignore_source == False:
            query = self.preprocess_src_attn(input)
            # note: "out" represents the weight sum of the source representation
            out, coverage = self.multihead_src(query, context, context, mask_src)
            input = self.postprocess_src_attn(out, input)
        else:
            coverage = None

        # gated linear combination
        input_norm = self.preprocess_latent(input)
        input_plus_z = torch.cat([input_norm, latent_z.expand_as(input_norm)], dim=-1)

        z_gate = torch.sigmoid(self.z_gate(input_plus_z))
        i_gate = torch.sigmoid(self.i_gate(input_plus_z))

        output = latent_z * z_gate + i_gate * input_norm

        # residual
        input = input + output

        # use residual combination for input and z
        # if latent_z is not None:
        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input))
        input = self.postprocess_ffn(out, input)

        return input, coverage

    def step(self, input, context, mask_tgt, mask_src, pad_mask_tgt=None, pad_mask_src=None, latent_z=None,
             buffer=None):
        """ Self attention layer
            layernorm > attn > dropout > residual
        """

        query = self.preprocess_attn(input)

        out, _, buffer = self.multihead_tgt.step(query, query, query, mask_tgt, buffer=buffer)

        input = self.postprocess_attn(out, input)

        """ Context Attention layer 
            layernorm > attn > dropout > residual
        """

        if self.ignore_source == False:
            query = self.preprocess_src_attn(input)
            out, coverage, buffer = self.multihead_src.step(query, context, context, mask_src, buffer=buffer)
            input = self.postprocess_src_attn(out, input)
        else:
            batch_size = query.size(1)
            length_tgt = query.size(0)
            length_src = mask_src.size(-1)
            coverage = input.new(batch_size, length_tgt, length_src).zero_()

        # use residual combination for input and z
        if latent_z is not None:
            # input = latent_z + input
            input_norm = self.preprocess_latent(input)
            input_plus_z = torch.cat([input_norm, latent_z], dim=-1)

            z_gate = torch.sigmoid(self.z_gate(input_plus_z))
            i_gate = torch.sigmoid(self.i_gate(input_plus_z))

            output = latent_z * z_gate + i_gate * input_norm

            # residual
            input = input + output
            # z_gate = torch.sigmoid(self.z_gate(input_plus_z))

        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input))
        input = self.postprocess_ffn(out, input)

        return input, coverage, buffer


