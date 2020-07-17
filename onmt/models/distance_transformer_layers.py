import torch
import torch.nn as nn
import onmt

from onmt.models.transformer_layers import PrePostProcessing, MultiHeadAttention, Linear
from onmt.modules.relative_attention import RelPartialLearnableMultiHeadAttn
from onmt.utils import flip
from onmt.modules.bottle import Bottle
from onmt.modules.linear import XavierLinear as Linear
from onmt.modules.linear import XavierLinear
from onmt.modules.linear import group_linear, FeedForwardSwish, FeedForward
from onmt.modules.attention import MultiHeadAttention
from onmt.modules.dropout import VariationalDropout
from onmt.modules.relative_attention import LearnableRelMultiHeadAttn


class DistanceTransformerEncoderLayer(nn.Module):
    def __init__(self, h, d_model, p, d_ff, attn_p=0.1, variational=False, death_rate=0.0,
                 max_len=64, **kwargs):
        super(DistanceTransformerEncoderLayer, self).__init__()
        self.variational = variational
        self.death_rate = death_rate

        self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model, p, sequence='da', variational=self.variational)
        self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, p, sequence='da', variational=self.variational)
        # self.multihead = MultiHeadAttention(h, d_model, attn_p=attn_p, share=2)
        d_head = d_model // h
        self.multihead = LearnableRelMultiHeadAttn(h, d_model, d_head, dropatt=attn_p, max_len=max_len)

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

    def forward(self, input, attn_mask, incremental=False, incremental_cache=None, mems=None):

        coin = True
        if self.training and self.death_rate > 0:
            coin = (torch.rand(1)[0].item() >= self.death_rate)

        if coin:

            if mems is not None and mems.size(0) > 0:
                mems = self.preprocess_attn(mems)
            else:
                mems = None

            query = self.preprocess_attn(input)
            out, _, incremental_cache = self.multihead(query, attn_mask=attn_mask, mems=mems,
                                                       incremental=incremental, incremental_cache=incremental_cache)

            # rescaling before residual
            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_attn(out, input)

            """ Feed forward layer 
                layernorm > ffn > dropout > residual
            """
            out = self.feedforward(self.preprocess_ffn(input))

            # rescaling before residual
            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_ffn(out, input)

        if incremental:
            return input, incremental_cache

        return input


class DistanceTransformerDecoderLayer(nn.Module):

    def __init__(self, h, d_model, p, d_ff, attn_p=0.1, version=1.0, ignore_source=False,
                 variational=False, death_rate=0.0, max_len=64):
        super(DistanceTransformerDecoderLayer, self).__init__()
        self.version = version
        self.ignore_source = ignore_source
        self.variational = variational
        self.death_rate = death_rate

        self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model, p, sequence='da', variational=self.variational)

        if not self.ignore_source:
            self.preprocess_src_attn = PrePostProcessing(d_model, p, sequence='n')
            self.postprocess_src_attn = PrePostProcessing(d_model, p, sequence='da', variational=self.variational)
            self.multihead_src = MultiHeadAttention(h, d_model, attn_p=attn_p, share=2)

        self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, p, sequence='da', variational=self.variational)

        d_head = d_model // h
        self.multihead_tgt = LearnableRelMultiHeadAttn(h, d_model, d_head, dropatt=attn_p, max_len=64)
        # self.multihead_tgt = MultiHeadAttention(h, d_model, attn_p=attn_p, share=1)

        if onmt.constants.activation_layer == 'linear_relu_linear':
            ff_p = p
            feedforward = FeedForward(d_model, d_ff, ff_p, variational=self.variational)
        elif onmt.constants.activation_layer == 'maxout':
            k = int(math.ceil(d_ff / d_model))
            feedforward = MaxOut(d_model, d_model, k)
        elif onmt.constants.activation_layer == 'linear_swish_linear':
            ff_p = p
            feedforward = FeedForwardSwish(d_model, d_ff, ff_p)
        else:
            raise NotImplementedError
        self.feedforward = Bottle(feedforward)

    # def forward(self, input, context, pos_emb, r_w_bias, r_r_bias, mask_tgt, mask_src):
    def forward(self, input, context, mask_tgt, mask_src,
                incremental=False, incremental_cache=None, reuse_source=True, mems=None):

        """ Self attention layer
            layernorm > attn > dropout > residual
        """
        if incremental and incremental_cache is None:
            incremental_cache = dict()

        coin = True
        if self.training and self.death_rate > 0:
            coin = (torch.rand(1)[0].item() >= self.death_rate)

        if coin:
            # input and context should be time first ?
            if mems is not None and mems.size(0) > 0:
                mems = self.preprocess_attn(mems)
            else:
                mems = None

            query = self.preprocess_attn(input)

            # out, _ = self.multihead_tgt(query, pos_emb, r_w_bias, r_r_bias, attn_mask=mask_tgt)
            # print(query.size(), pos_emb.size(), mask_tgt.size(), mems.size() if mems is not None else 0)
            out, _, = self.multihead_tgt(query, attn_mask=mask_tgt, mems=mems,
                                         incremental=incremental, incremental_cache=incremental_cache)

            # rescaling before residual
            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_attn(out, input)

            """ Context Attention layer 
                layernorm > attn > dropout > residual
            """
            if not self.ignore_source:
                query = self.preprocess_src_attn(input)
                incremental_source = incremental and reuse_source
                out, coverage = self.multihead_src(query, context, context, mask_src,
                                                   incremental=incremental_source,
                                                   incremental_cache=incremental_cache)

                # rescaling before residual
                if self.training and self.death_rate > 0:
                    out = out / (1 - self.death_rate)

                input = self.postprocess_src_attn(out, input)
            else:
                coverage = None

            """ Feed forward layer 
                layernorm > ffn > dropout > residual
            """
            out = self.feedforward(self.preprocess_ffn(input))

            # rescaling before residual
            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_ffn(out, input)
        else:
            coverage = None

        return input, coverage, incremental_cache

    def step(self, input, context, mask_tgt, mask_src, buffer=None):
        """ Self attention layer
            layernorm > attn > dropout > residual
        """

        query = self.preprocess_attn(input)

        out, _, buffer = self.multihead_tgt.step(query, attn_mask=mask_tgt, buffer=buffer)

        input = self.postprocess_attn(out, input)

        """ Context Attention layer
            layernorm > attn > dropout > residual
        """
        if not self.ignore_source:
            query = self.preprocess_src_attn(input)
            out, coverage, buffer = self.multihead_src.step(query, context, context, mask_src, buffer=buffer)
            input = self.postprocess_src_attn(out, input)
        else:
            coverage = None

        """ Feed forward layer
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input))
        input = self.postprocess_ffn(out, input)

        return input, coverage, buffer
