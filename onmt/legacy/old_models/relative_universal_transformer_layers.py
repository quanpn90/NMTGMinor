import torch
import torch.nn as nn
import onmt

from onmt.models.transformer_layers import PrePostProcessing, MultiHeadAttention, Linear
from onmt.modules.relative_attention import RelPartialLearnableMultiHeadAttn
from onmt.modules.optimized.relative_self_attention import RelativeSelfMultiheadAttn
from onmt.utils import flip
from onmt.modules.bottle import Bottle
from onmt.modules.linear import XavierLinear as Linear
from onmt.modules.linear import XavierLinear
from onmt.modules.linear import group_linear, FeedForwardSwish, FeedForward
from onmt.modules.attention import MultiHeadAttention
from onmt.modules.dropout import VariationalDropout
from onmt.modules.relative_attention import RelPartialLearnableMultiHeadAttn
from onmt.modules.optimized.encdec_attention import EncdecMultiheadAttn
from onmt.modules.optimized.feed_forward import PositionWiseFeedForward
from onmt.modules.adaptive.relative_self_attention import AdaptiveRelativeAttn
from onmt.modules.adaptive.encdec_attention import AdaptiveEncDecAttn
from onmt.modules.adaptive.feed_forward import AdaptiveFeedForward


class RelativeUniversalEncoderLayer(nn.Module):
    def __init__(self, opt, death_rate=0.0, **kwargs):
        super().__init__()
        self.variational = opt.variational_dropout
        self.death_rate = death_rate
        self.fast_self_attention = opt.fast_self_attention

        self.preprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                  variational=self.variational)
        self.preprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                 variational=self.variational)
        d_head = opt.model_size // opt.n_heads
        self.adaptive_type = opt.adaptive
        self.factor_size = opt.layers

        # this model defaults as fast relative self attention
        if self.adaptive_type == 'universal':
            self.multihead = RelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout)

            self.feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                       variational=self.variational)
        else:
            self.multihead = AdaptiveRelativeAttn(opt.model_size, opt.n_heads, self.factor_size, opt.attn_dropout)
            self.feedforward = AdaptiveFeedForward(opt.model_size, opt.inner_size, self.factor_size,
                                                   opt.dropout, variational=self.variational)

    def forward(self, input, pos_emb, layer_vector, attn_mask, incremental=False, incremental_cache=None, mems=None):

        if self.adaptive_type == 'universal':
            input = input + layer_vector

        if incremental and incremental_cache is None:
            incremental_cache = dict()

        coin = True
        # if self.training and self.death_rate > 0:
        #     coin = (torch.rand(1)[0].item() >= self.death_rate)

        if coin:

            if mems is not None and mems.size(0) > 0:
                mems = self.preprocess_attn(mems)
            else:
                mems = None

            query = self.preprocess_attn(input)
            if self.adaptive_type == 'universal':
                out, _ = self.multihead(query, pos_emb, attn_mask, None, mems=mems,
                                        incremental=incremental, incremental_cache=incremental_cache)
            else:
                out, _ = self.multihead(query, pos_emb, layer_vector, attn_mask, None, mems=mems,
                                        incremental=incremental, incremental_cache=incremental_cache)

            # rescaling before residual
            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_attn(out, input)

            """ Feed forward layer 
                layernorm > ffn > dropout > residual
            """
            if self.adaptive_type == 'universal':
                out = self.feedforward(self.preprocess_ffn(input))
            else:
                out = self.feedforward(self.preprocess_ffn(input), layer_vector)

            # rescaling before residual
            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_ffn(out, input)

        if incremental:
            return input, incremental_cache

        return input


class RelativeUniversalDecoderLayer(nn.Module):

    def __init__(self, opt, death_rate=0.0):
        super().__init__()
        self.ignore_source = opt.ignore_source
        self.variational = opt.variational_dropout
        self.death_rate = death_rate
        self.fast_self_attention = opt.fast_self_attention
        self.factor_size = opt.layers
        self.adaptive_type = opt.adaptive

        self.preprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                  variational=self.variational)

        if not self.ignore_source:
            self.preprocess_src_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
            self.postprocess_src_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                          variational=self.variational)

            if self.adaptive_type == 'universal':
                self.multihead_src = EncdecMultiheadAttn(opt.n_heads, opt.model_size, opt.attn_dropout)
            else:
                self.multihead_src = AdaptiveEncDecAttn(opt.n_heads, opt.model_size, self.factor_size, opt.attn_dropout)

        self.preprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                 variational=self.variational)

        if self.adaptive_type == 'universal':
            self.multihead_tgt = RelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout)

            self.feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                  variational=self.variational)
        else:
            self.multihead_tgt = AdaptiveRelativeAttn(opt.model_size, opt.n_heads, self.factor_size,
                                                      opt.attn_dropout)
            self.feedforward = AdaptiveFeedForward(opt.model_size, opt.inner_size, self.factor_size,
                                                   opt.dropout, variational=self.variational)


    # def forward(self, input, context, pos_emb, r_w_bias, r_r_bias, mask_tgt, mask_src):
    def forward(self, input, context, pos_emb, layer_vector, mask_tgt, mask_src,
                incremental=False, incremental_cache=None, reuse_source=True, mems=None):

        # sum up input with the layer embedding
        if self.adaptive_type == 'universal':
            input = input + layer_vector

        if incremental and incremental_cache is None:
            incremental_cache = dict()

        coin = True

        if coin:
            # input and context should be time first ?
            if mems is not None and mems.size(0) > 0:
                mems = self.preprocess_attn(mems)
            else:
                mems = None

            query = self.preprocess_attn(input)

            if self.adaptive_type == 'universal':
                out, _ = self.multihead_tgt(query, pos_emb, None, mask_tgt, mems=mems,
                                            incremental=incremental, incremental_cache=incremental_cache)
            else:
                out, _ = self.multihead_tgt(query, pos_emb, layer_vector, None, mask_tgt, mems=mems,
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
                if self.adaptive_type == 'universal':
                    out, coverage = self.multihead_src(query, context, context, mask_src,
                                                       incremental=incremental_source,
                                                       incremental_cache=incremental_cache)
                else:
                    out, coverage = self.multihead_src(query, context, context, layer_vector, mask_src,
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
            if self.adaptive_type == 'universal':
                out = self.feedforward(self.preprocess_ffn(input))
            else:
                out = self.feedforward(self.preprocess_ffn(input), layer_vector)

            # rescaling before residual
            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_ffn(out, input)
        else:
            coverage = None

        return input, coverage, incremental_cache
