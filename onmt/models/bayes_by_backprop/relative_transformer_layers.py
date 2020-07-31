import torch
import torch.nn as nn
import onmt

from onmt.models.transformer_layers import PrePostProcessing, MultiHeadAttention, Linear
from onmt.utils import flip
from onmt.modules.linear import XavierLinear as Linear
from onmt.modules.linear import XavierLinear
from onmt.modules.attention import MultiHeadAttention
from onmt.modules.dropout import VariationalDropout
from onmt.modules.bayes_by_backprop.encdec_attention import EncdecMultiheadAttn
from onmt.modules.bayes_by_backprop.feed_forward import PositionWiseFeedForward
from onmt.modules.bayes_by_backprop.relative_self_attention import RelativeSelfMultiheadAttn


class TransformerEncoderLayer(nn.Module):
    # def __init__(self, h, d_model, p, d_ff, attn_p=0.1, variational=False, death_rate=0.0, **kwargs):
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

        self.multihead = RelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout)
        self.feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                   variational=self.variational)

    def log_prior(self):

        log_prior = 0
        log_prior += self.multihead.log_prior
        self.multihead.log_prior = 0
        log_prior += self.feedforward.log_prior
        self.feedforward.log_prior = 0
        return log_prior

    def log_variational_posterior(self):

        log_variational_posterior = 0
        log_variational_posterior += self.multihead.log_variational_posterior
        self.multihead.log_variational_posterior = 0
        log_variational_posterior += self.feedforward.log_variational_posterior
        self.feedforward.log_variational_posterior = 0
        return log_variational_posterior

    def forward(self, input, pos_emb, attn_mask, incremental=False, incremental_cache=None, mems=None):

        if incremental and incremental_cache is None:
            incremental_cache = dict()

        coin = True
        if self.training and self.death_rate > 0:
            coin = (torch.rand(1)[0].item() >= self.death_rate)

        if coin:

            if mems is not None and mems.size(0) > 0:
                mems = self.preprocess_attn(mems)
            else:
                mems = None

            query = self.preprocess_attn(input)
            out, _ = self.multihead(query, pos_emb, attn_mask, None, mems=mems,
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


class TransformerDecoderLayer(nn.Module):
    def __init__(self, opt, death_rate=0.0):
        super().__init__()
        self.ignore_source = opt.ignore_source
        self.variational = opt.variational_dropout
        self.death_rate = death_rate
        self.fast_self_attention = opt.fast_self_attention

        self.preprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                  variational=self.variational)

        if not self.ignore_source:
            self.preprocess_src_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
            self.postprocess_src_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                          variational=self.variational)

            self.multihead_src = EncdecMultiheadAttn(opt.n_heads, opt.model_size, opt.attn_dropout)

        self.preprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                 variational=self.variational)

        self.multihead_tgt = RelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout)
        self.feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                   variational=self.variational)

    def forward(self, input, context, pos_emb, mask_tgt, mask_src,
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

            out, _ = self.multihead_tgt(query, pos_emb, None, mask_tgt, mems=mems,
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

    def log_prior(self):

        log_prior = 0
        log_prior += self.multihead_src.log_prior
        self.multihead_src.log_prior = 0
        log_prior += self.multihead_tgt.log_prior
        self.multihead_tgt.log_prior = 0
        log_prior += self.feedforward.log_prior
        self.feedforward.log_prior = 0
        return log_prior

    def log_variational_posterior(self):

        log_variational_posterior = 0
        log_variational_posterior += self.multihead_src.log_variational_posterior
        self.multihead_src.log_variational_posterior = 0
        log_variational_posterior += self.multihead_tgt.log_variational_posterior
        self.multihead_tgt.log_variational_posterior = 0
        log_variational_posterior += self.feedforward.log_variational_posterior
        self.feedforward.log_variational_posterior = 0
        return log_variational_posterior
