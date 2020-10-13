import torch
import torch.nn as nn
import torch.nn.functional as F
import onmt

from onmt.models.transformer_layers import PrePostProcessing
from onmt.modules.optimized.encdec_attention import EncdecMultiheadAttn
from onmt.modules.optimized.relative_self_attention import RelativeSelfMultiheadAttn
from onmt.modules.optimized.feed_forward import PositionWiseFeedForward
from onmt.modules.multilingual_factorized.linear import MFWPositionWiseFeedForward
from onmt.modules.multilingual_factorized.encdec_attention import MFWEncdecMultiheadAttn
from onmt.modules.multilingual_factorized.relative_attention import MFWRelativeSelfMultiheadAttn
from onmt.modules.dropout import variational_dropout


class RelativeTransformerEncoderLayer(nn.Module):
    def __init__(self, opt, death_rate=0.0, **kwargs):
        super(RelativeTransformerEncoderLayer, self).__init__()
        self.variational = opt.variational_dropout
        self.batch_ensemble = opt.batch_ensemble
        # self.multilingual_factorized_weights = opt.multilingual_factorized_weights
        self.death_rate = death_rate
        self.mfw = opt.multilingual_factorized_weights
        self.macaron = opt.macaron
        self.ffn_scale = 0.5 if self.macaron else 1
        self.dropout = opt.dropout

        if self.macaron:
            self.preprocess_mcr_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
            self.postprocess_mcr_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                         variational=self.variational)

            if self.mfw:
                self.mcr_feedforward = MFWPositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                                  variational=self.variational,
                                                                  n_languages=opt.n_languages, rank=opt.mfw_rank,
                                                                  use_multiplicative=opt.mfw_multiplicative)
            else:
                self.mcr_feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                               variational=self.variational)

        self.preprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                  variational=self.variational)
        self.preprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                 variational=self.variational)
        d_head = opt.model_size // opt.n_heads

        if self.mfw:
            self.feedforward = MFWPositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                          variational=self.variational,
                                                          n_languages=opt.n_languages, rank=opt.mfw_rank,
                                                          use_multiplicative=opt.mfw_multiplicative)

            self.multihead = MFWRelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout,
                                                          n_languages=opt.n_languages, rank=opt.mfw_rank,
                                                          use_multiplicative=opt.mfw_multiplicative)

        else:
            self.feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                       variational=self.variational)

            self.multihead = RelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout)

    def forward(self, input, pos_emb, attn_mask, incremental=False, incremental_cache=None, mems=None,
                src_lang=None):

        if incremental and incremental_cache is None:
            incremental_cache = dict()

        coin = True
        if self.training and self.death_rate > 0:
            coin = (torch.rand(1)[0].item() >= self.death_rate)
            ffn_scale = self.ffn_scale / (1 - self.death_rate)
        else:
            ffn_scale = self.ffn_scale

        if coin:
            if self.macaron:
                out = self.mcr_feedforward(self.preprocess_mcr_ffn(input), src_lang)

                if ffn_scale != 1:
                    out = out * ffn_scale

                input = self.postprocess_mcr_ffn(out, input)

            # self-attention block
            query = self.preprocess_attn(input)

            if self.mfw:
                out, _ = self.multihead(query, pos_emb, src_lang, attn_mask, None, mems=mems,
                                        incremental=incremental, incremental_cache=incremental_cache)
            else:
                out, _ = self.multihead(query, pos_emb, attn_mask, None, mems=mems,
                                        incremental=incremental, incremental_cache=incremental_cache)

            # rescaling before residual
            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_attn(out, input)

            """ Feed forward layer 
                layernorm > ffn > dropout > residual
            """
            out = self.feedforward(self.preprocess_ffn(input), src_lang)

            if ffn_scale != 1:
                out = out * ffn_scale

            input = self.postprocess_ffn(out, input)

        if incremental:
            return input, incremental_cache

        return input


class RelativeTransformerDecoderLayer(nn.Module):

    def __init__(self, opt, death_rate=0.0):
        super(RelativeTransformerDecoderLayer, self).__init__()
        self.ignore_source = opt.ignore_source
        self.variational = opt.variational_dropout
        self.death_rate = death_rate
        self.batch_ensemble = opt.batch_ensemble
        self.mfw = opt.multilingual_factorized_weights
        self.macaron = opt.macaron
        self.ffn_scale = 0.5 if self.macaron else 1
        self.dropout = opt.dropout

        if self.macaron:
            self.preprocess_mcr_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
            self.postprocess_mcr_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                         variational=self.variational)

            if self.mfw:
                self.mcr_feedforward = MFWPositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                                  variational=self.variational,
                                                                  n_languages=opt.n_languages, rank=opt.mfw_rank,
                                                                  use_multiplicative=opt.mfw_multiplicative)
            else:
                self.mcr_feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                               variational=self.variational)

        self.preprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                  variational=self.variational)

        if not self.ignore_source:
            self.preprocess_src_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
            self.postprocess_src_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                          variational=self.variational)
            # if self.batch_ensemble > 0:
            #     self.multihead_src = BEEncdecMultiheadAttn(opt.n_heads, opt.model_size, opt.attn_dropout,
            #                                                ensemble=self.batch_ensemble)
            # else:

            if not self.mfw:
                self.multihead_src = EncdecMultiheadAttn(opt.n_heads, opt.model_size, opt.attn_dropout)
            else:
                self.multihead_src = MFWEncdecMultiheadAttn(opt.n_heads, opt.model_size, opt.attn_dropout,
                                                            n_languages=opt.n_languages, rank=opt.mfw_rank,
                                                            use_multiplicative=opt.mfw_multiplicative)

        self.preprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                 variational=self.variational)

        d_head = opt.model_size // opt.n_heads

        if self.mfw:
            self.feedforward = MFWPositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                          variational=self.variational,
                                                          n_languages=opt.n_languages, rank=opt.mfw_rank,
                                                          use_multiplicative=opt.mfw_multiplicative)

            self.multihead_tgt = MFWRelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout,
                                                              n_languages=opt.n_languages, rank=opt.mfw_rank,
                                                              use_multiplicative=opt.mfw_multiplicative)
        else:

            self.feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                       variational=self.variational)

            self.multihead_tgt = RelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout)

    def forward(self, input, context, pos_emb, mask_tgt, mask_src,
                src_lang=None, tgt_lang=None,
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

            if self.macaron:
                out = self.mcr_feedforward(self.preprocess_mcr_ffn(input), src_lang)

                if self.training and self.death_rate > 0:
                    out = out / (1 - self.death_rate)

                if not self.variational:
                    out = F.dropout(out, p=self.dropout, training=self.training)
                else:
                    out = variational_dropout(out, p=self.dropout, training=self.training)

                input = input + self.ffn_scale * out

            query = self.preprocess_attn(input)

            if self.mfw:
                out, _ = self.multihead_tgt(query, pos_emb, tgt_lang, None, mask_tgt, mems=mems,
                                            incremental=incremental, incremental_cache=incremental_cache)
            else:
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

                if self.mfw:
                    out, coverage = self.multihead_src(query, context, context, src_lang, tgt_lang, mask_src,
                                                       incremental=incremental_source,
                                                       incremental_cache=incremental_cache)
                else:
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
            out = self.feedforward(self.preprocess_ffn(input), tgt_lang)

            # rescaling before residual
            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            if not self.variational:
                out = F.dropout(out, p=self.dropout, training=self.training)
            else:
                out = variational_dropout(out, p=self.dropout, training=self.training)

            input = input + self.ffn_scale * out
        else:
            coverage = None

        return input, coverage, incremental_cache
