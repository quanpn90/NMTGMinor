import torch
import torch.nn as nn
import torch.nn.functional as F
import onmt

from onmt.models.transformer_layers import MultiHeadAttention, Linear
from onmt.modules.relative_attention import RelPartialLearnableMultiHeadAttn
from onmt.modules.optimized.relative_self_attention import RelativeSelfMultiheadAttn
from onmt.modules.pre_post_processing import PrePostProcessing
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

from onmt.modules.multilingual_factorized.linear import MFWPositionWiseFeedForward
from onmt.modules.multilingual_factorized.encdec_attention import MFWEncdecMultiheadAttn
from onmt.modules.multilingual_factorized.relative_attention import MFWRelativeSelfMultiheadAttn

from onmt.modules.multilingual_partitioned.linear import MPPositionWiseFeedForward
from onmt.modules.multilingual_partitioned.encdec_attention import MPEncdecMultiheadAttn
from onmt.modules.multilingual_partitioned.relative_attention import MPRelativeSelfMultiheadAttn
from onmt.modules.convolution import ConformerConvBlock
from onmt.modules.identity import Identity


class LIDFeedForward(nn.Module):

    def __init__(self, *args):
        super().__init__()


def preprocessing(rezero, *args, **kwargs):
    if rezero:
        return Identity()
    else:
        return PrePostProcessing(*args, **kwargs)


class RelativeTransformerEncoderLayer(nn.Module):
    def __init__(self, opt, death_rate=0.0, **kwargs):
        super(RelativeTransformerEncoderLayer, self).__init__()
        self.variational = opt.variational_dropout
        self.death_rate = death_rate
        self.fast_self_attention = opt.fast_self_attention
        self.depthwise_conv = opt.depthwise_conv
        self.mfw = opt.multilingual_factorized_weights
        self.mpw = opt.multilingual_partitioned_weights
        self.mln = opt.multilingual_layer_norm
        self.no_ffn = opt.no_ffn
        self.weight_drop = opt.weight_drop
        self.multilingual_adapter = opt.multilingual_adapter
        self.adapter_bottleneck_size = opt.adapter_bottleneck_size
        self.macaron = opt.macaron
        self.ffn_scale = 0.5 if self.macaron else 1
        self.rezero = opt.rezero

        if self.macaron:
            self.preprocess_mcr_ffn = preprocessing(self.rezero, opt.model_size, opt.dropout,
                                                    multilingual=self.mln, sequence='n', n_languages=opt.n_languages)
            self.postprocess_mcr_ffn = PrePostProcessing(opt.model_size, opt.dropout,
                                                         sequence='dz' if self.rezero else 'da',
                                                         variational=self.variational)

            if self.mfw:
                self.mcr_feedforward = MFWPositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                                  variational=self.variational,
                                                                  n_languages=opt.n_languages, rank=opt.mfw_rank,
                                                                  use_multiplicative=opt.mfw_multiplicative,
                                                                  activation=opt.ffn_activation,
                                                                  glu=opt.ffn_glu)
            else:
                self.mcr_feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                               variational=self.variational,
                                                               activation=opt.ffn_activation,
                                                               glu=opt.ffn_glu)

        if self.mfw:
            assert not self.mpw, "[ERROR] factorized and partitioned weights cannot be used at the same time."

        self.preprocess_attn = preprocessing(self.rezero, opt.model_size, opt.dropout,
                                             multilingual=self.mln, sequence='n', n_languages=opt.n_languages)
        self.postprocess_attn = PrePostProcessing(opt.model_size, opt.dropout,
                                                  sequence='dz' if self.rezero else 'da',
                                                  variational=self.variational)

        if not self.no_ffn:
            self.preprocess_ffn = preprocessing(self.rezero, opt.model_size, opt.dropout,
                                                multilingual=self.mln, sequence='n', n_languages=opt.n_languages)
            self.postprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout,
                                                     sequence='dz' if self.rezero else 'da',
                                                     variational=self.variational)
        d_head = opt.model_size // opt.n_heads

        if self.mfw:
            if not self.no_ffn:
                self.feedforward = MFWPositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                              variational=self.variational,
                                                              n_languages=opt.n_languages, rank=opt.mfw_rank,
                                                              use_multiplicative=opt.mfw_multiplicative,
                                                              weight_drop=self.weight_drop,
                                                              mfw_activation=opt.mfw_activation,
                                                              activation=opt.ffn_activation,
                                                              glu=opt.ffn_glu)

            self.multihead = MFWRelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout,
                                                          n_languages=opt.n_languages, rank=opt.mfw_rank,
                                                          use_multiplicative=opt.mfw_multiplicative,
                                                          weight_drop=self.weight_drop,
                                                          mfw_activation=opt.mfw_activation)

        elif self.mpw:
            if not self.no_ffn:
                self.feedforward = MPPositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                             variational=self.variational,
                                                             factor_size=opt.mpw_factor_size)

            self.multihead = MPRelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout,
                                                         factor_size=opt.mpw_factor_size)

        else:
            if not self.no_ffn:
                self.feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                           variational=self.variational,
                                                           activation=opt.ffn_activation,
                                                           glu=opt.ffn_glu)

            self.multihead = RelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout)

        if self.depthwise_conv:
            self.preprocess_conv = preprocessing(self.rezero, opt.model_size, opt.dropout,
                                                 multilingual=self.mln, sequence='n', n_languages=opt.n_languages)
            self.postprocess_conv = PrePostProcessing(opt.model_size, opt.dropout,
                                                      sequence='dz' if self.rezero else 'da',
                                                      variational=self.variational)
            self.depthwise_conv = ConformerConvBlock(opt.model_size, opt.conv_kernel, bias=True)
        else:
            self.depthwise_conv = None

        if self.multilingual_adapter:
            from onmt.modules.multilingual_factorized.multilingual_adapters import MultilingualAdapter
            self.adapters = MultilingualAdapter(opt.model_size, opt.adapter_bottleneck_size,
                                                n_languages=opt.n_languages,
                                                dropout=opt.dropout)

    def forward(self, input, pos_emb, attn_mask, src_lang=None,
                incremental=False, incremental_cache=None, mems=None):
        """
        :param input: tensor [T x B x H]
        :param pos_emb: tensor [T x 1 x H]
        :param attn_mask: tensor [1 x T x B]
        :param src_lang: tensor [B] or None
        :param incremental: None
        :param incremental_cache:
        :param mems: None
        :return:
        """

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

            if self.macaron:
                out = self.mcr_feedforward(self.preprocess_mcr_ffn(input), src_lang)

                if self.training and self.death_rate > 0:
                    ffn_scale = self.ffn_scale / (1 - self.death_rate)
                else:
                    ffn_scale = self.ffn_scale

                input = self.postprocess_mcr_ffn(out * ffn_scale, input)

            """
            Self-attention block
            """
            query = self.preprocess_attn(input, factor=src_lang)

            if self.mfw or self.mpw:
                out, _ = self.multihead(query, pos_emb, src_lang, attn_mask, None, mems=mems,
                                        incremental=incremental, incremental_cache=incremental_cache)
            else:
                out, _ = self.multihead(query, pos_emb, attn_mask, None, mems=mems,
                                        incremental=incremental, incremental_cache=incremental_cache)

            # rescaling before residual
            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_attn(out, input)

            """
            Convolution block
            """
            if self.depthwise_conv:
                if attn_mask is not None and attn_mask.any():
                    conv_mask = attn_mask.squeeze(0).unsqueeze(-1)
                else:
                    conv_mask = None
                out = self.depthwise_conv(self.preprocess_conv(input, factor=src_lang), pad_mask=conv_mask)

                # rescaling before residual
                if self.training and self.death_rate > 0:
                    out = out / (1 - self.death_rate)

                input = self.postprocess_conv(out, input)

            """ 
            Feed forward layer 
            """
            if not self.no_ffn:
                out = self.feedforward(self.preprocess_ffn(input, factor=src_lang), src_lang)

                # rescaling before residual
                if self.training and self.death_rate > 0:
                    ffn_scale = self.ffn_scale / (1 - self.death_rate)
                else:
                    ffn_scale = self.ffn_scale

                input = self.postprocess_ffn(out * ffn_scale, input)

            if self.multilingual_adapter:
                input = self.adapters(input, src_lang)

            # checking for inf/nan which can happen randomly in fp16 ...
            if torch.isinf(input).any() or torch.isnan(input).any():
                clamp_value = torch.finfo(input.dtype).max - 1000
                input.clamp_(min=-clamp_value, max=clamp_value)

        if incremental:
            return input, incremental_cache

        return input


class RelativeTransformerDecoderLayer(nn.Module):

    def __init__(self, opt, death_rate=0.0, lid_net=None):
        super(RelativeTransformerDecoderLayer, self).__init__()
        self.ignore_source = opt.ignore_source
        self.variational = opt.variational_dropout
        self.death_rate = death_rate
        self.mfw = opt.multilingual_factorized_weights
        self.mpw = opt.multilingual_partitioned_weights
        self.mln = opt.multilingual_layer_norm
        self.weight_drop = opt.weight_drop
        self.multilingual_adapter = opt.multilingual_adapter
        self.adapter_bottleneck_size = opt.adapter_bottleneck_size
        self.macaron = opt.macaron
        self.ffn_scale = 0.5 if self.macaron else 1
        self.rezero = opt.rezero

        self.preprocess_attn = preprocessing(self.rezero, opt.model_size, opt.dropout, sequence='n',
                                             multilingual=self.mln, n_languages=opt.n_languages)

        self.postprocess_attn = PrePostProcessing(opt.model_size, opt.dropout,
                                                  sequence='dz' if self.rezero else 'da',
                                                  variational=self.variational)

        if self.macaron:
            self.preprocess_mcr_ffn = preprocessing(self.rezero, opt.model_size, opt.dropout, sequence='n',
                                                    multilingual=self.mln, n_languages=opt.n_languages)
            self.postprocess_mcr_ffn = PrePostProcessing(opt.model_size, opt.dropout,
                                                         sequence='dz' if self.rezero else 'da',
                                                         variational=self.variational)

            if self.mfw:
                self.mcr_feedforward = MFWPositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                                  variational=self.variational,
                                                                  n_languages=opt.n_languages, rank=opt.mfw_rank,
                                                                  use_multiplicative=opt.mfw_multiplicative,
                                                                  activation=opt.ffn_activation,
                                                                  glu=opt.ffn_glu)
            else:
                self.mcr_feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                               variational=self.variational,
                                                               activation=opt.ffn_activation,
                                                               glu=opt.ffn_glu)

        if not self.ignore_source:
            self.preprocess_src_attn = preprocessing(self.rezero, opt.model_size, opt.dropout, sequence='n',
                                                     multilingual=self.mln, n_languages=opt.n_languages)
            self.postprocess_src_attn = PrePostProcessing(opt.model_size, opt.dropout,
                                                          sequence='dz' if self.rezero else 'da',
                                                          variational=self.variational)

            if self.mfw:
                self.multihead_src = MFWEncdecMultiheadAttn(opt.n_heads, opt.model_size, opt.attn_dropout,
                                                            n_languages=opt.n_languages, rank=opt.mfw_rank,
                                                            use_multiplicative=opt.mfw_multiplicative,
                                                            weight_drop=self.weight_drop,
                                                            mfw_activation=opt.mfw_activation)
            elif self.mpw:
                self.multihead_src = MPEncdecMultiheadAttn(opt.n_heads, opt.model_size, opt.attn_dropout,
                                                           factor_size=opt.mpw_factor_size)

            else:
                self.multihead_src = EncdecMultiheadAttn(opt.n_heads, opt.model_size, opt.attn_dropout)

        self.preprocess_ffn = preprocessing(self.rezero, opt.model_size, opt.dropout, sequence='n',
                                            multilingual=self.mln, n_languages=opt.n_languages)
        self.postprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='dz' if self.rezero else 'da',
                                                 variational=self.variational)

        d_head = opt.model_size // opt.n_heads

        if self.mfw:
            self.feedforward = MFWPositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                          variational=self.variational,
                                                          n_languages=opt.n_languages, rank=opt.mfw_rank,
                                                          use_multiplicative=opt.mfw_multiplicative,
                                                          weight_drop=self.weight_drop,
                                                          mfw_activation=opt.mfw_activation,
                                                          activation=opt.ffn_activation,
                                                          glu=opt.ffn_glu)

            self.multihead_tgt = MFWRelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout,
                                                              n_languages=opt.n_languages, rank=opt.mfw_rank,
                                                              use_multiplicative=opt.mfw_multiplicative,
                                                              weight_drop=self.weight_drop,
                                                              mfw_activation=opt.mfw_activation)
        elif self.mpw:
            self.feedforward = MPPositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                         variational=self.variational,
                                                         factor_size=opt.mpw_factor_size)

            self.multihead_tgt = MPRelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout,
                                                             factor_size=opt.mpw_factor_size)
        else:
            self.multihead_tgt = RelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout)

            self.feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                       variational=self.variational,
                                                       activation=opt.ffn_activation,
                                                       glu=opt.ffn_glu)

        self.lfv_multilingual = opt.lfv_multilingual

        if opt.lfv_multilingual:
            self.lid_net = lid_net
            self.lfv_mapper = nn.Linear(opt.bottleneck_size, opt.model_size)
        else:
            self.lid_net = None
            self.lfv_mapper = None

        if self.multilingual_adapter:
            from onmt.modules.multilingual_factorized.multilingual_adapters import MultilingualAdapter
            self.adapters = MultilingualAdapter(opt.model_size, opt.adapter_bottleneck_size,
                                                n_languages=opt.n_languages,
                                                dropout=opt.dropout)

    def forward(self, input, context, pos_emb, lfv=None, mask_tgt=None, mask_src=None,
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
                mems = self.preprocess_attn(mems, factor=tgt_lang)
            else:
                mems = None

            if self.macaron:
                out = self.mcr_feedforward(self.preprocess_mcr_ffn(input), src_lang)

                if self.training and self.death_rate > 0:
                    ffn_scale = self.ffn_scale / (1 - self.death_rate)
                else:
                    ffn_scale = self.ffn_scale

                input = self.postprocess_mcr_ffn(out * ffn_scale, input)

            query = self.preprocess_attn(input, factor=tgt_lang)
            if self.mfw or self.mpw:
                out, _ = self.multihead_tgt(query, pos_emb, tgt_lang, None, mask_tgt, mems=mems,
                                            incremental=False, incremental_cache=incremental_cache)
                # incremental=incremental, incremental_cache=incremental_cache)
            else:
                out, _ = self.multihead_tgt(query, pos_emb, None, mask_tgt, mems=mems,
                                            incremental=False, incremental_cache=incremental_cache)
                # incremental=incremental, incremental_cache=incremental_cache)

            # rescaling before residual
            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_attn(out, input)

            """ Context Attention layer 
                layernorm > attn > dropout > residual
            """
            if not self.ignore_source:
                query = self.preprocess_src_attn(input, factor=tgt_lang)
                incremental_source = incremental and reuse_source

                if self.mfw or self.mpw:
                    out, coverage = self.multihead_src(query, context, context, tgt_lang, tgt_lang, mask_src,
                                                       incremental=incremental_source,
                                                       incremental_cache=incremental_cache)
                else:
                    out, coverage = self.multihead_src(query, context, context, mask_src,
                                                       incremental=incremental_source,
                                                       incremental_cache=incremental_cache)

                # rescaling before residual
                if self.training and self.death_rate > 0:
                    out = out / (1 - self.death_rate)

                if self.lid_net is not None:
                    lid_logits, lfv = self.lid_net(out)
                else:
                    lid_logits, lfv = None, None

                input = self.postprocess_src_attn(out, input)
            else:
                coverage = None

            """ Feed forward layer 
                layernorm > ffn > dropout > residual
            """
            out = self.feedforward(self.preprocess_ffn(input, factor=tgt_lang), tgt_lang)

            # rescaling before residual
            if self.training and self.death_rate > 0:
                ffn_scale = self.ffn_scale / (1 - self.death_rate)
            else:
                ffn_scale = self.ffn_scale

            input = self.postprocess_ffn(out * ffn_scale, input)

            if self.multilingual_adapter:
                input = self.adapters(input, tgt_lang)

            # checking for inf/nan which can happen randomly in fp16 ...
            if torch.isinf(input).any() or torch.isnan(input).any():
                clamp_value = torch.finfo(input.dtype).max - 1000
                input.clamp_(min=-clamp_value, max=clamp_value)
        else:
            coverage = None
            lid_logits = None
            lfv = None

        if self.lfv_multilingual:
            return input, coverage, incremental_cache, lid_logits, lfv

        return input, coverage, incremental_cache
