import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math

from onmt.modules.optimized.relative_self_attention import RelativeSelfMultiheadAttn
from onmt.modules.optimized.feed_forward import PositionWiseFeedForward
from onmt.modules.dropout import variational_dropout
from onmt.modules.convolution import ConformerConvBlock
from onmt.models.transformer_layers import PrePostProcessing
from onmt.modules.multilingual_factorized.linear import MFWPositionWiseFeedForward
from onmt.modules.multilingual_factorized.encdec_attention import MFWEncdecMultiheadAttn
from onmt.modules.multilingual_factorized.relative_attention import MFWRelativeSelfMultiheadAttn


class ConformerEncoderLayer(nn.Module):

    def __init__(self, opt, death_rate=0.0):

        super(ConformerEncoderLayer, self).__init__()

        # FFN -> SelfAttention -> Conv -> FFN
        # PreNorm
        self.opt = opt
        self.variational = opt.variational_dropout
        self.death_rate = death_rate
        self.dropout = opt.dropout
        self.ffn_scale = 0.5
        self.mfw = opt.multilingual_factorized_weights
        self.weight_drop = opt.weight_drop

        self.preprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                  variational=self.variational)

        if self.mfw:
            self.attn = MFWRelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout,
                                                     n_languages=opt.n_languages, rank=opt.mfw_rank,
                                                     use_multiplicative=opt.mfw_multiplicative,
                                                     weight_drop=self.weight_drop,
                                                     mfw_activation=opt.mfw_activation)
        else:
            self.attn = RelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout)

        self.preprocess_mcr_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')

        if self.mfw:
            self.mcr_feedforward = MFWPositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                              variational=self.variational,
                                                              n_languages=opt.n_languages, rank=opt.mfw_rank,
                                                              use_multiplicative=opt.mfw_multiplicative,
                                                              weight_drop=self.weight_drop,
                                                              mfw_activation=opt.mfw_activation)
        else:
            self.mcr_feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                           variational=self.variational)

        self.preprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')

        if self.mfw:

            self.feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                       variational=self.variational)
        else:
            self.feedforward = MFWPositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                          variational=self.variational,
                                                          n_languages=opt.n_languages, rank=opt.mfw_rank,
                                                          use_multiplicative=opt.mfw_multiplicative,
                                                          weight_drop=self.weight_drop,
                                                          mfw_activation=opt.mfw_activation)

        # there is batch norm inside convolution already
        # so no need for layer norm?
        self.preprocess_conv = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_conv = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                  variational=self.variational)
        self.conv = ConformerConvBlock(opt.model_size, opt.conv_kernel)

    def forward(self, input, pos_emb, attn_mask, incremental=False, incremental_cache=None, mems=None,
                src_lang=None):

        assert incremental is False
        assert incremental_cache is None

        coin = True
        if self.training and self.death_rate > 0:
            coin = (torch.rand(1)[0].item() >= self.death_rate)
            ffn_scale = self.ffn_scale / (1 - self.death_rate)

        else:
            ffn_scale = self.ffn_scale

        if coin:
            out = self.mcr_feedforward(self.preprocess_mcr_ffn(input), src_lang)

            out = out * ffn_scale

            if not self.variational:
                out = F.dropout(out, p=self.dropout, training=self.training)
            else:
                out = variational_dropout(out, p=self.dropout, training=self.training)

            input = input + out

            # attention
            attn_input = self.preprocess_attn(input)
            if self.mfw:
                out, _ = self.attn(attn_input, pos_emb, src_lang, attn_mask, None)
            else:
                out, _ = self.attn(attn_input, pos_emb, attn_mask, None)

            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_attn(out, input)

            # convolution
            conv_input = self.preprocess_conv(input)
            out = self.conv(conv_input)

            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_conv(out, input)

            # last ffn
            out = self.feedforward(self.preprocess_ffn(input), src_lang)

            out = out * ffn_scale

            if not self.variational:
                out = F.dropout(out, p=self.dropout, training=self.training)
            else:
                out = variational_dropout(out, p=self.dropout, training=self.training)

            input = input + out

            return input

        return input
