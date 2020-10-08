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

        self.preprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                  variational=self.variational)

        self.attn = RelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout)

        self.preprocess_mcr_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')

        self.mcr_feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                       variational=self.variational, activation='swish')

        self.preprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')

        self.feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                   variational=self.variational, activation='swish')

        # there is batch norm inside convolution already
        # so no need for layer norm?
        self.preprocess_conv = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_conv = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                  variational=self.variational)
        self.conv = ConformerConvBlock(opt.model_size, opt.conv_kernel, activation='swish')

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




