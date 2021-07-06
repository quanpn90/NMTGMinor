import torch
import torch.nn as nn
import torch.nn.functional as F
import onmt

from onmt.modules.pre_post_processing import PrePostProcessing
from onmt.modules.linear import FeedForward
from onmt.modules.linear import XavierLinear as Linear
from onmt.modules.attention import MultiHeadAttention
from onmt.modules.dropout import VariationalDropout

from onmt.modules.optimized.encdec_attention import EncdecMultiheadAttn
from onmt.modules.optimized.feed_forward import PositionWiseFeedForward
from onmt.modules.optimized.relative_self_attention import RelativeSelfMultiheadAttn

from onmt.modules.multilingual_factorized.linear import MFWPositionWiseFeedForward
from onmt.modules.multilingual_factorized.encdec_attention import MFWEncdecMultiheadAttn
from onmt.modules.multilingual_factorized.relative_attention import MFWRelativeSelfMultiheadAttn

from onmt.modules.multilingual_partitioned.linear import MPPositionWiseFeedForward
from onmt.modules.multilingual_partitioned.encdec_attention import MPEncdecMultiheadAttn
from onmt.modules.multilingual_partitioned.relative_attention import MPRelativeSelfMultiheadAttn


def preprocessing(rezero, *args, **kwargs):
    if rezero:
        return Identity()
    else:
        return PrePostProcessing(*args, **kwargs)


class RelativeGateEncoderLayer(nn.Module):
    def __init__(self, opt, **kwargs):
        super(RelativeGateEncoderLayer, self).__init__()
        self.variational = opt.variational_dropout
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
        self.learnable_pos = opt.learnable_position_encoding
        self.residual_dropout = opt.residual_dropout if opt.residual_dropout >= 0 else opt.dropout
        self.ffn_dropout = opt.ffn_dropout if opt.ffn_dropout >= 0 else opt.dropout

        if self.macaron:
            self.preprocess_mcr_ffn = preprocessing(self.rezero, opt.model_size, 0.0,
                                                    multilingual=self.mln, sequence='n', n_languages=opt.n_languages)
            self.postprocess_mcr_ffn = PrePostProcessing(opt.model_size, self.residual_dropout,
                                                         sequence='dz' if self.rezero else 'da',
                                                         variational=self.variational)

            if self.mfw:
                self.mcr_feedforward = MFWPositionWiseFeedForward(opt.model_size, opt.inner_size, self.ffn_dropout,
                                                                  variational=self.variational,
                                                                  n_languages=opt.n_languages, rank=opt.mfw_rank,
                                                                  use_multiplicative=opt.mfw_multiplicative,
                                                                  activation=opt.ffn_activation,
                                                                  glu=opt.ffn_glu)
            else:
                self.mcr_feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, self.ffn_dropout,
                                                               variational=self.variational,
                                                               activation=opt.ffn_activation,
                                                               glu=opt.ffn_glu)

        if self.mfw:
            assert not self.mpw, "[ERROR] factorized and partitioned weights cannot be used at the same time."

        self.preprocess_attn = preprocessing(self.rezero, opt.model_size, 0.0,
                                             multilingual=self.mln, sequence='n', n_languages=opt.n_languages)
        self.postprocess_attn = PrePostProcessing(opt.model_size, self.residual_dropout,
                                                  sequence='dz' if self.rezero else 'da',
                                                  variational=self.variational)

        self.preprocess_src_attn = preprocessing(self.rezero, opt.model_size, 0.0, sequence='n',
                                                 multilingual=self.mln, n_languages=opt.n_languages)
        self.postprocess_src_attn = PrePostProcessing(opt.model_size, self.residual_dropout,
                                                      sequence='dz' if self.rezero else 'da',
                                                      variational=self.variational)

        self.preprocess_ffn = preprocessing(self.rezero, opt.model_size, 0.0,
                                            multilingual=self.mln, sequence='n', n_languages=opt.n_languages)
        self.postprocess_ffn = PrePostProcessing(opt.model_size, self.residual_dropout,
                                                 sequence='dz' if self.rezero else 'da',
                                                 variational=self.variational)
        d_head = opt.model_size // opt.n_heads

        if self.mfw:

            self.feedforward = MFWPositionWiseFeedForward(opt.model_size, opt.inner_size, self.ffn_dropout,
                                                          variational=self.variational,
                                                          n_languages=opt.n_languages, rank=opt.mfw_rank,
                                                          use_multiplicative=opt.mfw_multiplicative,
                                                          weight_drop=self.weight_drop,
                                                          mfw_activation=opt.mfw_activation,
                                                          activation=opt.ffn_activation,
                                                          glu=opt.ffn_glu)

            self.multihead = MFWRelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout,
                                                          learnable_pos=self.learnable_pos,
                                                          max_pos=opt.max_pos_length,
                                                          n_languages=opt.n_languages, rank=opt.mfw_rank,
                                                          use_multiplicative=opt.mfw_multiplicative,
                                                          weight_drop=self.weight_drop,
                                                          mfw_activation=opt.mfw_activation)

            self.multihead_src = MFWEncdecMultiheadAttn(opt.n_heads, opt.model_size, opt.attn_dropout,
                                                        n_languages=opt.n_languages, rank=opt.mfw_rank,
                                                        use_multiplicative=opt.mfw_multiplicative,
                                                        weight_drop=self.weight_drop,
                                                        mfw_activation=opt.mfw_activation)

        elif self.mpw:
            if not self.no_ffn:
                self.feedforward = MPPositionWiseFeedForward(opt.model_size, opt.inner_size, self.ffn_dropout,
                                                             variational=self.variational,
                                                             factor_size=opt.mpw_factor_size)

            self.multihead = MPRelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout,
                                                         factor_size=opt.mpw_factor_size)

        else:
            self.feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, self.ffn_dropout,
                                                       variational=self.variational,
                                                       activation=opt.ffn_activation,
                                                       glu=opt.ffn_glu)

            self.multihead = RelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout,
                                                       learnable_pos=self.learnable_pos,
                                                       max_pos=opt.max_pos_length)

            self.multihead_src = EncdecMultiheadAttn(opt.n_heads, opt.model_size, opt.attn_dropout)

        if self.depthwise_conv:
            self.preprocess_conv = preprocessing(self.rezero, opt.model_size, 0.0,
                                                 multilingual=self.mln, sequence='n', n_languages=opt.n_languages)
            self.postprocess_conv = PrePostProcessing(opt.model_size, self.residual_dropout,
                                                      sequence='dz' if self.rezero else 'da',
                                                      variational=self.variational)
            self.depthwise_conv = ConformerConvBlock(opt.model_size, opt.conv_kernel, bias=True)
        else:
            self.depthwise_conv = None

        self.gate_linear = Linear(2 * opt.model_size, opt.model_size)
        self.preprocess_gate = preprocessing(self.rezero, 2 * opt.model_size, 0.0,
                                             multilingual=self.mln, sequence='n', n_languages=opt.n_languages)

    def forward(self, input, context, pos_emb, attn_mask, context_mask, src_lang=None, factorize=False):
        """
        :param context: discourse context [T_d x B x H]
        :param factorize:
        :param input: tensor [T x B x H]
        :param pos_emb: tensor [T x 1 x H]
        :param attn_mask: tensor [1 x T x B]
        :param context_mask: tensor [1 x T_d x B]
        :param src_lang: tensor [B] or None
        :return:
        """

        if self.macaron:
            out = self.mcr_feedforward(self.preprocess_mcr_ffn(input), src_lang, factorize=factorize)
            ffn_scale = self.ffn_scale

            input = self.postprocess_mcr_ffn(out * ffn_scale, input)

        """
        Self-attention block
        """
        query = self.preprocess_attn(input, factor=src_lang)

        if self.mfw or self.mpw:
            out, _ = self.multihead(query, pos_emb, src_lang, attn_mask, None, factorize=factorize)
        else:
            out, _ = self.multihead(query, pos_emb, attn_mask, None)

        input_present = self.postprocess_attn(out, input)

        """
        Context attention block
        """

        query = self.preprocess_src_attn(input, factor=src_lang)

        if self.mfw or self.mpw:
            out, _ = self.multihead_src(query, context, context, src_lang, src_lang, context_mask,
                                        factorize=factorize)
        else:
            out, _ = self.multihead_src(query, context, context, context_mask)

        input_past = self.postprocess_src_attn(out, input)

        """ 
        Gate
        """

        gate_input = self.preprocess_gate(torch.cat([input_past, input_present], dim=-1))

        gate = torch.sigmoid(self.gate_linear(gate_input))

        input = gate * input_present + (1 - gate) * input_past

        """ 
        Feed forward layer 
        """
        if not self.no_ffn:
            out = self.feedforward(self.preprocess_ffn(input, factor=src_lang), src_lang, factorize=factorize)

            # rescaling before residual
            ffn_scale = self.ffn_scale

            input = self.postprocess_ffn(out * ffn_scale, input)

        if self.multilingual_adapter:
            input = self.adapters(input, src_lang)

        return input
