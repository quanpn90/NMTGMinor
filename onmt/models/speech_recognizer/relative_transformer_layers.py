import torch
import torch.nn as nn
import torch.nn.functional as F
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


class LIDFeedForward(nn.Module):

    def __init__(self, input_size, hidden_size, bottleneck_size, output_size, n_hidden=2, dropout=0.0):
        """
        :param input_size:
        :param hidden_size:
        :param bottleneck_size:
        :param output_size:
        :param n_hidden: number of hidden states between first hidden and the bottleneck
        """
        super().__init__()
        self.input_size = input_size
        self.dropout = dropout
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.hiddens = nn.ModuleList()
        for i in range(n_hidden):
            self.hiddens.append(nn.Linear(hidden_size, hidden_size))

        self.bottleneck_in = nn.Linear(hidden_size, bottleneck_size)
        self.bottleneck_out = nn.Linear(bottleneck_size, hidden_size)
        self.last_linear = nn.Linear(hidden_size, output_size)

        try:
            from apex.mlp.mlp import mlp_function
            self.optimized = 1
            self.fast_mlp_func = mlp_function
        except ModuleNotFoundError as e:
            self.optimized = 2

    def forward(self, input):
        """
        :param input: Input should have size [len x bsz x input_size]
        :return:
        """
        assert self.input_size == input.size(-1)
        input = input.detach()

        if self.optimized == 1:
            weights = [self.linear_in.weight]
            biases = [self.linear_in.bias]
            for i in range(len(self.hiddens)):
                weights.append(self.hiddens[i].weight)
                biases.append(self.hiddens[i].bias)
            weights.append(self.bottleneck_in.weight)
            biases.append(self.bottleneck_in.bias)

            seq_len, bsz, inp_size = input.size(0), input.size(1), input.size(2)
            hidden = self.fast_mlp_func(True, 1, input.view(seq_len * bsz, -1), *weights, *biases)

            bottleneck = F.relu(hidden)
            bottleneck = F.dropout(bottleneck, p=self.dropout, training=self.training)

            weights = [self.bottleneck_out.weight, self.last_linear.weight]
            biases = [self.bottleneck_out.bias, self.last_linear.bias]

            logits = self.fast_mlp_func(True, 1, bottleneck, *weights, *biases)

            logits = logits.view(seq_len, bsz, -1)
            bottleneck = bottleneck.view(seq_len, bsz, -1)
        else:
            hidden = F.relu(self.linear_in(input))

            for i in range(len(self.hiddens)):
                hidden = F.relu(self.hiddens[i](hidden))

            bottleneck = F.relu(self.bottleneck_in(hidden))

            hidden = F.relu(self.bottleneck_out(F.dropout(bottleneck, p=self.dropout, training=self.training)))

            logits = self.last_linear(hidden)

        return logits, bottleneck


class RelativeTransformerEncoderLayer(nn.Module):
    # def __init__(self, h, d_model, p, d_ff, attn_p=0.1, variational=False, death_rate=0.0, **kwargs):
    def __init__(self, opt, death_rate=0.0, **kwargs):
        super(RelativeTransformerEncoderLayer, self).__init__()
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


class RelativeTransformerDecoderLayer(nn.Module):

    # def __init__(self, h, d_model, p,    d_ff, attn_p=0.1, version=1.0, ignore_source=False,
    #              variational=False, death_rate=0.0):
    def __init__(self, opt, death_rate=0.0, lid_net=None):
        super(RelativeTransformerDecoderLayer, self).__init__()
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

        d_head = opt.model_size // opt.n_heads

        self.multihead_tgt = RelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout)

        self.feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                   variational=self.variational)

        self.lfv_multilingual = opt.lfv_multilingual
        if opt.lfv_multilingual:
            self.lid_net = lid_net
            self.lfv_mapper = nn.Linear(opt.bottleneck_size, opt.model_size)
        else:
            self.lid_net = None
            self.lfv_mapper = None

    def forward(self, input, context, pos_emb, lfv=None, mask_tgt=None, mask_src=None,
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

            if lfv is not None:
                # multiply the input with the bottleneck lfv features from the LID network
                # print(lfv.size())
                input = torch.mul(torch.tanh(self.lfv_mapper(lfv)), input)

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
            out = self.feedforward(self.preprocess_ffn(input))

            # rescaling before residual
            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_ffn(out, input)
        else:
            coverage = None
            lid_logits = None
            lfv = None

        if self.lfv_multilingual:
            return input, coverage, incremental_cache, lid_logits, lfv

        return input, coverage, incremental_cache
