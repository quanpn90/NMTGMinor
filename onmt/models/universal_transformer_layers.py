import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.utils.weight_norm as WeightNorm
import onmt
import torch.nn.functional as F
from onmt.modules.bottle import Bottle
from onmt.modules.static_dropout import StaticDropout
from onmt.modules.linear import XavierLinear as Linear
from onmt.modules.linear import XavierLinear
from onmt.modules.linear import group_linear, FeedForwardSwish
from onmt.modules.linear import FeedForward
from onmt.modules.attention import MultiHeadAttention
from onmt.modules.dropout import VariationalDropout
from onmt.modules.optimized.encdec_attention import EncdecMultiheadAttn
from onmt.modules.optimized.self_attention import SelfMultiheadAttn
from collections import defaultdict
from .transformers import PrePostProcessing, EncoderLayer, DecoderLayer


class UniversalEncoderLayer(EncoderLayer):

    def __init__(self, opt, death_rate=0.0, **kwargs):
        super().__init__(opt, death_rate=death_rate)

    def forward(self, input, time_embedding, layer_vector, attn_mask):

        input = input + time_embedding.unsqueeze(1) + layer_vector

        coin = True
        if self.training:
            coin = (torch.rand(1)[0].item() >= self.death_rate)

        if coin:
            query = self.preprocess_attn(input)

            # print(query.size(), attn_mask.size())

            if self.fast_self_attention:
                out, _ = self.multihead(query, query, query, attn_mask, None)
            else:
                out, _ = self.multihead(query, query, query, attn_mask)

            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_attn(out, input)

            """ Feed forward layer 
                layernorm > ffn > dropout > residual
            """
            out = self.feedforward(self.preprocess_ffn(input))

            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_ffn(out, input)

        return input


class UniversalDecoderLayer(DecoderLayer):

    def __init__(self, opt, death_rate=0.0):
        super().__init__(opt, death_rate=death_rate)

    def forward(self, input, time_embedding, layer_vector, context, mask_tgt, mask_src,
                incremental=False, incremental_cache=None, reuse_source=True):
        """
        :param input:
        :param layer_vector:
        :param context:
        :param mask_tgt:
        :param mask_src:
        :param incremental:
        :param incremental_cache:
        :param reuse_source:
        :return:
        """
        # sum up
        input = input + time_embedding.unsqueeze(1) + layer_vector
        assert(len(input.shape) == 3)
        if incremental:
            if incremental_cache is None:
                incremental_cache = dict()

        coverage = None

        coin = True
        if self.training:
            coin = (torch.rand(1)[0].item() >= self.death_rate)

        if coin:

            query = self.preprocess_attn(input)

            if self.fast_self_attention:
                out, _, = self.multihead_tgt(query, query, query, None, mask_tgt,
                                             incremental=incremental,
                                             incremental_cache=incremental_cache)
            else:
                out, _, = self.multihead_tgt(query, query, query, mask_tgt,
                                             incremental=incremental,
                                             incremental_cache=incremental_cache)

            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_attn(out, input)

            """ Context Attention layer 
                layernorm > attn > dropout > residual
            """
            if not self.ignore_source:
                query = self.preprocess_src_attn(input)
                out, coverage = self.multihead_src(query, context, context, mask_src,
                                                   incremental=incremental,
                                                   incremental_cache=incremental_cache)

                if self.training and self.death_rate > 0:
                    out = out / (1 - self.death_rate)

                input = self.postprocess_src_attn(out, input)
            else:
                coverage = None

            """ Feed forward layer 
                layernorm > ffn > dropout > residual
            """
            out = self.feedforward(self.preprocess_ffn(input))

            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_ffn(out, input)

        return input, coverage, incremental_cache

