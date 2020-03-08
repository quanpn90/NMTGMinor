import torch
import torch.nn as nn
import torch.nn.functional as F
from onmt.modules.relative_attention import RelPartialLearnableMultiHeadAttn
from onmt.models.transformer_layers import PositionalEncoding, PrePostProcessing
from onmt.models.transformer_layers import EncoderLayer, DecoderLayer
from onmt.models.transformers import TransformerEncoder, TransformerDecoder, TransformerDecodingState
import onmt
from onmt.modules.bottle import Bottle
from onmt.modules.dropout import embedded_dropout
from onmt.models.transformer_layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing
from onmt.models.transformer_layers import EncoderLayer, DecoderLayer
from onmt.models.relative_transformer_layers import RelativeTransformerEncoderLayer, RelativeTransformerDecoderLayer
from onmt.models.unified_transformer import UnifiedTransformer
from onmt.models.relative_transformer import SinusoidalPositionalEmbedding, LearnablePostionEmbedding, \
    StreamState, StreamDecodingState
from onmt.utils import flip, expected_length
from collections import defaultdict
import math


class TransformerXLDecoderLayer(nn.Module):

    def __init__(self, h, d_model, p, d_ff, attn_p=0.1, version=1.0, ignore_source=False,
                 variational=False, death_rate=0.0):
        super(MemoryTransformerDecoderLayer, self).__init__()
        self.version = version
        self.ignore_source = ignore_source
        self.variational = variational
        self.death_rate = death_rate

        self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model, p, sequence='da', variational=self.variational)

        self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, p, sequence='da', variational=self.variational)

        d_head = d_model // h
        self.multihead_tgt = RelPartialLearnableMultiHeadAttn(h, d_model, d_head, dropatt=attn_p)

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

    def forward(self, input_, context, pos_emb, mask_tgt, mask_src, mems=None,
                incremental=False, incremental_cache=None):
        # incremental=False, incremental_cache=None, reuse_source=True):

        """ Self attention layer with memory
            layernorm > attn > dropout > residual
        """
        assert context is None, "This model does not have an context encoder"

        coin = True
        if self.training and self.death_rate > 0:
            coin = (torch.rand(1)[0].item() >= self.death_rate)

        if coin:
            # input and context should be time first ?
            query = self.preprocess_attn(input_)

            if mems is not None and mems.size(0) > 0:
                mems = self.preprocess_attn(mems)
            else:
                mems = None

            # out, _ = self.multihead_tgt(query, pos_emb, r_w_bias, r_r_bias, attn_mask=mask_tgt)
            out, _, incremental_cache = self.multihead_tgt(query, pos_emb, attn_mask=mask_tgt,
                                                           incremental=incremental, incremental_cache=incremental_cache)

            # rescaling before residual
            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input_ = self.postprocess_attn(out, input_)

            """ Context Attention layer 
                layernorm > attn > dropout > residual
            """

            coverage = None

            """ Feed forward layer 
                layernorm > ffn > dropout > residual
            """
            out = self.feedforward(self.preprocess_ffn(input_))

            # rescaling before residual
            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input_ = self.postprocess_ffn(out, input_)
        else:
            coverage = None

        if incremental:
            return input_, coverage, incremental_cache

        return input_, coverage


class TransformerXL(TransformerDecoder):
    """
    This class combines the encoder and the decoder into one single sequence
    Joined attention between encoder and decoder parts
    """

    def __init__(self, opt, tgt_embedding, generator,
                 language_embeddings=None, **kwargs):
        self.tgt_embedding = tgt_embedding
        self.model_size = opt.model_size

        # build_modules will be called from the inherited constructor
        super(UnifiedTransformer, self).__init__(opt, tgt_embedding,
                                                 None,
                                                 language_embeddings=language_embeddings,
                                                 allocate_positions=False)

        self.generator = generator
        self.ignore_source = True
        self.d_head = self.model_size // self.n_heads

    def build_modules(self):
        e_length = expected_length(self.layers, self.death_rate)

        print("* Transformer Decoder with Absolute Attention with %.2f expected layers" % e_length)

        self.layer_modules = nn.ModuleList()

        for l in range(self.layers):
            # linearly decay the death rate
            death_r = (l + 1.0) / self.layers * self.death_rate

            block = TransformerXLDecoderLayer(self.n_heads, self.model_size,
                                              self.dropout, self.inner_size, self.attn_dropout,
                                              ignore_source=True,
                                              variational=self.variational_dropout, death_rate=death_r)
            self.layer_modules.append(block)

    def forward(self, batch, target_mask=None, **kwargs):

        tgt = batch.get('target_input')
        tgt_len = tgt.size(1)