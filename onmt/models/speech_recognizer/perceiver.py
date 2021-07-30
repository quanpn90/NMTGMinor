import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.models.transformer_layers import PositionalEncoding
from onmt.modules.pre_post_processing import PrePostProcessing
from onmt.models.transformers import TransformerEncoder, TransformerDecoder, Transformer, TransformerDecodingState
from onmt.modules.sinusoidal_positional_encoding import SinusoidalPositionalEmbedding
import onmt
from onmt.modules.base_seq2seq import NMTModel, Reconstructor, DecoderState
from onmt.modules.dropout import embedded_dropout
from .relative_transformer_layers import RelativeTransformerEncoderLayer, RelativeTransformerDecoderLayer
from onmt.utils import flip, expected_length
from collections import defaultdict
import math
import sys
from onmt.modules.checkpoint import checkpoint
# from torch.utils.checkpoint import checkpoint
from onmt.modules.identity import Identity

torch.set_printoptions(threshold=500000)


class SpeechTransformerEncoder(TransformerEncoder):

    def __init__(self, opt, dicts, positional_encoder, encoder_type='text', language_embeddings=None):
        self.death_rate = opt.death_rate
        self.learnable_position_encoding = opt.learnable_position_encoding
        self.layer_modules = list()
        self.asynchronous = opt.asynchronous
        self.max_memory_size = opt.max_memory_size
        self.extra_context_size = opt.extra_context_size
        self.experimental = opt.experimental
        self.unidirectional = opt.unidirectional
        self.reversible = opt.src_reversible
        self.n_heads = opt.n_heads
        self.fast_self_attn = opt.fast_self_attention
        self.checkpointing = opt.checkpointing
        self.mpw = opt.multilingual_partitioned_weights
        self.multilingual_linear_projection = opt.multilingual_linear_projection
        self.mln = opt.multilingual_layer_norm
        self.no_input_scale = opt.no_input_scale
        self.learnable_position_encoding = opt.learnable_position_encoding
        self.max_pos_length = opt.max_pos_length

        # TODO: multilingually linear transformation

        # build_modules will be called from the inherited constructor
        super().__init__(opt, dicts, positional_encoder, encoder_type, language_embeddings)

        # learnable position encoding
        if self.learnable_position_encoding:
            # raise NotImplementedError
            self.positional_encoder = None
        else:
            # or using pre-set sinusoidal
            self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)

        self.d_head = self.model_size // self.n_heads

        if self.multilingual_linear_projection:
            self.linear_proj = nn.Parameter(torch.Tensor(opt.n_languages, self.model_size, self.model_size))

            std_ = math.sqrt(2.0 / (self.model_size + self.model_size))
            torch.nn.init.normal_(self.linear_proj, 0.0, std_)

        self.mln = opt.multilingual_layer_norm

        if not opt.rezero:
            self.postprocess_layer = PrePostProcessing(opt.model_size, opt.dropout, sequence='n', multilingual=self.mln,
                                                       n_languages=opt.n_languages)
        else:
            self.postprocess_layer = Identity()

        