import torch
import torch.nn as nn
from onmt.models.transformer_layers import PositionalEncoding, PrePostProcessing
from onmt.models.transformer_layers import EncoderLayer, DecoderLayer
from onmt.models.tranformers import TransformerEncoder, TransformerDecoder
import onmt
from onmt.modules.dropout import embedded_dropout
from onmt.models.transformer_layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing
from onmt.models.relative_transformer_layers import RelativeTransformerEncoderLayer, RelativeTransformerDecoderLayer
from onmt.utils import flip, expected_length
from collections import defaultdict
import math


class DistanceTransformerEncoder(TransformerEncoder):
    """
    Self-attention with learnable past and future relative positions (with embeddings)
    """

    def __init__(self, opt, dicts, positional_encoder, encoder_type='text'):
        self.death_rate = opt.death_rate
        self.double_position = opt.double_position

        # build_modules will be called from the inherited constructor
        super(RelativeTransformerEncoder, self).__init__(opt, dicts, positional_encoder, encoder_type)

        print("Encoder type: %s", encoder_type)
        self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)
        self.d_head = self.model_size // self.n_heads

        e_length = expected_length(self.layers, self.death_rate)
        # # Parameters for the position biases
        # self.r_w_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))
        # self.r_r_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))

        print("* Transformer Encoder with Relative Attention with %.2f expected layers" % e_length)