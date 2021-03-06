import numpy as np
import torch, math
import torch.nn as nn
import onmt
from onmt.models.transformer_layers import PositionalEncoding
from onmt.models.transformer_layers import EncoderLayer, DecoderLayer
from onmt.legacy.stochastic_transformer_layers import StochasticEncoderLayer, StochasticDecoderLayer
from onmt.models.transformers import TransformerEncoder, TransformerDecoder
from onmt.modules.base_seq2seq import NMTModel, Reconstructor
from onmt.modules.dropout import embedded_dropout
from onmt.models.transformer_layers import XavierLinear, MultiHeadAttention, PrePostProcessing
from onmt.modules.linear import FeedForward, FeedForwardSwish


def custom_layer(module):
    def custom_forward(*args):
        output = module(*args)
        return output
    return custom_forward


def expected_length(length, death_rate):
    
    e_length = 0
    
    for l in range(length):
        survival_rate = 1.0 - (l+1)/length*death_rate
        
        e_length += survival_rate
        
    return e_length


class StochasticTransformerEncoder(TransformerEncoder):
    """Encoder in 'Attention is all you need'
    
    Args:
        opt: list of options ( see train.py )
        dicts : dictionary (for source language)
        
    """
    
    def __init__(self, opt, dicts, positional_encoder, encoder_type='text'):

        self.death_rate = opt.death_rate
        
        # build_modules will be called from the inherited constructor
        super(StochasticTransformerEncoder, self).__init__(opt, dicts, positional_encoder, encoder_type)

        e_length = expected_length(self.layers, self.death_rate)
        
        print("Stochastic Encoder with %.2f expected layers" % e_length)

    def build_modules(self):

        self.layer_modules = nn.ModuleList()

        for l in range(self.layers):
            
            # linearly decay the death rate
            death_r = ( l + 1.0 ) / self.layers * self.death_rate
            
            block = StochasticEncoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, self.attn_dropout, death_rate=death_r)
            
            self.layer_modules.append(block)


class StochasticTransformerDecoder(TransformerDecoder):
    """Encoder in 'Attention is all you need'
    
    Args:
        opt
        dicts 
        
        
    """
    
    def __init__(self, opt, dicts, positional_encoder, attribute_embeddings=None, ignore_source=False):

        self.death_rate = opt.death_rate
        
        # build_modules will be called from the inherited constructor
        super(StochasticTransformerDecoder, self).__init__(opt, dicts,
                                                           positional_encoder,
                                                           attribute_embeddings,
                                                           ignore_source)
        
        e_length = expected_length(self.layers, self.death_rate)
        
        print("Stochastic Decoder with %.2f expected layers" % e_length)

    def build_modules(self):
        
        self.layer_modules = nn.ModuleList()

        for l in range(self.layers):
            
            # linearly decay the death rate
            death_r = ( l + 1 ) / self.layers * self.death_rate
            
            block = StochasticDecoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, self.attn_dropout, death_rate=death_r)
            
            self.layer_modules.append(block)

