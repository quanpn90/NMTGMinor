import numpy as np
import torch, math
import torch.nn as nn
from onmt.modules.Transformer.Layers import PositionalEncoding
from onmt.modules.Transformer.Layers import EncoderLayer, DecoderLayer
from onmt.modules.StochasticTransformer.Layers import StochasticEncoderLayer, StochasticDecoderLayer
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder
from onmt.modules.BaseModel import NMTModel, Reconstructor
import onmt
from onmt.modules.WordDrop import embedded_dropout
from onmt.modules.Checkpoint import checkpoint

from onmt.modules.Transformer.Layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing


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
    
    def __init__(self, opt, dicts, positional_encoder):

        self.death_rate = opt.death_rate
        
        # build_modules will be called from the inherited constructor
        super(StochasticTransformerEncoder, self).__init__(opt, dicts, positional_encoder)

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
    
    def __init__(self, opt, dicts, positional_encoder):

        self.death_rate = opt.death_rate
        
        # build_modules will be called from the inherited constructor
        super(StochasticTransformerDecoder, self).__init__(opt, dicts, positional_encoder)
        
        e_length = expected_length(self.layers, self.death_rate)
        
        print("Stochastic Decoder with %.2f expected layers" % e_length)
        

    def build_modules(self):
        
        self.layer_modules = nn.ModuleList()

        for l in range(self.layers):
            
            # linearly decay the death rate
            death_r = ( l + 1 ) / self.layers * self.death_rate
            
            block = StochasticDecoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, self.attn_dropout, death_rate=death_r)
            
            self.layer_modules.append(block)


# class StochasticTransformer(NMTModel):
#     """Main model in 'Attention is all you need' """
#
#
#     def forward(self, input):
#         """
#         Inputs Shapes:
#             src: len_src x batch_size
#             tgt: len_tgt x batch_size
#
#         Outputs Shapes:
#             out:      batch_size*len_tgt x model_size
#
#
#         """
#         src = input[0]
#         tgt = input[1][:-1]  # exclude last target from inputs
#
#         src = src.transpose(0, 1) # transpose to have batch first
#         tgt = tgt.transpose(0, 1)
#
#         context, src_mask = self.encoder(src)
#
#         output, coverage = self.decoder(tgt, context, src)
#
#         output = output.transpose(0, 1) # transpose to have time first, like RNN models
#
#         return output
#
#
#     def create_decoder_state(self, src, context, beamSize=1):
#
#         from onmt.modules.TransformerLM.Models import TransformerDecodingState
#
#         decoder_state = TransformerDecodingState(src, context, beamSize=beamSize)
#         return decoder_state
