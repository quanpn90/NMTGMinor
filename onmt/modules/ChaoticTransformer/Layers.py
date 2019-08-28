import math
import torch
import torch.nn as nn
import onmt

from onmt.modules.Transformer.Layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing, EncoderLayer, DecoderLayer
from onmt.modules.Bottle import Bottle


class StochasticEncoderLayer(EncoderLayer):
    """Wraps multi-head attentions and position-wise feed forward into one encoder layer
    
    Args:
        h:       number of heads
        d_model: dimension of model
        p:       dropout probabolity 
        d_ff:    dimension of feed forward
        
    Params:
        multihead:    multi-head attentions layer
        feedforward:  feed forward layer
    
    Input Shapes:
        query: batch_size x len_query x d_model 
        key:   batch_size x len_key x d_model   
        value: batch_size x len_key x d_model
        mask:  batch_size x len_query x len_key or broadcastable 
    
    Output Shapes:
        out: batch_size x len_query x d_model
    """
    
    def __init__(self, h, d_model, p, d_ff, attn_p=0.1, version=1.0, death_rate=0.0):
        super().__init__(h, d_model, p, d_ff, attn_p, version)
        #~ super(StochasticEncoderLayer, self).__init__()
        self.death_rate = death_rate
        
            
    def forward(self, input, attn_mask, pad_mask=None):
        
        coin = True
        if self.training:
            coin = (torch.rand(1)[0].item() >= self.death_rate)
        
        if coin:
            query = self.preprocess_attn(input)
            out, _ = self.multihead(query, query, query, attn_mask)
            
            if self.training:
                out = out / ( 1 - self.death_rate)
            
            input = self.postprocess_attn(out, input)

            """ Feed forward layer 
                layernorm > ffn > dropout > residual
            """
            out = self.feedforward(self.preprocess_ffn(input), 
                                   mask=pad_mask)

            if self.training:
                out = out / ( 1 - self.death_rate)

            input = self.postprocess_ffn(out, input)
        
        return input
    
    
class StochasticDecoderLayer(DecoderLayer):
    """Wraps multi-head attentions and position-wise feed forward into one layer of decoder
    
    Args:
        h:       number of heads
        d_model: dimension of model
        p:       dropout probabolity 
        d_ff:    dimension of feed forward
        
    Params:
        multihead_tgt:  multi-head self attentions layer
        multihead_src:  multi-head encoder-decoder attentions layer        
        feedforward:    feed forward layer
    
    Input Shapes:
        query:    batch_size x len_query x d_model 
        key:      batch_size x len_key x d_model   
        value:    batch_size x len_key x d_model
        context:  batch_size x len_src x d_model
        mask_tgt: batch_size x len_query x len_key or broadcastable 
        mask_src: batch_size x len_query x len_src or broadcastable 
    
    Output Shapes:
        out:      batch_size x len_query x d_model
        coverage: batch_size x len_query x len_key
        
    """    
    
    def __init__(self, h, d_model, p, d_ff, attn_p=0.1, version=1.0, death_rate=0.0):
        super().__init__(h, d_model, p, d_ff, attn_p, version)
        self.death_rate = death_rate
        
    
    def forward(self, input, context, mask_tgt, mask_src, pad_mask_tgt=None, pad_mask_src=None, residual_dropout=0.0):
        
        """ Self attention layer 
            layernorm > attn > dropout > residual
        """
        
        """
            input is 'unnormalized' so the first preprocess layer is to normalize it before attention
            
            output (input after stacked with other outputs) is also unnormalized (to be normalized in the next layer)
            
            so if we skip the layer and propagate input forward:

        """
        coverage = None
        
        coin = True
        if self.training:
            coin = (torch.rand(1)[0].item() >= self.death_rate)
        
        if coin:
            query = self.preprocess_attn(input)
            
            self_context = query
            
            out, _ = self.multihead_tgt(query, self_context, self_context, mask_tgt, 
                                        query_mask=pad_mask_tgt, value_mask=pad_mask_tgt)
            
            if self.training:
                out = out / ( 1 - self.death_rate)
            
            input = self.postprocess_attn(out, input)


            """ Context Attention layer 
                layernorm > attn > dropout > residual
            """
            query = self.preprocess_src_attn(input, mask=pad_mask_tgt)
            out, coverage = self.multihead_src(query, context, context, mask_src, 
                                               query_mask=pad_mask_tgt, value_mask=pad_mask_src)
            
            if self.training:
                out = out / ( 1 - self.death_rate)
            
            input = self.postprocess_src_attn(out, input)


            
            """ Feed forward layer 
                layernorm > ffn > dropout > residual
            """
            out = self.feedforward(self.preprocess_ffn(input, mask=pad_mask_tgt), 
                                               mask=pad_mask_tgt)
            # During testing we scale the output to match its participation during training                                   
            if self.training:
                out = out / ( 1 - self.death_rate)
                
            input = self.postprocess_ffn(out, input)
            
 
    
        return input, coverage
