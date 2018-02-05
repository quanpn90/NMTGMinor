import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from onmt.modules.LayerNorm import LayerNorm
import torch.nn.init as init
import torch.nn.utils.weight_norm as WeightNorm
import onmt 

from onmt.modules.Transformer.Layers import XavierLinear, MultiHeadAttention, FeedForward


    
class StochasticEncoderLayer(nn.Module):
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
    
    def __init__(self, h, d_model, p, d_ff, attn_p=0.1, death_rate=0.):
        super(StochasticEncoderLayer, self).__init__()
        self.multihead = MultiHeadAttention(h, d_model, p, attn_p=attn_p)
        self.feedforward = FeedForward(d_model, d_ff, p)
        self.death_rate = death_rate
    def forward(self, query, key, value, mask):
        x = query
        residual = x
        if not self.training or torch.rand(1)[0] >= self.death_rate:
            residual, _ = self.multihead(residual, key, value, mask)
            residual = self.feedforward(residual)
            
            if self.training:
                residual /= (1. - self.death_rate)
            x = x + residual
        return x
    
    
class StochasticDecoderLayer(nn.Module):
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
    
    def __init__(self, h, d_model, p, d_ff, attn_p=0.1, death_rate=0.1):
        super(StochasticDecoderLayer, self).__init__()
        self.multihead_tgt = MultiHeadAttention(h, d_model, p, attn_p=attn_p)
        self.multihead_src = MultiHeadAttention(h, d_model, p, attn_p=attn_p)
        self.feedforward = FeedForward(d_model, d_ff, p)
        self.death_rate = death_rate
    
    def forward(self, query, key, value, context, mask_tgt, mask_src):
        x = query
        residual = x
        coverage = None
        if not self.training or torch.rand(1)[0] >= self.death_rate:
            residual, _ = self.multihead_tgt(query, key, value, mask_tgt)
            residual, coverage = self.multihead_src(residual, context, context, mask_src)
            residual = self.feedforward(residual)
            if self.training:
                residual /= (1. - self.death_rate)
            x = x + residual
        return x, coverage
