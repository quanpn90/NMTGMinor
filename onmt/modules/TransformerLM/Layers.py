import math
import torch
import torch.nn as nn
import torch.nn.init as init
import onmt
import torch.nn.functional as F
from onmt.modules.Transformer.Layers import PrePostProcessing, MultiHeadAttention, Bottle, FeedForward
    
class LMDecoderLayer(nn.Module):
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
    
    def __init__(self, h, d_model, p, d_ff, attn_p=0.1, ):
        super(LMDecoderLayer, self).__init__()

        self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model, p, sequence='da', static=onmt.Constants.static)
        
        self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, p, sequence='da', static=onmt.Constants.static)
        
        
        self.multihead_tgt = MultiHeadAttention(h, d_model, attn_p=attn_p, static=onmt.Constants.static, share=1)

        ff_p = p
        feedforward = FeedForward(d_model, d_ff, ff_p, static=onmt.Constants.static)
        self.feedforward = Bottle(feedforward)
    
    def forward(self, input, mask_tgt):
        
        """ Self attention layer 
            layernorm > attn > dropout > residual
        """

        # input and context should be time first ?
        
        query = self.preprocess_attn(input)
        
        self_context = query
        
        out, _ = self.multihead_tgt(query, self_context, self_context, mask_tgt)

        input = self.postprocess_attn(out, input)
        
        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input))
        input = self.postprocess_ffn(out, input)

        coverage = None
        return input, coverage
        
    def step(self, input, mask_tgt, buffer=None):
        """ Self attention layer 
            layernorm > attn > dropout > residual
        """
        
        query = self.preprocess_attn(input)
        
        out, _, buffer = self.multihead_tgt.step(query, query, query, mask_tgt, buffer=buffer)

        input = self.postprocess_attn(out, input)


        coverage = None
        
        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input))

        input = self.postprocess_ffn(out, input)
        
        return input, coverage, buffer

