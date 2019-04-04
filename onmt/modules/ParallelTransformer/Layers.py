import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.utils.weight_norm as WeightNorm
import onmt 
import torch.nn.functional as F
from onmt.modules.Bottle import Bottle

from onmt.modules.Transformer.Layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing
from onmt.modules.StaticDropout import StaticDropout

Linear=XavierLinear

def contiguous(tensor):

    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()
        


class ParallelEncoderLayer(nn.Module):
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
    
    def __init__(self, h, d_model, p, d_ff, attn_p=0.1, version=1.0):
        super(ParallelEncoderLayer, self).__init__()
        self.version = version
        
        self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model, p, sequence='da', static=onmt.Constants.static)
        self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, p, sequence='da', static=onmt.Constants.static)
        self.multihead = MultiHeadAttention(h, d_model, attn_p=attn_p, static=onmt.Constants.static)
        
        if onmt.Constants.activation_layer == 'linear_relu_linear':
            ff_p = p
            feedforward = FeedForward(d_model, d_ff, ff_p)
        elif onmt.Constants.activation_layer == 'maxout':
            k = int(math.ceil(d_ff / d_model))
            feedforward = MaxOut(d_model, d_model, k)
        self.feedforward = Bottle(feedforward)
            
    def forward(self, input, attn_mask, pad_mask=None, residual_dropout=0.0):
        
        query = self.preprocess_attn(input)
        out, _ = self.multihead(query, query, query, attn_mask, 
                                query_mask=pad_mask, value_mask=pad_mask)
                                
        if residual_dropout > 0:
            input_ = F.dropout(input, residual_dropout, self.training, False)
            input = self.postprocess_attn(out, input_, mask=pad_mask)
            #~ input = self.postprocess_attn(out) + input
        else:
            input = self.postprocess_attn(out, input, mask=pad_mask)
        
        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input), 
                               mask=pad_mask)
        input = self.postprocess_ffn(out, input)
        
        # return the query which is the normalized input
        return input, query

#~ 
#~ class ParallelDecoderLayer(nn.Module):
    #~ """Wraps multi-head attentions and position-wise feed forward into one layer of decoder
    #~ 
    #~ Args:
        #~ h:       number of heads
        #~ d_model: dimension of model
        #~ p:       dropout probabolity 
        #~ d_ff:    dimension of feed forward
        #~ 
    #~ Params:
        #~ multihead_tgt:  multi-head self attentions layer
        #~ multihead_src:  multi-head encoder-decoder attentions layer        
        #~ feedforward:    feed forward layer
    #~ 
    #~ Input Shapes:
        #~ query:    batch_size x len_query x d_model 
        #~ key:      batch_size x len_key x d_model   
        #~ value:    batch_size x len_key x d_model
        #~ context:  batch_size x len_src x d_model
        #~ mask_tgt: batch_size x len_query x len_key or broadcastable 
        #~ mask_src: batch_size x len_query x len_src or broadcastable 
    #~ 
    #~ Output Shapes:
        #~ out:      batch_size x len_query x d_model
        #~ coverage: batch_size x len_query x len_key
        #~ 
    #~ """    
    #~ 
    #~ def __init__(self, h, d_model, p, d_ff, attn_p=0.1):
        #~ super(FCTDecoderLayer, self).__init__()
        #~ 
        #~ self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        #~ self.postprocess_attn = PrePostProcessing(d_model, p, sequence='da', static=True)
        #~ 
        #~ self.preprocess_src_attn = PrePostProcessing(d_model, p, sequence='n')
        #~ self.postprocess_src_attn = PrePostProcessing(d_model, p, sequence='da', static=True)
        #~ 
        #~ self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        #~ self.postprocess_ffn = PrePostProcessing(d_model, p, sequence='da', static=True)
        #~ 
        #~ 
        #~ self.multihead_tgt = HierarchicalMultiHeadAttention(h, d_model, attn_p=attn_p)
        #~ self.multihead_tgt = UniformMultiHeadAttention(h, d_model, attn_p=attn_p)
        #~ self.multihead_tgt = FlatSumMultiHeadAttention(h, d_model, attn_p=attn_p)
        #~ self.multihead_src = MultiHeadAttention(h, d_model, attn_p=attn_p)
        #~ self.multihead_src = UniformMultiHeadAttention(h, d_model, attn_p=attn_p)
        #~ self.multihead_src = FlatSumMultiHeadAttention(h, d_model, attn_p=attn_p)
        #~ 
        #~ if onmt.Constants.activation_layer == 'linear_relu_linear':
            #~ ff_p = p
            #~ feedforward = FeedForward(d_model, d_ff, ff_p)
        #~ elif onmt.Constants.activation_layer == 'maxout':
            #~ k = int(math.ceil(d_ff / d_model))
            #~ feedforward = MaxOut(d_model, d_model, k)
        #~ self.feedforward = Bottle(feedforward)
    #~ 
    #~ 
    #~ def forward(self, input, context, memory_bank, mask_tgt, mask_src, pad_mask_tgt=None, pad_mask_src=None):
        #~ 
        #~ """ Self attention layer 
            #~ layernorm > attn > dropout > residual
        #~ """
        #~ 
        #~ query = self.preprocess_attn(input, mask=pad_mask_tgt)
        #~ 
        #~ if memory_bank is None:
            #~ memory_bank = query.unsqueeze(0)
        #~ 
        #~ else:
            #~ memory_bank = query.unsqueeze(0)
            #~ memory_bank = torch.cat([memory_bank, query.unsqueeze(0)], dim=0) # n_layer x batch_size x len_src x hidden
        #~ 
        #~ 
        #~ out, _ = self.multihead_tgt(query, memory_bank, mask_tgt, 
                                    #~ query_mask=pad_mask_tgt, value_mask=pad_mask_tgt)
        #~ 
        #~ input = self.postprocess_attn(out, input)
        #~ 
        #~ """ Context Attention layer 
            #~ layernorm > attn > dropout > residual
        #~ """
        #~ 
        #~ query = self.preprocess_src_attn(input, mask=pad_mask_tgt)
        #~ out, coverage = self.multihead_src(query, context, mask_src, 
                                           #~ query_mask=pad_mask_tgt, value_mask=pad_mask_src)
        #~ input = self.postprocess_src_attn(out, input)
        #~ 
        #~ """ Feed forward layer 
            #~ layernorm > ffn > dropout > residual
        #~ """
        #~ out = self.feedforward(self.preprocess_ffn(input, mask=pad_mask_tgt), 
                                           #~ mask=pad_mask_tgt)
        #~ input = self.postprocess_ffn(out, input)
        #~ 
        #~ return input, memory_bank, coverage
    #~ 
    #~ 
    #~ def step(self, input, context, memory_bank, mask_tgt, mask_src, pad_mask_tgt=None, pad_mask_src=None, buffer=None):
        #~ 
        #~ query = self.preprocess_attn(input, mask=pad_mask_tgt)
        #~ 
        #~ if buffer is not None:
            #~ buffer = torch.cat([buffer, query], dim=1)
        #~ else:
            #~ buffer = query
        #~ 
        #~ if memory_bank is None:
            #~ memory_bank = buffer.unsqueeze(0)
        #~ 
        #~ else:
            #~ memory_bank = torch.cat([memory_bank, buffer.unsqueeze(0)], dim=0) # batch_size x n_layer x len_src x hidden
        #~ 
        #~ 
        #~ out, _ = self.multihead_tgt(query, memory_bank, mask_tgt, 
                                    #~ query_mask=None, value_mask=None)
        #~ 
        #~ input = self.postprocess_attn(out, input)
        #~ 
        #~ """ Context Attention layer 
            #~ layernorm > attn > dropout > residual
        #~ """
        #~ 
        #~ query = self.preprocess_src_attn(input, mask=pad_mask_tgt)
        #~ out, coverage = self.multihead_src(query, context, mask_src, 
                                           #~ query_mask=None, value_mask=None)
        #~ input = self.postprocess_src_attn(out, input)
        #~ 
        #~ """ Feed forward layer 
            #~ layernorm > ffn > dropout > residual
        #~ """
        #~ out = self.feedforward(self.preprocess_ffn(input, mask=pad_mask_tgt), 
                                           #~ mask=pad_mask_tgt)
        #~ input = self.postprocess_ffn(out, input)
        #~ 
        #~ return input, memory_bank, coverage, buffer
