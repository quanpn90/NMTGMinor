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
        


class UniversalEncoderLayer(nn.Module):
    """Wraps multi-head attentions and position-wise feed forward into one encoder layer
    
    Args:
        h:       number of heads
        d_model: dimension of model
        p:       dropout probabolity 
        d_ff:    dimension of feed forward
        position encoder: adding embedding based on position
        time encoder: adding embedding based on time (the loop)
        
        
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
    
    def __init__(self, h, d_model, p, d_ff, pos_encoder, time_encoder, attn_p=0.1, version=1.0):
        super(UniversalEncoderLayer, self).__init__()
        self.version = version
        # position and time embedding is added into the input before the layer
        self.pos_encoder = pos_encoder
        self.time_encoder = time_encoder
        
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
            
    def forward(self, input, attn_mask, t, pad_mask=None):
            
        # apply layer normalization
        query = self.preprocess_attn(input)
        
        # add position encoding and time encoding
        query = self.pos_encoder(query) + self.time_encoder(t)
        
        out, _ = self.multihead(query, query, query, attn_mask, 
                                query_mask=pad_mask, value_mask=pad_mask)
                                
       
        input = self.postprocess_attn(out, input, mask=pad_mask)
        
        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input), 
                               mask=pad_mask)
        input = self.postprocess_ffn(out, input)
        
        return input


class UniversalDecoderLayer(nn.Module):
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
    
    def __init__(self, h, d_model, p, d_ff, position_encoder, time_encoder, attn_p=0.1, version=1.0):
        super(UniversalDecoderLayer, self).__init__()
        self.version = version
        self.position_encoder = position_encoder
        self.time_encoder = time_encoder
        
        self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model, p, sequence='da', static=onmt.Constants.static)
        
        self.preprocess_src_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_src_attn = PrePostProcessing(d_model, p, sequence='da', static=onmt.Constants.static)
        
        self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, p, sequence='da', static=onmt.Constants.static)
        
        
        self.multihead_tgt = MultiHeadAttention(h, d_model, attn_p=attn_p, static=onmt.Constants.static)
        self.multihead_src = MultiHeadAttention(h, d_model, attn_p=attn_p, static=onmt.Constants.static)
        
        if onmt.Constants.activation_layer == 'linear_relu_linear':
            ff_p = p
            feedforward = FeedForward(d_model, d_ff, ff_p, static=onmt.Constants.static)
        elif onmt.Constants.activation_layer == 'maxout':
            k = int(math.ceil(d_ff / d_model))
            feedforward = MaxOut(d_model, d_model, k)
        self.feedforward = Bottle(feedforward)
    
    def forward(self, input, context, t, mask_tgt, mask_src, pad_mask_tgt=None, pad_mask_src=None):
        
        """ Self attention layer 
            layernorm > attn > dropout > residual
        """
        #~ print(input.size())
        #~ print(context.size())
        #~ print(pad_mask_tgt.size())
        
        query = self.preprocess_attn(input)
        
        # add position encoding and time encoding
        query = self.position_encoder(query) + self.time_encoder(t)
        
        self_context = query
        
        out, _ = self.multihead_tgt(query, self_context, self_context, mask_tgt, 
                                    query_mask=pad_mask_tgt, value_mask=pad_mask_tgt)
        
       
        input = self.postprocess_attn(out, input)
        
        
        
        """ Context Attention layer 
            layernorm > attn > dropout > residual
        """
        
        query = self.preprocess_src_attn(input, mask=pad_mask_tgt)
        out, coverage = self.multihead_src(query, context, context, mask_src, 
                                           query_mask=pad_mask_tgt, value_mask=pad_mask_src)
        input = self.postprocess_src_attn(out, input)
        
        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input, mask=pad_mask_tgt), 
                                           mask=pad_mask_tgt)
        input = self.postprocess_ffn(out, input)
    
        return input, coverage
        
    def step(self, input, context, pos_step, t, mask_tgt,  mask_src, pad_mask_tgt=None, pad_mask_src=None, buffer=None):
        """ Self attention layer 
            layernorm > attn > dropout > residual
        """
        
        
        query = self.preprocess_attn(input, mask=pad_mask_tgt)
        
        # add position encoding and time encoding (before the buffer because the previous steps are already added)
        query = self.position_encoder(query, t=pos_step) + self.time_encoder(t)
        
        if buffer is not None:
            buffer = torch.cat([buffer, query], dim=1)
        else:
            buffer = query
            

        out, _ = self.multihead_tgt(query, buffer, buffer, mask_tgt, 
                                    query_mask=pad_mask_tgt, value_mask=pad_mask_tgt)
        

        input = self.postprocess_attn(out, input)
        
        """ Context Attention layer 
            layernorm > attn > dropout > residual
        """
        
        
        query = self.preprocess_src_attn(input, mask=pad_mask_tgt)
        out, coverage = self.multihead_src(query, context, context, mask_src, 
                                           query_mask=pad_mask_tgt, value_mask=None)
        input = self.postprocess_src_attn(out, input)
        
        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input, mask=pad_mask_tgt), 
                                           mask=pad_mask_tgt)
        input = self.postprocess_ffn(out, input)
        
        return input, coverage, buffer



class TimeEncoding(nn.Module):
    """Adds positional embeddings to standard word embeddings 
    This matches the original TensorFlow implementation at https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py.
    
    Args:
        d_model: dimension of model
        p:       dropout probability  
        len_max: max seq length for pre-calculated positional embeddings
        
    Inputs Shapes: 
        word_emb: batch_size x len_seq x d_model 
        
    Outputs Shapes:
        out:   batch_size x len_seq x d_model
        
    """
    def __init__(self, d_model, p=0, len_max=64):
        # save a fixed positional embedding matrix up to len_max,
        # so that no need to recreate it everytime
        super(TimeEncoding , self).__init__()
        self.len_max=len_max
        self.d_model = d_model
        
        self.renew(len_max)
        
        self.p = p
    
    def renew(self, new_max_len):
        
        ## detele the old variable to avoid Pytorch's error when register new buffer
        if hasattr(self, 'time_emb'):
            del self.time_emb
        times = torch.arange(0,new_max_len).float()                      
        num_timescales = self.d_model // 2
        log_timescale_increment = math.log(10000) / (num_timescales-1)
        inv_timescales = torch.exp(torch.arange(0, num_timescales).float() * -log_timescale_increment)
        scaled_time = times.unsqueeze(1) * inv_timescales.unsqueeze(0)
        time_emb = torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), 1)
        # wrap in a buffer so that model can be moved to GPU
        self.register_buffer('time_emb', time_emb)

        
    def forward(self, t):
        
        # print('hello')
        # out = word_emb + Variable(self.pos_emb[:len_seq, :][-1, :], requires_grad=False)
        time_emb = Variable(self.time_emb[t, :], requires_grad=False) # 1 x dim
        # out should have size 1 x 1 x dim
        # all positions share the time embedding 
        # all batch elements share the time embedding
        out = time_emb.unsqueeze(0)
        return out
