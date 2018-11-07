import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.utils.weight_norm as WeightNorm
import onmt 

from onmt.modules.Transformer.Layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing
from onmt.modules.Bottle import Bottle


class CompactMultiHeadAttention(nn.Module):
    """Applies multi-head attentions to inputs (query, key, value)
    Args:
        h:       number of heads
        d_model: dimension of model
        p:       dropout probabolity  
        
    Params:
        fc_query:  FC layer to project query, d_model x (h x d_head)
        fc_key:    FC layer to project key,   d_model x (h x d_head)
        fc_value:  FC layer to project value, d_model x (h x d_head)
        fc_concat: FC layer to concat and project multiheads, d_model x (h x d_head)
        
    Inputs Shapes: 
        query: batch_size x len_query x d_model 
        key:   batch_size x len_key x d_model   
        value: batch_size x len_key x d_model
        mask:  batch_size x len_query x len_key or broadcastable 
        
    Outputs Shapes:
        out:      batch_size x len_query x d_model
        coverage: batch_size x len_query x len_key
        
    """
    
    def __init__(self, h, d_model, attn_p=0.1, static=True):
        super(CompactMultiHeadAttention, self).__init__()      
        self.h = h
        self.d = d_model
        
        assert d_model % h == 0
        
        self.d_head = d_model//h
        self.fc_query = Bottle(Linear(d_model, h*self.d_head, bias=False))
        self.fc_key = Bottle(Linear(d_model, h*self.d_head, bias=False))
        self.fc_value = Bottle(Linear(d_model, h*self.d_head, bias=False))
        
        self.attention_out = onmt.Constants.attention_out
        self.fc_concat = Bottle(Linear(h*self.d_head, d_model, bias=False))

        self.sm = nn.Softmax(dim=-1)
        
        if static:
            self.attn_dropout = StaticDropout(attn_p)
        else:
            self.attn_dropout = nn.Dropout(attn_p)
        
      
    def _prepare_proj(self, x):
        """Reshape the projectons to apply softmax on each head
        """
        b, l, d = x.size()
        return contiguous(x.view(b, l, self.h, self.d_head).transpose(1,2)).view(b*self.h, l, self.d_head)
    
    def shape(self, x):
        
        b, l, d = x.size()
        return x.view(b, l, self.h, self.d_head) \
                .transpose(1, 2)


    def forward(self, query, key, value, mask, query_mask=None, value_mask=None):
        b, len_query = query.size(0), query.size(1)
        len_key = key.size(1)
        
        key_mask = value_mask
        
        # project inputs to multi-heads
        proj_query = self.fc_query(query, mask=query_mask)       # batch_size x len_query x h*d_head
        proj_key   = self.fc_key(key, mask=key_mask)             # batch_size x len_key x h*d_head
        proj_value = self.fc_value(value, mask=value_mask)       # batch_size x len_key x h*d_head
        
        # prepare the shape for applying softmax
        proj_query = self._prepare_proj(proj_query)  # batch_size*h x len_query x d_head
        proj_key = self._prepare_proj(proj_key)           # batch_size*h x len_key x d_head
        proj_value = self._prepare_proj(proj_value)       # batch_size*h x len_key x d_head
        
        proj_query = proj_query * (self.d_head**-0.5)
        
        # get dotproduct softmax attns for each head
        attns = torch.bmm(proj_query, proj_key.transpose(1,2))  # batch_size*h x len_query x len_key
        
        attns = attns.view(b, self.h, len_query, len_key) 
        if isinstance(mask, Variable):
            mask_ = mask.unsqueeze(-3)
        elif torch.is_tensor(mask):
            mask_ = Variable(mask.unsqueeze(-3))    
        #~ attns = attns.masked_fill_(mask_, -float('inf'))
        # FP16 support: cast to float and back
        attns = attns.float().masked_fill_(mask_, -float('inf')).type_as(attns)
        #~ attns = 
        #~ attns = attns.masked_fill_(mask_, -0)
        attns = self.sm(attns)
        # return mean attention from all heads as coverage 
        coverage = torch.mean(attns, dim=1) 
        attns = self.attn_dropout(attns)
        attns = attns.view(b*self.h, len_query, len_key)
        
        # apply attns on value
        out = torch.bmm(attns, proj_value)      # batch_size*h x len_query x d_head
        out = contiguous(out.view(b, self.h, len_query, self.d_head).transpose(1,2))
        out = out.view(b, len_query, self.h*self.d_head)
            
        out = self.fc_concat(out, mask=query_mask)
       
        return out, coverage

    
class CompactEncoderLayer(nn.Module):
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
        super(StochasticEncoderLayer, self).__init__()
        self.death_rate = death_rate
        
        self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model, p, sequence='da', static=onmt.Constants.static)
        self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, p, sequence='da', static=onmt.Constants.static)
        self.multihead = MultiHeadAttention(h, d_model, attn_p=attn_p, static=onmt.Constants.static)
        
        if onmt.Constants.activation_layer == 'linear_relu_linear':
            ff_p = p
            feedforward = FeedForward(d_model, d_ff, ff_p,static=onmt.Constants.static)
        elif onmt.Constants.activation_layer == 'maxout':
            k = int(math.ceil(d_ff / d_model))
            feedforward = MaxOut(d_model, d_model, k)
        self.feedforward = Bottle(feedforward)
            
    def forward(self, input, attn_mask, pad_mask=None):
        
        coin = True
        if self.training == True:
            coin = (torch.rand(1)[0].item() >= self.death_rate)
        
        if coin==True:
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
    
    def __init__(self, h, d_model, p, d_ff, attn_p=0.1, version=1.0, death_rate=0.0):
        super(StochasticDecoderLayer, self).__init__()
        self.death_rate = death_rate
        
        self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model,p, sequence='da', static=onmt.Constants.static)
        
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
        if self.training == True:
            coin = (torch.rand(1)[0].item() >= self.death_rate)
        
        if coin == True: 
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
        
    def step(self, input, context, mask_tgt, mask_src, pad_mask_tgt=None, pad_mask_src=None, buffer=None):
        """ Self attention layer 
            layernorm > attn > dropout > residual
        """
        #~ print(buffer)
        
        query = self.preprocess_attn(input, mask=pad_mask_tgt)
        
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
        
        #~ if self.training == False:
            #~ input = input * ( 1 - self.death-rate )
        
        return input, coverage, buffer
