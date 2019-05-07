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
        
class UniformMultiHeadAttention(nn.Module):
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
    
    def __init__(self, h, d_model, attn_p=0.1):
        super(UniformMultiHeadAttention, self).__init__()      
        self.h = h
        self.d = d_model
        
        assert d_model % h == 0
        
        self.d_head = d_model//h
        
        # first attention layer for states
        self.fc_query = Bottle(Linear(d_model, h*self.d_head, bias=False))
        self.fc_key = Bottle(Linear(d_model, h*self.d_head, bias=False))
        self.fc_value = Bottle(Linear(d_model, h*self.d_head, bias=False))
        
        
        # second attention for layers
        #~ self.fc_query_2 = Bottle(Linear(d_model, h*self.d_head, bias=False))
        #~ self.fc_key_2 = Bottle(Linear(d_model, h*self.d_head, bias=False))
        #~ self.fc_value_2 = Bottle(Linear(d_model, h*self.d_head, bias=False))
        
        # for output
        self.sm = nn.Softmax(dim=-1)
        self.fc_concat = Bottle(Linear(h*self.d_head, d_model, bias=False))
        #~ self.fc_concat_2 = Bottle(Linear(d_model, d_model, bias=False))
        
        #~ self.attn_dropout = nn.Dropout(attn_p)
        
        self.attn_dropout = StaticDropout(attn_p)
        #~ self.attn_dropout_2 = StaticDropout(attn_p)
        
      
    def _prepare_proj(self, x):
        """Reshape the projectons to apply softmax on each head
        """
        b, l, d = x.size()
        return contiguous(x.view(b, l, self.h, self.d_head).transpose(1,2)).view(b*self.h, l, self.d_head)
        
    def shape(self, x):
        
        b, l, d = x.size()
        return x.view(b, l, self.h, self.d_head) \
                .transpose(1, 2)
    
        
    def forward(self, query, key, mask=None, query_mask=None, value_mask=None):
        
        n_layer, b, len_key = key.size(0), key.size(1), key.size(2)
        if value_mask is not None:
            value_mask = value_mask.unsqueeze(0).repeat(n_layer, 1, 1)
        
        key_mask = value_mask # B x T 
        
        b, len_query = query.size(0), query.size(1)
        
        
        value = key        
        # project inputs to multi-heads
        proj_query = self.fc_query(query, mask=query_mask)       # batch_size x len_query x h*d_head
        proj_key   = self.fc_key(key, mask=key_mask).transpose(0,1).contiguous().view(b, -1, self.h * self.d_head)            # batch_size x (n_layer x len_key) x h*d_head
        proj_value = self.fc_value(value, mask=value_mask).transpose(0,1).contiguous().view(b, -1, self.h * self.d_head)        # batch_size x (n_layer x len_key) x h*d_head
        
        # prepare the shape for applying softmax
        proj_query = self.shape(proj_query)  # batch_size x h x len_query x d_head
        proj_key = self.shape(proj_key)           # batch_size x h x (n_layer * len_key) x d_head
        proj_value = self.shape(proj_value)       # batch_size x h x (n_layer * len_key) x d_head
        
        proj_query = proj_query * (self.d_head**-0.5)
        
        # get dotproduct softmax attns for each head
        scores = torch.matmul(proj_query, proj_key.transpose(2,3)) # b x self.h x len_query x n_layer*len_key
        
        # applying mask using broadcasting
        mask_ = Variable(mask.unsqueeze(-3).unsqueeze(-2))
        
        scores = scores.view(b, self.h, len_query, n_layer, len_key)
        scores = scores.masked_fill_(mask_, -float('inf'))
        scores = scores.view(b, self.h, len_query, n_layer*len_key)
        
        
        # softmax on the last dimension (all of the previous states)
        attns = self.sm(scores) # b x 1 x len_query x n_layer*lenkey      
        
        attns = self.attn_dropout(attns)
        
        
        out = torch.matmul(attns, proj_value) # b x self.h x len_query x self.d_head)
        
        out = out.transpose(1, 2).contiguous().view(b, len_query, self.h * self.d_head)
                
        out = self.fc_concat(out, mask=query_mask)
            
        #~ out = final_out.view(b, len_query, self.h*self.d_head)
            
        coverage = None
        
        return out, coverage
    

class HierarchicalMultiHeadAttention(nn.Module):
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
    
    def __init__(self, h, d_model, attn_p=0.1):
        super(HierarchicalMultiHeadAttention, self).__init__()      
        self.h = h
        self.d = d_model
        
        assert d_model % h == 0
        
        self.d_head = d_model//h
        
        # first attention layer for states
        self.fc_query = Bottle(Linear(d_model, h*self.d_head, bias=False))
        self.fc_key = Bottle(Linear(d_model, h*self.d_head, bias=False))
        self.fc_value = Bottle(Linear(d_model, h*self.d_head, bias=False))
        
        
        # second attention for layers
        self.fc_query_2 = Bottle(Linear(d_model, h*self.d_head, bias=False))
        #~ self.fc_key_2 = Bottle(Linear(d_model, h*self.d_head, bias=False))
        #~ self.fc_value_2 = Bottle(Linear(d_model, h*self.d_head, bias=False))
        
        # for output
        self.fc_concat = Bottle(Linear(h*self.d_head, d_model, bias=False))
        self.fc_concat_2 = Bottle(Linear(d_model, d_model, bias=False))
        

        self.sm = nn.Softmax(dim=-1)
        self.sm_2 = nn.Softmax(dim=-1)
        #~ self.attn_dropout = nn.Dropout(attn_p)
        
        self.attn_dropout = StaticDropout(attn_p)
        self.attn_dropout_2 = StaticDropout(attn_p)
        
      
    def _prepare_proj(self, x):
        """Reshape the projectons to apply softmax on each head
        """
        b, l, d = x.size()
        return contiguous(x.view(b, l, self.h, self.d_head).transpose(1,2)).view(b*self.h, l, self.d_head)
        
    def shape(self, x):
        
        b, l, d = x.size()
        return x.view(b, l, self.h, self.d_head) \
                .transpose(1, 2)
        
    def forward(self, query, key, mask=None, query_mask=None, value_mask=None):
        
        n_layer, b, len_key = key.size(0), key.size(1), key.size(2)
        #~ query_mask = None
        #~ value_mask = None
        if value_mask is not None:
            value_mask = value_mask.unsqueeze(0).repeat(n_layer, 1, 1)
        
        key_mask = value_mask # n_layer x B x T 
        
        b, len_query = query.size(0), query.size(1)
        
        #~ key = key.transpose(0,1).contiguous().view(b, n_layer * len_key, -1)
        
        value = key
        # FIRST ATTENTION STEP 
        
        # project inputs to multi-heads
        proj_query = self.fc_query(query, mask=query_mask)       # batch_size x len_query x h*d_head
        proj_key   = self.fc_key(key, mask=key_mask).transpose(0,1).contiguous().view(b, -1, self.h * self.d_head)            # batch_size x (n_layer x len_key) x h*d_head
        proj_value = self.fc_value(value, mask=value_mask).transpose(0,1).contiguous().view(b, -1, self.h * self.d_head)        # batch_size x (n_layer x len_key) x h*d_head
        
        # prepare the shape for applying softmax
        proj_query = self.shape(proj_query)  # batch_size x h x len_query x d_head
        proj_key = self.shape(proj_key)           # batch_size x h x (n_layer * len_key) x d_head
        proj_value = self.shape(proj_value)       # batch_size x h x (n_layer * len_key) x d_head
        
        proj_query = proj_query * (self.d_head**-0.5)
        
        # get dotproduct softmax attns for each head
        scores = torch.matmul(proj_query, proj_key.transpose(2,3)) # b x self.h x len_query x n_layer*len_key
        
        # unshape to softmax on only the len_key dimension
        scores = scores.view(b, self.h, len_query, n_layer, len_key)
        
        mask_ = Variable(mask.unsqueeze(1).unsqueeze(-2)) # b x 1 x len_query x 1 x len_key
        #~ mask_ = Variable(mask.unsqueeze(-3))        
        scores = scores.masked_fill_(mask_, -float('inf'))
        
        # softmax on the last dimension (len_key)
        #~ attns = self.sm(scores) # b x self.h x len_query x n_layer x len_key
        attns = F.softmax(scores, dim=-1)
        
        attns = self.attn_dropout(attns)
        
        # apply attns on value
        proj_value = proj_value.view(b, self.h, n_layer, len_key, self.d_head)
        
        attns = attns.transpose(2, 3) # b, self.h, n_layer, len_query, len_key
        
        out = torch.matmul(attns, proj_value) # b x self.h x n_layer x len_query x self.d_head
        
        out = out.transpose(1, 3).contiguous().view(b, len_query, n_layer, self.h * self.d_head)
        
        out = self.fc_concat(out, query_mask.unsqueeze(-1).repeat(1, 1, n_layer))
        
        # 2ND ATTENTION LAYER

        new_query = self.fc_query_2(query, mask=query_mask)  
        new_query = new_query.view(-1, new_query.size(-1)).unsqueeze(1) # batch_size*len_query x 1 x h*d_head
        proj_query = self.shape(new_query) # batch_size*len_query x h x 1 x d_head
        
        
        new_key = out.view(-1, n_layer, self.h * self.d_head) # b*len_query x n_layer x h*self.d_head
        proj_key = self.shape(new_key) # batch_size*len_query x h x n_layer x d_head
        
        if query_mask is not None:
            flattened_mask = query_mask.view(-1)
            
            non_pad_indices = torch.nonzero(flattened_mask).squeeze(1)
            
           
            proj_query = proj_query.index_select(0, non_pad_indices)
            proj_key = proj_key.index_select(0, non_pad_indices)
            
        
        proj_value = proj_key
        
        scores_2 = torch.matmul(proj_query, proj_key.transpose(2,3)) # batch_size*len_query x h x 1 x n_layer
        
        # no need to mask this time 
        attns_2 = F.softmax(scores_2, dim=-1) # batch_size*len_query x h x 1 x n_layer
        #~ attns_2 = self.attn_dropout(attns_2)
         
        out = torch.matmul(attns_2, proj_value) # batch_size*len_query x h x 1 x d_head
        
        b_ = out.size(0)        
        
        #~ out = out.transpose(1, 2).unsqueeze(1).contiguous().view(b_, self.h * self.d_head) # batch_size x len_query x h*d_head
        out = out.unsqueeze(2).view(-1, self.h * self.d_head)
        
        out = self.fc_concat_2(out)
        
        if query_mask is not None:
            
            final_out = Variable(out.data.new(b*len_query, self.h * self.d_head).zero_())
            
            final_out.index_copy_(0, non_pad_indices, out)
            
        else:
            final_out = out
            
        out = final_out.view(b, len_query, self.h*self.d_head)
            
        coverage = None
        
        return out, coverage


class FCTEncoderLayer(nn.Module):
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
    
    def __init__(self, h, d_model, p, d_ff, attn_p=0.1):
        super(FCTEncoderLayer, self).__init__()
       
       
        self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model, p, sequence='da', static=True)
        #~ self.multihead = HierarchicalMultiHeadAttention(h, d_model, attn_p=attn_p)
        self.multihead = UniformMultiHeadAttention(h, d_model, attn_p=attn_p)
        
        
        self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, p, sequence='da', static=True)
        
        if onmt.Constants.activation_layer == 'linear_relu_linear':
            ff_p = p
            feedforward = FeedForward(d_model, d_ff, ff_p)
        elif onmt.Constants.activation_layer == 'maxout':
            k = int(math.ceil(d_ff / d_model))
            feedforward = MaxOut(d_model, d_model, k)
        self.feedforward = Bottle(feedforward)
        
    
    def forward(self, input, memory_bank, attn_mask, pad_mask=None):
        
        query = self.preprocess_attn(input)
        
        if memory_bank is None:
            memory_bank = query.unsqueeze(0)
        
        else:
            #~ memory_bank = query.unsqueeze(0)
            memory_bank = torch.cat([memory_bank, query.unsqueeze(0)], dim=0) # batch_size x n_layer x len_src x hidden
        """ Deep attention layer """
        
        out, _ = self.multihead(query, memory_bank, attn_mask, 
                                query_mask=pad_mask, value_mask=pad_mask)
        input = self.postprocess_attn(out, input, mask=pad_mask)
        
        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input, mask=pad_mask), 
                               mask=pad_mask)
        input = self.postprocess_ffn(out, input, mask=pad_mask)
        
        return input, memory_bank


class FCTDecoderLayer(nn.Module):
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
    
    def __init__(self, h, d_model, p, d_ff, attn_p=0.1):
        super(FCTDecoderLayer, self).__init__()
        
        self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model, p, sequence='da', static=True)
        
        self.preprocess_src_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_src_attn = PrePostProcessing(d_model, p, sequence='da', static=True)
        
        self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, p, sequence='da', static=True)
        
        
        #~ self.multihead_tgt = HierarchicalMultiHeadAttention(h, d_model, attn_p=attn_p)
        self.multihead_tgt = UniformMultiHeadAttention(h, d_model, attn_p=attn_p)
        #~ self.multihead_src = MultiHeadAttention(h, d_model, attn_p=attn_p)
        self.multihead_src = UniformMultiHeadAttention(h, d_model, attn_p=attn_p)
        
        if onmt.Constants.activation_layer == 'linear_relu_linear':
            ff_p = p
            feedforward = FeedForward(d_model, d_ff, ff_p)
        elif onmt.Constants.activation_layer == 'maxout':
            k = int(math.ceil(d_ff / d_model))
            feedforward = MaxOut(d_model, d_model, k)
        self.feedforward = Bottle(feedforward)
    
    
    def forward(self, input, context, memory_bank, mask_tgt, mask_src, pad_mask_tgt=None, pad_mask_src=None):
        
        """ Self attention layer 
            layernorm > attn > dropout > residual
        """
        
        query = self.preprocess_attn(input, mask=pad_mask_tgt)
        
        if memory_bank is None:
            memory_bank = query.unsqueeze(0)
        
        else:
            #~ memory_bank = query.unsqueeze(0)
            memory_bank = torch.cat([memory_bank, query.unsqueeze(0)], dim=0) # n_layer x batch_size x len_src x hidden
        
        
        out, _ = self.multihead_tgt(query, memory_bank, mask_tgt, 
                                    query_mask=pad_mask_tgt, value_mask=pad_mask_tgt)
        
        input = self.postprocess_attn(out, input)
        
        """ Context Attention layer 
            layernorm > attn > dropout > residual
        """
        
        query = self.preprocess_src_attn(input, mask=pad_mask_tgt)
        out, coverage = self.multihead_src(query, context, mask_src, 
                                           query_mask=pad_mask_tgt, value_mask=pad_mask_src)
        input = self.postprocess_src_attn(out, input)
        
        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input, mask=pad_mask_tgt), 
                                           mask=pad_mask_tgt)
        input = self.postprocess_ffn(out, input)
        
        return input, memory_bank, coverage
    
    
    def step(self, input, context, memory_bank, mask_tgt, mask_src, pad_mask_tgt=None, pad_mask_src=None, buffer=None):
        
        query = self.preprocess_attn(input, mask=pad_mask_tgt)
        
        if buffer is not None:
            buffer = torch.cat([buffer, query], dim=1)
        else:
            buffer = query
        
        if memory_bank is None:
            memory_bank = buffer.unsqueeze(0)
        
        else:
            memory_bank = torch.cat([memory_bank, buffer.unsqueeze(0)], dim=0) # batch_size x n_layer x len_src x hidden
        
        
        out, _ = self.multihead_tgt(query, memory_bank, mask_tgt, 
                                    query_mask=pad_mask_tgt, value_mask=pad_mask_tgt)
        
        input = self.postprocess_attn(out, input)
        
        """ Context Attention layer 
            layernorm > attn > dropout > residual
        """
        
        query = self.preprocess_src_attn(input, mask=pad_mask_tgt)
        out, coverage = self.multihead_src(query, context, mask_src, 
                                           query_mask=pad_mask_tgt, value_mask=pad_mask_src)
        input = self.postprocess_src_attn(out, input)
        
        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input, mask=pad_mask_tgt), 
                                           mask=pad_mask_tgt)
        input = self.postprocess_ffn(out, input)
        
        return input, memory_bank, coverage, buffer
