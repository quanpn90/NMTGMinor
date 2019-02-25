# """
# Global attention takes a matrix and a query vector. It
# then computes a parameterized convex combination of the matrix
# based on the input query.
# 
# 
        # H_1 H_2 H_3 ... H_n
          # q   q   q       q
            # |  |   |       |
              # \ |   |      /
                      # .....
                  # \   |  /
                          # a
# 
# Constructs a unit mapping.
# $$(H_1 + H_n, q) => (a)$$
# Where H is of `batch x n x dim` and q is of `batch x dim`.
#
# The full def is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.:
# 
# """
# 
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.utils.weight_norm as WeightNorm
import onmt 
import torch.nn.functional as F
from onmt.modules.Bottle import Bottle
from onmt.modules.StaticDropout import StaticDropout
from onmt.modules.Linear import XavierLinear as Linear
from onmt.modules.Linear import group_linear
from onmt.modules.MaxOut import MaxOut


class GlobalAttention(nn.Module):

    def __init__(self, dim):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_context = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim*2, dim, bias=False)
        self.linear_to_one = nn.Linear(dim, 1, bias=True)
        self.tanh = nn.Tanh()
        self.mlp_tanh = nn.Tanh()
        self.mask = None
        
        # For context gate
        self.linear_cg = nn.Linear(dim*2, dim, bias=True)
        self.sigmoid_cg = nn.Sigmoid()

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context, attn_mask=None):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        bsize = context.size(0)
        seq_length = context.size(1)
        dim = context.size(2)
        
        # project the hidden state (query)
        targetT = self.linear_in(input).unsqueeze(1)  # batch x 1 x dim
        
        # project the context (keys and values)
        reshaped_ctx = context.contiguous().view(bsize * seq_length, dim)
        
        projected_ctx = self.linear_context(reshaped_ctx)
        
        projected_ctx = projected_ctx.view(bsize, seq_length, dim)
        
        # MLP attention model
        repeat = targetT.expand_as(projected_ctx)
        sum_query_ctx = repeat + projected_ctx 
        sum_query_ctx = sum_query_ctx.view(bsize * seq_length, dim)
        
        mlp_input = self.mlp_tanh(sum_query_ctx)
        mlp_output = self.linear_to_one(mlp_input)
        
        mlp_output = mlp_output.view(bsize, seq_length, 1)
        attn = mlp_output.squeeze(2) # bsize x seq_length

        # Get attention
        if attn_mask is not None:
            # if not reshape then a bug may happen at decoding 
            reshaped_attn = attn_mask.view(*attn.size())
            attn = attn.masked_fill(reshaped_attn, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        contextCombined = torch.cat((weightedContext, input), 1)
        
        #ContextGate
        contextGate = self.sigmoid_cg(self.linear_cg(contextCombined))
        inputGate = 1 - contextGate
        
        gatedContext = weightedContext * contextGate
        gatedInput = input * inputGate
        gatedContextCombined = torch.cat((gatedContext, gatedInput), 1)

        contextOutput = self.tanh(self.linear_out(gatedContextCombined))

        return contextOutput, attn


class MultiHeadAttention(nn.Module):

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
    
    def __init__(self, h, d_model, attn_p=0.1, static=True, share=3):
        super(MultiHeadAttention, self).__init__()      
        self.h = h
        self.d = d_model
        self.share = share
        
        assert d_model % h == 0
        
        self.d_head = d_model//h
        self.fc_query = Bottle(Linear(d_model, h*self.d_head, bias=False))
        self.fc_key = Bottle(Linear(d_model, h*self.d_head, bias=False))
        self.fc_value = Bottle(Linear(d_model, h*self.d_head, bias=False))
        
        self.fc_concat = Bottle(Linear(h*self.d_head, d_model, bias=False))

        if static:
            self.attn_dropout = StaticDropout(attn_p)
        else:
            self.attn_dropout = nn.Dropout(attn_p)

    def forward(self, query, key, value, mask, query_mask=None, value_mask=None):

        len_query, b = query.size(0), query.size(1)
        len_key,  b_ = key.size(0), key.size(1)
        
        key_mask = value_mask

        # project inputs to multi-heads
        if self.share == 1:
            shared_qkv = group_linear([self.fc_query.function.linear, self.fc_key.function.linear, self.fc_value.function.linear], query)
            proj_query, proj_key, proj_value = shared_qkv.chunk(3, dim=-1)
        elif self.share == 2:
            proj_query = self.fc_query(query) # batch_size x len_query x h*d_head
            shared_kv = group_linear([self.fc_key.function.linear, self.fc_value.function.linear], key)
            proj_key, proj_value = shared_kv.chunk(2, dim=-1)
        else:
            proj_query = self.fc_query(query, mask=query_mask)  
            proj_key   = self.fc_key(key, mask=key_mask)             # batch_size x len_key x h*d_head
            proj_value = self.fc_value(value, mask=value_mask)       # batch_size x len_key x h*d_head
        
        q, k, v = proj_query, proj_key, proj_value
        # prepare the shape for applying softmax
        q = q.contiguous().view(len_query, b*self.h, self.d_head).transpose(0, 1)
        k = k.contiguous().view(len_key,   b*self.h, self.d_head).transpose(0, 1)
        v = v.contiguous().view(len_key,   b*self.h, self.d_head).transpose(0, 1)
        
        q = q * (self.d_head**-0.5)
        
        # get dotproduct softmax attns for each head
        attns = torch.bmm(q, k.transpose(1,2))  # batch_size*h x len_query x len_key
        
        attns = attns.view(b, self.h, len_query, len_key) 
        mask_ = mask.unsqueeze(-3)
        # FP16 support: cast to float and back
        attns = attns.float().masked_fill_(mask_, -float('inf')).type_as(attns)
        attns = F.softmax(attns.float(), dim=-1).type_as(attns)
        # return mean attention from all heads as coverage 
        coverage = torch.mean(attns, dim=1) 
        attns = self.attn_dropout(attns)
        attns = attns.view(b*self.h, len_query, len_key)
        
        # apply attns on value
        out = torch.bmm(attns, v)      # batch_size*h x len_query x d_head
        out = out.transpose(0, 1).contiguous().view(len_query, b, self.d)
            
        out = self.fc_concat(out)
               
        return out, coverage
        
    def step(self, query, key, value, mask, query_mask=None, value_mask=None, buffer=None):
    
        len_query, b = query.size(0), query.size(1)
        len_key,  b_ = key.size(0), key.size(1)
        
        key_mask = value_mask
        
        # project inputs to multi-heads
        
        if self.share == 1:
            # proj_query = self.fc_query(query, mask=query_mask)   # batch_size*h x len_query x d_head
            # proj_key   = self.fc_key(key, mask=key_mask)             # batch_size x len_key x h*d_head
            # proj_value = self.fc_value(value, mask=value_mask)       # batch_size x len_key x h*d_head
            shared_qkv = group_linear([self.fc_query.function.linear, self.fc_key.function.linear, self.fc_value.function.linear], query)
            proj_query, proj_key, proj_value = shared_qkv.chunk(3, dim=-1)
            if buffer is not None and 'k' in buffer and 'v' in buffer:
                proj_key = torch.cat([buffer['k'], proj_key], dim=0) # time first
                buffer['k'] = proj_key
                proj_value = torch.cat([buffer['v'], proj_value], dim=0) # time first
                buffer['v'] = proj_value
                len_key,  b_ = proj_key.size(0), proj_key.size(1)
            else:
                if buffer is None:
                    buffer = dict()
                buffer['k'] = proj_key
                buffer['v'] = proj_value
        elif self.share == 2:
            proj_query = self.fc_query(query) # batch_size x len_query x h*d_head
            if buffer is not None and 'c_k' in buffer and 'c_v' in buffer:
                proj_key = buffer['c_k']
                proj_value = buffer['c_v']
            else:
                if buffer is None:
                    buffer = dict()
                shared_kv = group_linear([self.fc_key.function.linear, self.fc_value.function.linear], key)
                proj_key, proj_value = shared_kv.chunk(2, dim=-1)
                buffer['c_k'] = proj_key
                buffer['c_v'] = proj_value
        else:
            proj_query = self.fc_query(query, mask=query_mask)  
            proj_key   = self.fc_key(key, mask=key_mask)             # batch_size x len_key x h*d_head
            proj_value = self.fc_value(value, mask=value_mask)       # batch_size x len_key x h*d_head
        
        q, k, v = proj_query, proj_key, proj_value
                
        # prepare the shape for applying soft_max
        q = q.contiguous().view(len_query, b*self.h, self.d_head).transpose(0, 1)
        k = k.contiguous().view(len_key,   b*self.h, self.d_head).transpose(0, 1)
        v = v.contiguous().view(len_key,   b*self.h, self.d_head).transpose(0, 1)
        
        q = q * (self.d_head**-0.5)
        
        # get dot_product softmax attns for each head
        attns = torch.bmm(q, k.transpose(1,2))  # batch_size*h x len_query x len_key
        
        attns = attns.view(b, self.h, len_query, len_key) 
        mask_ = mask.unsqueeze(-3)
        # FP16 support: cast to float and back
        attns = attns.float().masked_fill_(mask_, -float('inf')).type_as(attns)
        attns = F.softmax(attns.float(), dim=-1).type_as(attns)
        # return mean attention from all heads as coverage 
        coverage = torch.mean(attns, dim=1) 
        attns = self.attn_dropout(attns)
        attns = attns.view(b*self.h, len_query, len_key)
        
        # apply attn on value
        out = torch.bmm(attns, v)      # batch_size*h x len_query x d_head
        out = out.transpose(0, 1).contiguous().view(len_query, b, self.d)
            
        out = self.fc_concat(out)
       
        return out, coverage, buffer

