import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from onmt.modules.LayerNorm import LayerNorm
import torch.nn.init as init
import torch.nn.utils.weight_norm as WeightNorm
import onmt 
import torch.nn.functional as F
from onmt.modules.Bottle import Bottle
from onmt.modules.StaticDropout import StaticDropout
def contiguous(tensor):

    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def uniform_unit_scaling(tensor, nonlinearity="linear", gain=1.0):
    
    size = 1.
    # Estimate the input size. This won't work perfectly,
    # but it covers almost all use cases where this initialiser
    # would be expected to be useful, i.e in large linear and
    # convolutional layers, as the last dimension will almost
    # always be the output size.
    
    if isinstance(tensor, Variable):
        uniform_unit_scaling(tensor.data, nonlinearity)
        return tensor
    
    for dimension in list(tensor.size())[:-1]:
        size *= dimension
        
    activation_scaling = torch.nn.init.calculate_gain(nonlinearity, tensor)
    
    max_value = math.sqrt(3 / size) * activation_scaling
    
    return tensor.uniform_(-max_value, max_value)

class XavierLinear(nn.Module):
    
    ''' Simple Linear layer with xavier init '''
    def __init__(self, d_in, d_out, bias=True, nonlinearity='linear'):
        super(XavierLinear, self).__init__()
        linear = nn.Linear(d_in, d_out, bias=bias)
        
        weight_norm = onmt.Constants.weight_norm
        self.weight_norm = weight_norm
        
        if weight_norm:
            self.linear = WeightNorm(linear, name='weight')
        else:
            self.linear = linear
            
        init.xavier_uniform_(self.linear.weight)
        
        if bias:
            self.linear.bias.data.zero_()
    

    def forward(self, x):
        return self.linear(x)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.linear.in_features) \
            + ', out_features=' + str(self.linear.out_features) \
            + ', bias=' + str(self.linear.bias is not None) \
            + ', weight_norm=' + str(self.weight_norm) + ')'
        

Linear = XavierLinear

def variational_dropout(input, p, training=False):
    """Applies Variational Dropout (query, key, value)
    Inputs:
        input: Variable - batch_size * time_steps * hidden
    """
    if training:
        bsize = input.size(0)
        hsize = input.size(2)
        
        # create a mask for one time step
        mask = Variable(input.data.new(bsize, hsize).bernoulli_(1 - p).div_(1 - p), requires_grad=False)
        
        # then expand it to all time steps 
        mask = mask.unsqueeze(1).expand_as(input)
        output = input * mask
        return output
    # if eval then return the input
    return input
    
class MaxOut(nn.Module):
    def __init__(self, d, m, k):
        super(MaxOut, self).__init__()
        self.d_in, self.d_out, self.pool_size = d, m, k
        self.lin = Linear(d, m * k)

    def forward(self, inputs):
        
        original_size = inputs.size()
        
        inputs = inputs.view(-1, inputs.size(-1))
        
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(dim=max_dim)
        
        m = m.view(*original_size[:-1], m.size(-1))
        
        return m

class PrePostProcessing(nn.Module):
    
    """Applies processing to tensors 
    Args:
        d_model: dimension of model
        p:       dropout probabolity  
        sequence of processing steps: 
            n = normalization
            d = dropout
            a = adding previous input to output (residual)
    """
    
    def __init__(self, d_model, dropout_p, sequence='nda', static=True, elementwise_affine=True):
        super(PrePostProcessing, self).__init__() 
        self.d_model = d_model
        self.dropout_p = dropout_p     
        
        self.steps = list(sequence)
        
        if onmt.Constants.residual_type == 'gated':
            # gated residual
            # initialize k with one 
            self.k = nn.Parameter(torch.ones(1))
        
        if 'n' in self.steps:
            
            ln = nn.LayerNorm((self.d_model,),elementwise_affine=elementwise_affine)
            self.layer_norm = Bottle(ln)
        if 'd' in self.steps:
            if static:
                self.dropout = StaticDropout(self.dropout_p)
            else:
                self.dropout = nn.Dropout(self.dropout_p, inplace=False)
    
    def forward(self, tensor, input_tensor=None, mask=None):
        #~ mask = None
        output = tensor
        for step in self.steps:
            if step == 'n':
                output = self.layer_norm(output, mask=mask)
                output = output
            if step == 'd':
                output = self.dropout(output)
            if step == 'a':
                if input_tensor is not None:
                    if onmt.Constants.residual_type != 'gated':
                        output = output + input_tensor
                    else:
                        output = F.relu(self.k) * output + input_tensor
        return output
        
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
    
    def __init__(self, h, d_model, attn_p=0.1, static=True):
        super(MultiHeadAttention, self).__init__()      
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
        # FP16 support: cast to float and back
        attns = attns.float().masked_fill_(mask_, -float('inf')).type_as(attns)
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
        
        
    def step(self, query, key, value, mask, query_mask=None, value_mask=None):
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
        # FP16 support: cast to float and back
        attns = attns.float().masked_fill_(mask_, -float('inf')).type_as(attns)
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
    
class FeedForward(nn.Module):
    """Applies position-wise feed forward to inputs
    
    Args:
        d_model: dimension of model 
        d_ff:    dimension of feed forward
        p:       dropout probability 
        
    Params:
        fc_1: FC layer from d_model to d_ff
        fc_2: FC layer from d_ff to d_model
        
    Input Shapes:
        input: batch_size x len x d_model
        
    Output Shapes:
        out: batch_size x len x d_model
    """
    
    def __init__(self, d_model, d_ff, p, static=True):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc_1 = Linear(d_model, d_ff, nonlinearity="relu")
        self.fc_2 = Linear(d_ff, d_model)

        if static:
            self.dropout = StaticDropout(p)
        else:
            self.dropout = nn.Dropout(p)
        
    def forward(self, input):
        
        out = F.relu(self.fc_1(input), inplace=False)
        out = self.dropout(out)
        out = self.fc_2(out)
        return out

    
class EncoderLayer(nn.Module):
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
        super(EncoderLayer, self).__init__()
        self.version = version
        
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
        pad_mask = None
        query = self.preprocess_attn(input)
        out, _ = self.multihead(query, query, query, attn_mask)
        input = self.postprocess_attn(out, input)
        
        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input), 
                               mask=pad_mask)
        input = self.postprocess_ffn(out, input)
        
        return input
    
    
    
class DecoderLayer(nn.Module):
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
    
    def __init__(self, h, d_model, p, d_ff, attn_p=0.1, version=1.0):
        super(DecoderLayer, self).__init__()
        self.version = version
        
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
    
    def forward(self, input, context, mask_tgt, mask_src, pad_mask_tgt=None, pad_mask_src=None, residual_dropout=0.0):
        
        """ Self attention layer 
            layernorm > attn > dropout > residual
        """
        #~ print(input.size())
        #~ print(context.size())
        #~ print(pad_mask_tgt.size())
        
        query = self.preprocess_attn(input)
        
        self_context = query
        
        out, _ = self.multihead_tgt(query, self_context, self_context, mask_tgt)
        
        if residual_dropout > 0:
            input_ = F.dropout(input, residual_dropout, self.training, False)
            input = self.postprocess_attn(out, input_)
        else:
            input = self.postprocess_attn(out, input)
        
        
        
        """ Context Attention layer 
            layernorm > attn > dropout > residual
        """
        
        query = self.preprocess_src_attn(input)
        out, coverage = self.multihead_src.step(query, context, context, mask_src)
        input = self.postprocess_src_attn(out, input)
        
        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input))
        input = self.postprocess_ffn(out, input)
    
        return input, coverage
        
    def step(self, input, context, mask_tgt, mask_src, pad_mask_tgt=None, pad_mask_src=None, buffer=None):
        """ Self attention layer 
            layernorm > attn > dropout > residual
        """
        
        query = self.preprocess_attn(input)
        
        if buffer is not None:
            buffer = torch.cat([buffer, query], dim=1)
        else:
            buffer = query
            

        out, _ = self.multihead_tgt(query, buffer, buffer, mask_tgt)
        

        input = self.postprocess_attn(out, input)
        
        """ Context Attention layer 
            layernorm > attn > dropout > residual
        """
        
        query = self.preprocess_src_attn(input)
        out, coverage = self.multihead_src(query, context, context, mask_src)
        input = self.postprocess_src_attn(out, input)
        
        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input))
        input = self.postprocess_ffn(out, input)
        
        return input, coverage, buffer

class PositionalEncoding(nn.Module):
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
    def __init__(self, d_model, p=0, len_max=512):
        # save a fixed positional embedding matrix up to len_max,
        # so that no need to recreate it everytime
        super(PositionalEncoding, self).__init__()
        self.len_max=len_max
        self.d_model = d_model
        
        self.renew(len_max)
        
        self.p = p
    
    def renew(self, new_max_len):
        
        ## detele the old variable to avoid Pytorch's error when register new buffer
        if hasattr(self, 'pos_emb'):
            del self.pos_emb
        position = torch.arange(0,new_max_len).float()                      
        num_timescales = self.d_model // 2
        log_timescale_increment = math.log(10000) / (num_timescales-1)
        inv_timescales = torch.exp(torch.arange(0, num_timescales).float() * -log_timescale_increment)
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        pos_emb = torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), 1)
        # wrap in a buffer so that model can be moved to GPU
        self.register_buffer('pos_emb', pos_emb)

        
    def forward(self, word_emb, t=None):
        len_seq = t if t else word_emb.size(1)
        if word_emb.size(1) == len_seq:
            out = word_emb + Variable(self.pos_emb[:len_seq, :], requires_grad=False)
        else:
            # out = word_emb + Variable(self.pos_emb[:len_seq, :][-1, :], requires_grad=False)
            time_emb = Variable(self.pos_emb[len_seq-1, :], requires_grad=False) # 1 x dim
            # out should have size bs x 1 x dim
            out = word_emb + time_emb.unsqueeze(0).repeat(word_emb.size(0), 1, 1)
        return out
