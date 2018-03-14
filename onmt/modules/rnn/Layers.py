import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import onmt

from onmt.modules.Transformer.Layers import PrePostProcessing, MultiHeadAttention, FeedForward
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


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
        input : PackedSequence containing(len_query x batch_size x d_model)
        
        mask:  len_query x batch_size x len_key or broadcastable 
    
    Output Shapes:
        out: batch_size x len_query x d_model
    """
    
    def __init__(self, h, d_model, p, d_ff, attn_p=0.1):
        
        super(EncoderLayer, self).__init__()
        self.preprocess_rnn = PrePostProcessing(d_model, p, sequence='n')
        
        self.postprocess_rnn = PrePostProcessing(d_model, p, sequence='da')
        
        self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, p, sequence='da')
        
        
        self.rnn = nn.LSTM(d_model, d_model//2, 1, bidirectional=True)
        
        #~ feedforward = FeedForward(d_model, d_ff, p)
        self.ffn = FeedForward(d_model, d_ff, p)
    
    
    def forward(self, input, pad_mask=None):
        """ input should be packed sequence """
        batch_sizes = input.batch_sizes
        
        # should be (T * H - M) x H
        input = input.data
        
        
        # normalize the rnn_input and create the pack 
        rnn_input = PackedSequence(self.preprocess_rnn(input), batch_sizes)
        rnn_output, rnn_hidden = self.rnn(rnn_input)
        
        # dropout and add - on data level
        input = self.postprocess_rnn(rnn_output.data, input)
        
        
        # normalize
        ffn_input = self.preprocess_ffn(input)
        output = self.ffn(input)
        input = self.postprocess_ffn(output, input)
        
        # pack again
        input = PackedSequence(input, batch_sizes)
        
        return input, rnn_hidden
    
    
class DecoderLayer(nn.Module):
    
    def __init__(self, h, d_model, p, d_ff, attn_p=0.1):
        
        super(DecoderLayer, self).__init__()
        self.preprocess_rnn = PrePostProcessing(d_model, p, sequence='n')
        
        self.postprocess_rnn = PrePostProcessing(d_model, p, sequence='da')
        
        self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model, p, sequence='da')
        
        self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, p, sequence='da')
        
        
        self.multihead_src = MultiHeadAttention(h, d_model, attn_p=attn_p)
        self.rnn = nn.LSTM(d_model, d_model, 1, bidirectional=False)
        feedforward = FeedForward(d_model, d_ff, p)
        self.feedforward = feedforward  
    def forward(self, input, context, mask_src):
        """ input should NOT be packed sequence """
        """ Time x Batch x Hidden """
        
        # batch first
        memory = context.transpose(0, 1)
        # normalize the rnn_input
        rnn_input = self.preprocess_rnn(input)
        out, rnn_hidden = self.rnn(rnn_input)
        
        input = self.postprocess_rnn(out, input)
        
        """ Context Attention layer 
            layernorm > attn > dropout > residual
        """
        
        # transpose for batch first as in attention requirement
        query = self.preprocess_attn(input).transpose(0, 1)

        out, coverage = self.multihead_src(query, memory, memory, mask_src)
        
        # transpose back to time x batch
        out = out.transpose(0, 1)
        
        
        input = self.postprocess_attn(out, input)
        # input should have dim time x batch x dim now
        
        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input))
        input = self.postprocess_ffn(out, input)
        
        return input, rnn_hidden, coverage
