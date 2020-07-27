import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.utils.weight_norm as WeightNorm
import onmt
import torch.nn.functional as F
from onmt.modules.bottle import Bottle
from onmt.modules.static_dropout import StaticDropout
from onmt.modules.linear import XavierLinear as Linear
from onmt.modules.linear import XavierLinear
from onmt.modules.linear import group_linear, FeedForwardSwish
from onmt.modules.linear import FeedForward
from onmt.modules.attention import MultiHeadAttention
from onmt.modules.dropout import VariationalDropout
from onmt.modules.optimized.encdec_attention import EncdecMultiheadAttn
from onmt.modules.optimized.self_attention import SelfMultiheadAttn
from onmt.modules.optimized.feed_forward import PositionWiseFeedForward
from collections import defaultdict


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

    def __init__(self, d_model, dropout_p, sequence='nda', variational=False, elementwise_affine=True):
        super(PrePostProcessing, self).__init__()
        self.d_model = d_model
        self.dropout_p = dropout_p

        self.steps = list(sequence)

        if onmt.constants.residual_type == 'gated':
            # gated residual
            # initialize k with one 
            self.k = nn.Parameter(torch.ones(1))

        if 'n' in self.steps:
            ln = nn.LayerNorm((self.d_model,), elementwise_affine=elementwise_affine)
            self.layer_norm = Bottle(ln)
        if 'd' in self.steps:
            if variational:
                self.dropout = VariationalDropout(self.dropout_p, batch_first=False)
            else:
                self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, tensor, input_tensor=None, mask=None):

        output = tensor
        for step in self.steps:
            if step == 'n':
                # this cast is needed for O1 and FusedLayerNorm
                output = self.layer_norm(output.type_as(self.layer_norm.function.weight), mask=mask)
                output = output
            if step == 'd':
                output = self.dropout(output)
            if step == 'a':
                if input_tensor is not None:
                    if onmt.constants.residual_type != 'gated':
                        output = output + input_tensor
                    else:
                        output = F.relu(self.k) * output + input_tensor
        return output


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

    # def __init__(self, h, d_model, p, d_ff, attn_p=0.1, variational=False, death_rate=0.0, **kwargs):
    def __init__(self, opt, death_rate=0.0, **kwargs):
        super(EncoderLayer, self).__init__()
        self.variational = opt.variational_dropout
        self.death_rate = death_rate
        self.fast_self_attention = opt.fast_self_attention

        self.preprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                  variational=self.variational)
        self.preprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                 variational=self.variational)

        if opt.fast_self_attention:
            self.multihead = SelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout)
        else:
            self.multihead = MultiHeadAttention(opt.n_heads, opt.model_size, attn_p=opt.attn_dropout, share=1)

        if not opt.fast_feed_forward:

            feedforward = FeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                      variational=self.variational)
            self.feedforward = Bottle(feedforward)
        else:
            self.feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                       variational=self.variational)

    def forward(self, input, attn_mask):

        coin = True
        if self.training:
            coin = (torch.rand(1)[0].item() >= self.death_rate)

        if coin:
            query = self.preprocess_attn(input)

            if self.fast_self_attention:
                out, _ = self.multihead(query, query, query, attn_mask, None)
            else:
                out, _ = self.multihead(query, query, query, attn_mask)

            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_attn(out, input)

            """ Feed forward layer 
                layernorm > ffn > dropout > residual
            """
            out = self.feedforward(self.preprocess_ffn(input))

            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

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

    # def __init__(self, h, d_model, p, d_ff, attn_p=0.1, version=1.0, ignore_source=False,
    #              variational=False, death_rate=0.0):
    def __init__(self, opt, death_rate=0.0):
        super(DecoderLayer, self).__init__()
        self.ignore_source = opt.ignore_source
        self.variational = opt.variational_dropout
        self.death_rate = death_rate
        self.fast_self_attention = opt.fast_self_attention

        self.preprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                  variational=self.variational)

        if opt.fast_self_attention:
            self.multihead_tgt = SelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout)
        else:
            self.multihead_tgt = MultiHeadAttention(opt.n_heads, opt.model_size, attn_p=opt.attn_dropout, share=1)

        if not self.ignore_source:
            self.preprocess_src_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
            self.postprocess_src_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                          variational=self.variational)

            if not opt.fast_xattention:
                self.multihead_src = MultiHeadAttention(opt.n_heads, opt.model_size, attn_p=opt.attn_dropout, share=2)
            else:
                self.multihead_src = EncdecMultiheadAttn(opt.n_heads, opt.model_size, opt.attn_dropout)

        self.preprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                 variational=self.variational)

        if not opt.fast_feed_forward:

            feedforward = FeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                      variational=self.variational)
            self.feedforward = Bottle(feedforward)
        else:
            self.feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                       variational=self.variational)

    def forward(self, input, context, mask_tgt, mask_src,
                incremental=False, incremental_cache=None, reuse_source=True):

        """ Self attention layer 
            layernorm > attn > dropout > residual
        """
        if incremental:
            if incremental_cache is None:
                incremental_cache = dict()

        coverage = None

        coin = True
        if self.training:
            coin = (torch.rand(1)[0].item() >= self.death_rate)

        if coin:

            query = self.preprocess_attn(input)

            if self.fast_self_attention:
                out, _, = self.multihead_tgt(query, query, query, None, mask_tgt,
                                             incremental=incremental,
                                             incremental_cache=incremental_cache)
            else:
                out, _, = self.multihead_tgt(query, query, query, mask_tgt,
                                             incremental=incremental,
                                             incremental_cache=incremental_cache)

            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_attn(out, input)

            """ Context Attention layer 
                layernorm > attn > dropout > residual
            """
            if not self.ignore_source:
                query = self.preprocess_src_attn(input)
                out, coverage = self.multihead_src(query, context, context, mask_src,
                                                   incremental=incremental,
                                                   incremental_cache=incremental_cache)

                if self.training and self.death_rate > 0:
                    out = out / (1 - self.death_rate)

                input = self.postprocess_src_attn(out, input)
            else:
                coverage = None

            """ Feed forward layer 
                layernorm > ffn > dropout > residual
            """
            out = self.feedforward(self.preprocess_ffn(input))

            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_ffn(out, input)

        return input, coverage, incremental_cache


class PositionalEncoding(nn.Module):
    """Adds positional embeddings to standard word embeddings 
    This matches the original TensorFlow implementation at
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py.
    
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
        self.len_max = len_max
        self.d_model = d_model
        self.data_type = None

        self.renew(len_max)

        self.p = p

    def renew(self, new_max_len):
        # detele the old variable to avoid Pytorch's error when register new buffer
        cuda = False
        if hasattr(self, 'pos_emb'):
            cuda = self.pos_emb.is_cuda
            # self.data_type = torch.type(self.pos_emb)
            del self.pos_emb

        position = torch.arange(0, new_max_len).float()

        num_timescales = self.d_model // 2
        log_timescale_increment = math.log(10000) / (num_timescales - 1)
        inv_timescales = torch.exp(torch.arange(0, num_timescales).float() * -log_timescale_increment)
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        pos_emb = torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), 1)

        if cuda:
            pos_emb = pos_emb.cuda()

        if self.data_type is not None:
            pos_emb.type(self.data_type)
        # wrap in a buffer so that model can be moved to GPU
        self.register_buffer('pos_emb', pos_emb)
        # self.data_type = self.pos_emb.type()
        self.len_max = new_max_len

    def forward(self, word_emb, t=None):

        len_seq = t if t else word_emb.size(1)

        self.data_type = word_emb.type()

        if len_seq > self.len_max:
            self.renew(len_seq)

        if word_emb.size(1) == len_seq:
            time_ = self.pos_emb[:len_seq, :].type_as(word_emb)
            out = word_emb + time_
        else:
            # out = word_emb + Variable(self.pos_emb[:len_seq, :][-1, :], requires_grad=False)
            time_emb = self.pos_emb[len_seq - 1, :]  # 1 x dim
            # out should have size bs x 1 x dim
            out = word_emb + time_emb.unsqueeze(0).repeat(word_emb.size(0), 1, 1).type_as(word_emb)
        return out

    def get_positional_embeddings(self, word_emb, t=None):

        len_seq = t if t else word_emb.size(1)

        self.data_type = word_emb.type()
        if len_seq > self.len_max:
            self.renew(len_seq)

        if word_emb.size(1) == len_seq:
            time_emb = self.pos_emb[:len_seq, :].type_as(word_emb)

        else:
            time_emb = self.pos_emb[len_seq - 1, :].unsqueeze(0).type_as(word_emb)

        return time_emb
