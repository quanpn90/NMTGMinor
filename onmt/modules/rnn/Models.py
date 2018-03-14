import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
from onmt.modules.rnn import mLSTMCell, StackedLSTM, RecurrentSequential
from onmt.modules.WordDrop import embedded_dropout
import torch.nn.functional as F
import random
from onmt.modules.BaseModel import NMTModel
from onmt.modules.rnn.Layers import EncoderLayer, DecoderLayer
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import PackedSequence
from onmt.modules.Transformer.Layers import PrePostProcessing 


def unsort(input, indices, dim=1):
    
    """ unsort the tensor based on indices which are created by sort """
    
    """ dim is the dimension of batch size """
    output = input.new(*input.size())
    
    output.scatter_(dim, indices.unsqueeze(0).unsqueeze(2), input)
    
    return output
    
    

class RecurrentEncoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.model_size = opt.model_size
        self.inner_size = opt.inner_size
        self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout
        self.n_heads = opt.n_heads
        super(RecurrentEncoder, self).__init__()

        
        self.word_lut = nn.Embedding(dicts.size(),
                                     self.model_size,
                                     padding_idx=onmt.Constants.PAD)
        
        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d')
        
        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')
        
        self.layer_modules = nn.ModuleList([EncoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, self.attn_dropout) for _ in range(self.layers)])
                                                                                                                                    
        
        
    def forward(self, input):
        
        """
        Inputs Shapes: 
            input: len_src x batch_size  (wanna tranpose)
        """
        
        # first, create the inputs for packed sequence 
        mask = input.data.ne(onmt.Constants.PAD)
        
        lengths = Variable(torch.sum(mask, dim=0)) 
        
        # sort the lengths by descending order
        # remember the ind to unsort the output tensors
        sorted_lengths, ind = torch.sort(lengths, 0, descending=True)
        
        # sort the input by length
        sorted_input = input.index_select(1, ind)
        
        packed_input = pack(sorted_input, sorted_lengths)
        batch_sizes = packed_input.batch_sizes
        
        emb = embedded_dropout(self.word_lut, packed_input.data, dropout=self.word_dropout if self.training else 0)
        
        # add dropout ( works on 2D tensor)
        emb = self.preprocess_layer(emb)
        
        # pack the input in a PackedSequence
        packed_input = PackedSequence(emb, batch_sizes)
        
        rnn_hiddens = []
        
        output = packed_input
        
        for layer in self.layer_modules:                          
            output, rnn_hidden = layer(output)      # len_src x batch_size x d_model
            rnn_hiddens.append(rnn_hidden)
            
            
        output = PackedSequence(self.postprocess_layer(output.data), batch_sizes) 
        
        # restore the mask to the tensor 
        context = unpack(output)[0]
        
        # unsort the context and the rnn_hiddens 
        context = unsort(context, ind, dim=1)
        #~ 
        #~ for i, hidden in rnn_hiddens:
            #~ rnn_hiddens[i] = unsort(hidden, ind, dim=1)
        
        return context, rnn_hiddens

class RecurrentDecoder(nn.Module):

    def __init__(self, opt, dicts):
        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout 
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout
        
        super(RecurrentDecoder, self).__init__()
        
        self.word_lut = nn.Embedding(dicts.size(),
                                     self.model_size,
                                     padding_idx=onmt.Constants.PAD)
        
        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d')
        
        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')
        
        self.layer_modules = nn.ModuleList([DecoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, self.attn_dropout) for _ in range(self.layers)])


    def forward(self, input, context, src, hidden=None):
        """ Inputs:
        context (Variable): len_src * batch_size * H
        input ( Variable): len_tgt * batch_size
        src ( Variable) : len_src * batch_size
        
        """
        
        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        
        # transpose to have batch first to fit attention format
        mask_src = src.data.eq(onmt.Constants.PAD).transpose(0, 1).unsqueeze(1)
        
        # normalize the embedding 
        emb = self.preprocess_layer(emb)
        
        output = emb
        
        rnn_hiddens = list()
        
        for layer in self.layer_modules:
            output, rnn_hidden, coverage = layer(output, context, mask_src)
            
            rnn_hiddens.append(rnn_hidden)

        output = self.postprocess_layer(output)
        
        return output, rnn_hiddens, coverage


        
class RecurrentModel(NMTModel):

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    # def _fix_enc_hidden(self, h):
         # the encoder hidden is  (layers*directions) x batch x dim
         # we need to convert it to layers x batch x (directions*dim)
        # if self.encoder.num_directions == 2:
            # return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    # .transpose(1, 2).contiguous() \
                    # .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        # else:
            # return h

    def forward(self, input):
        src = input[0]
        tgt = input[1][:-1]  # exclude last target from inputs
        
        
        
        #~ attn_mask = src.eq(onmt.Constants.PAD).t() # batch x time
        #~ print(tgt.size())
        
        # to debug: detach the context
        
        context, hiddens = self.encoder(src)
        
        context = Variable(context.data)
        
        out, hiddens, coverage = self.decoder(tgt, context, src)
        #~ init_output = self.make_init_decoder_output(context)

        #~ out, dec_hidden, _attn = self.decoder(tgt, enc_hidden,
                                              #~ context, init_output, attn_mask=attn_mask)
        
        #~ print(out.size())
        
        return out

    
