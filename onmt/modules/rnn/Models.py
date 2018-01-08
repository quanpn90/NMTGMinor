import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
from onmt.modules.rnn import mLSTMCell, StackedLSTM, RecurrentSequential
from onmt.modules.WordDrop import embedded_dropout
import torch.nn.functional as F
import random
from onmt.modules.BaseModel import NMTModel

        

class CuDNNEncoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        self.word_dropout = opt.word_dropout
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        input_size = opt.word_vec_size

        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)
        self.rnn = nn.LSTM(input_size, self.hidden_size,
                           num_layers=opt.layers,
                           dropout=opt.dropout,
                           bidirectional=opt.brnn)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden=None):
        if isinstance(input, tuple):
            # Lengths data is wrapped inside a Variable.
            lengths = input[1].data.view(-1).tolist()
            emb = pack(self.word_lut(input[0]), lengths)
        else:
            emb = self.word_lut(input)
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]
        return hidden_t, outputs

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

class RecurrentEncoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        input_size = opt.word_vec_size
        self.word_dropout = opt.word_dropout

        super(RecurrentEncoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)
                                     
        self.forward_rnn = RecurrentSequential('mlstm', opt.layers, input_size,
                               self.hidden_size, dropout=opt.dropout)
                               
        self.backward_rnn = RecurrentSequential('mlstm', opt.layers, input_size,
                               self.hidden_size, dropout=opt.dropout)                  
        

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden=None):
    
        # embedding
        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
            
        emb_split = emb.split(1)
        
        seq_mask = input.ne(onmt.Constants.PAD) # time x batch
        
        seq_mask = seq_mask.split(1) # convert into a list
        
        forward_output, forward_hidden = self.forward_rnn(emb_split, seq_masks=seq_mask)
        
        # reverse input and mask for the backward RNN
        rev_input = flip(input, 0)
        
        rev_seq_mask = rev_input.ne(onmt.Constants.PAD) # time x batch
        rev_seq_mask = rev_seq_mask.split(1)
        
        backward_output, backward_hidden = self.backward_rnn(reversed(emb_split), seq_masks=rev_seq_mask)
        
        # reverse the backward output to match forward's timeline
        backward_output = flip(backward_output, 0)

        # brnn output is concatenation of two 
        outputs = torch.cat((forward_output, backward_output), dim=2)
        
        
        # also concatenation the hidden states
        hidden_t = []
        
        h_t = torch.cat((forward_hidden[0], backward_hidden[0]), dim=2) # layers * batch_size * dim
        c_t = torch.cat((forward_hidden[1], backward_hidden[1]), dim=2)
        
        hidden_t = (h_t, c_t)
        
        return hidden_t, outputs


class RecurrentDecoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(RecurrentDecoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)
        self.rnn = StackedLSTM(mLSTMCell, opt.layers, input_size,
                               opt.rnn_size, dropout=opt.dropout)
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)
        self.word_dropout = opt.word_dropout
        self.hidden_size = opt.rnn_size

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden, context, init_output, attn_mask=None):
        
        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)

        outputs = []
        output = init_output
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)

            output, hidden = self.rnn(emb_t, hidden)
            output, attn = self.attn(output, context.transpose(0, 1), attn_mask=attn_mask)
            output = self.dropout(output)
            outputs += [output]

        outputs = torch.stack(outputs)
        return outputs, hidden, attn



        
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
        
        attn_mask = src.eq(onmt.Constants.PAD).t() # batch x time
        
        enc_hidden, context = self.encoder(src)
        init_output = self.make_init_decoder_output(context)

        out, dec_hidden, _attn = self.decoder(tgt, enc_hidden,
                                              context, init_output, attn_mask=attn_mask)

        return out
		
    
