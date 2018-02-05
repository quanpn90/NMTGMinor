import numpy as np
import torch, math
import torch.nn as nn
from onmt.modules.Transformer.Layers import PositionalEncoding
from onmt.modules.StochasticTransformer.Layers import StochasticEncoderLayer, StochasticDecoderLayer
from onmt.modules.BaseModel import NMTModel, Reconstructor
import onmt
from onmt.modules.WordDrop import embedded_dropout



class StochasticTransformerEncoder(nn.Module):
    """Encoder in 'Attention is all you need'
    
    Args:
        opt: list of options ( see train.py )
        dicts : dictionary (for source language)
        
    """
    
    def __init__(self, opt, dicts, positional_encoder):
    
        super(StochasticTransformerEncoder, self).__init__()
        
        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.death_rate = opt.death_rate
        
        self.word_lut = nn.Embedding(dicts.size(),
                                     self.model_size,
                                     padding_idx=onmt.Constants.PAD)
        
        self.emb_drop_layer = nn.Dropout(opt.emb_dropout)
        
        self.positional_encoder = positional_encoder
        
        self.layer_modules = nn.ModuleList([StochasticEncoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, 
                                                                self.attn_dropout, self.death_rate) for _ in range(self.layers)])

    def forward(self, input):
        """
        Inputs Shapes: 
            input: batch_size x len_src (wanna tranpose)
        
        Outputs Shapes:
            out: batch_size x len_src x d_model
            mask_src 
            
        """
        
        """ Embedding: batch_size x len_src x d_model """
        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        """ Scale the emb by sqrt(d_model) """
        emb = emb * math.sqrt(self.model_size)
        """ Adding positional encoding """
        emb = self.positional_encoder(emb)
        
        emb = self.emb_drop_layer(emb)
        
        mask_src = input.data.eq(onmt.Constants.PAD).unsqueeze(1)
        
        context = emb
        
        for layer in self.layer_modules:                          
            context = layer(context, context, context, mask_src)      # batch_size x len_src x d_model
        
        return context, mask_src
        

class StochasticTransformerDecoder(nn.Module):
    """Encoder in 'Attention is all you need'
    
    Args:
        opt
        dicts 
        
        
    """
    
    def __init__(self, opt, dicts, positional_encoder):
    
        super(StochasticTransformerDecoder, self).__init__()
        
        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout 
        self.attn_dropout = opt.attn_dropout
        self.death_rate = opt.death_rate
        
        self.emb_drop_layer = nn.Dropout(opt.emb_dropout)
        
        self.word_lut = nn.Embedding(dicts.size(),
                                     self.model_size,
                                     padding_idx=onmt.Constants.PAD)
        
        self.positional_encoder = positional_encoder
        
        self.layer_modules = nn.ModuleList([StochasticDecoderLayer(self.n_heads, self.model_size, self.dropout, 
                                                                self.inner_size, self.attn_dropout, self.death_rate) 
                                                                                        for _ in range(self.layers)])
        
        len_max = self.positional_encoder.len_max
        mask = torch.ByteTensor(np.triu(np.ones((len_max,len_max)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)
    
    def renew_buffer(self, new_len):
        
        self.positional_encoder.renew(new_len)
        mask = torch.ByteTensor(np.triu(np.ones((new_len,new_len)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)
        
    def forward(self, input, context, mask_src):
        """
        Inputs Shapes: 
            input: (Variable) batch_size x len_tgt (wanna tranpose)
            context: (Variable) batch_size x len_src x d_model
            mask_src (Tensor) batch_size x len_src
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src
            
        """
        
        """ Embedding: batch_size x len_src x d_model """
        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        """ Scale the emb by sqrt(d_model) """
        emb = emb * math.sqrt(self.model_size)
        """ Adding positional encoding """
        emb = self.positional_encoder(emb)
        
        emb = self.emb_drop_layer(emb)
        
        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.Constants.PAD).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)
        
        output = emb
        
        for layer in self.layer_modules:
            output, coverage = layer(output, output, output, context, mask_tgt, mask_src) # batch_size x len_src x d_model
        
        return output, coverage

  
        
class StochasticTransformer(NMTModel):
    """Main model in 'Attention is all you need' """
    
        
    def forward(self, input):
        """
        Inputs Shapes: 
            src: len_src x batch_size
            tgt: len_tgt x batch_size
        
        Outputs Shapes:
            out:      batch_size*len_tgt x model_size
            
            
        """
        src = input[0]
        tgt = input[1][:-1]  # exclude last target from inputs
        
        src = src.transpose(0, 1) # transpose to have batch first
        tgt = tgt.transpose(0, 1)
        
        context, src_mask = self.encoder(src)
        
        output, coverage = self.decoder(tgt, context, src_mask)
        
        output = output.transpose(0, 1) # transpose to have time first, like RNN models
        
        return output


class TrasnformerReconstructor(Reconstructor):
    
    def forward(self, src, contexts, context_mask):
        
        """
        Inputs Shapes: 
            src: len_src x batch_size
            context: batch_size x len_tgt x model_size
            context_mask: batch_size x len_tgt
        
        Outputs Shapes:
            output:      batch_size*(len_src-1) x model_size
            
            
        """
        src_input = src[:-1] # exclude last unit from source
        
        src_input = src_input.transpose(0, 1) # transpose to have batch first
        output, coverage = self.decoder(src, context, context_mask)
        
        output = output.transpose(0, 1) # transpose to have time first, like RNN models
        
        return output
        #~ source = source.transpose(0, 1)
        
        
