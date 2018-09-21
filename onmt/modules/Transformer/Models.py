import numpy as np
import torch, math
import torch.nn as nn
from onmt.modules.Transformer.Layers import EncoderLayer, DecoderLayer, PositionalEncoding, variational_dropout, PrePostProcessing
from onmt.modules.BaseModel import NMTModel, Reconstructor, DecoderState
import onmt
from onmt.modules.WordDrop import embedded_dropout
#~ from onmt.modules.Checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint
from torch.autograd import Variable



def custom_layer(module):
    def custom_forward(*args):
        output = module(*args)
        return output
    return custom_forward

class TransformerEncoder(nn.Module):
    """Encoder in 'Attention is all you need'
    
    Args:
        opt: list of options ( see train.py )
        dicts : dictionary (for source language)
        
    """
    
    def __init__(self, opt, dicts, positional_encoder):
    
        super(TransformerEncoder, self).__init__()
        
        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout
        self.time = opt.time
        self.version = opt.version
        
        self.word_lut = nn.Embedding(dicts.size(),
                                     self.model_size,
                                     padding_idx=onmt.Constants.PAD)
        
        if opt.time == 'positional_encoding':
            self.time_transformer = positional_encoder
        elif opt.time == 'gru':
            self.time_transformer = nn.GRU(self.model_size, self.model_size, 1, batch_first=True)
        elif opt.time == 'lstm':
            self.time_transformer = nn.LSTM(self.model_size, self.model_size, 1, batch_first=True)
        
        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d', static=False)
        
        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')
        
        self.positional_encoder = positional_encoder
        
        self.layer_modules = nn.ModuleList([EncoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, self.attn_dropout) for _ in range(self.layers)])
    
        self.pretrained_point = -1
        
    def mark_pretrained(self):
        
        self.pretrained_point = self.layers
        
    
    def add_layers(self, n_new_layer):
        
        self.new_modules = list()
        self.layers += n_new_layer
        
        for i in range(n_new_layer):
            layer = EncoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, self.attn_dropout) 
            
            # the first layer will use the preprocessing which is the last postprocessing
            if i == 0:
                layer.preprocess_attn = self.postprocess_layer
                # replace the last postprocessing layer with a new one
                self.postprocess_layer = PrePostProcessing(d_model, 0, sequence='n')
            
            self.layer_modules.append(layer)

    def forward(self, input, **kwargs):
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
        
        if self.time == 'positional_encoding':
            emb = emb * math.sqrt(self.model_size)
        """ Adding positional encoding """
        emb = self.time_transformer(emb)
        if isinstance(emb, tuple):
            emb = emb[0]
        emb = self.preprocess_layer(emb)
        
        mask_src = input.data.eq(onmt.Constants.PAD).unsqueeze(1) # batch_size x len_src x 1 for broadcasting
        
        pad_mask = torch.autograd.Variable(input.data.ne(onmt.Constants.PAD)) # batch_size x len_src
        #~ pad_mask = None
        
        context = emb.contiguous()
        
        for i, layer in enumerate(self.layer_modules):
            
            if len(self.layer_modules) - i <= onmt.Constants.checkpointing and self.training:        
                context = checkpoint(custom_layer(layer), context, mask_src, pad_mask)

            else:
                context = layer(context, mask_src, pad_mask=None)      # batch_size x len_src x d_model
            
        
        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.    
        context = self.postprocess_layer(context)
            
        
        return context, mask_src    
        

class TransformerDecoder(nn.Module):
    """Encoder in 'Attention is all you need'
    
    Args:
        opt
        dicts 
        
        
    """
    
    def __init__(self, opt, dicts, positional_encoder):
    
        super(TransformerDecoder, self).__init__()
        
        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout 
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout
        self.time = opt.time
        self.version = opt.version
        
        if opt.time == 'positional_encoding':
            self.time_transformer = positional_encoder
        elif opt.time == 'gru':
            self.time_transformer = nn.GRU(self.model_size, self.model_size, 1, batch_first=True)
        elif opt.time == 'lstm':
            self.time_transformer = nn.LSTM(self.model_size, self.model_size, 1, batch_first=True)
        
        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d', static=False)
        
        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')
        
        self.word_lut = nn.Embedding(dicts.size(),
                                     self.model_size,
                                     padding_idx=onmt.Constants.PAD)
        
        self.positional_encoder = positional_encoder
        
        self.layer_modules = nn.ModuleList([DecoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, self.attn_dropout) for _ in range(self.layers)])
        
        len_max = self.positional_encoder.len_max
        mask = torch.ByteTensor(np.triu(np.ones((len_max,len_max)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)
    
    def renew_buffer(self, new_len):
        
        print(new_len)
        self.positional_encoder.renew(new_len)
        mask = torch.ByteTensor(np.triu(np.ones((new_len,new_len)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)
    
    def mark_pretrained(self):
        
        self.pretrained_point = self.layers
        
    
    def add_layers(self, n_new_layer):
        
        self.new_modules = list()
        self.layers += n_new_layer
        
        for i in range(n_new_layer):
            layer = EncoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, self.attn_dropout) 
            
            # the first layer will use the preprocessing which is the last postprocessing
            if i == 0:
                layer.preprocess_attn = self.postprocess_layer
                # replace the last postprocessing layer with a new one
                self.postprocess_layer = PrePostProcessing(d_model, 0, sequence='n')
            
            self.layer_modules.append(layer)
        
    def forward(self, input, context, src, **kwargs):
        """
        Inputs Shapes: 
            input: (Variable) batch_size x len_tgt (wanna tranpose)
            context: (Variable) batch_size x len_src x d_model
            mask_src (Tensor) batch_size x len_src
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src
            
        """
        
        """ Embedding: batch_size x len_tgt x d_model """
        
        
        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        if self.time == 'positional_encoding':
            emb = emb * math.sqrt(self.model_size)
        """ Adding positional encoding """
        emb = self.time_transformer(emb)
        if isinstance(emb, tuple):
            emb = emb[0]
        emb = self.preprocess_layer(emb)
        

        mask_src = src.data.eq(onmt.Constants.PAD).unsqueeze(1)
        
        pad_mask_src = torch.autograd.Variable(src.data.ne(onmt.Constants.PAD))
        
        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.Constants.PAD).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)
        
        output = emb.contiguous()
        
        pad_mask_tgt = torch.autograd.Variable(input.data.ne(onmt.Constants.PAD)) # batch_size x len_src
        pad_mask_src = torch.autograd.Variable(1 - mask_src.squeeze(1))
        
        
        for i, layer in enumerate(self.layer_modules):
            
            if len(self.layer_modules) - i <= onmt.Constants.checkpointing and self.training:           
                
                output, coverage = checkpoint(custom_layer(layer), output, context, mask_tgt, mask_src, 
                                            pad_mask_tgt, pad_mask_src) # batch_size x len_src x d_model
                
            else:
                output, coverage = layer(output, context, mask_tgt, mask_src, 
                                            pad_mask_tgt, pad_mask_src) # batch_size x len_src x d_model
            
            
        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.    
        output = self.postprocess_layer(output)
            
        
        return output, coverage
    
    #~ def step(self, input, context, src, buffer=None):
    def step(self, input, decoder_state):
        """
        Inputs Shapes: 
            input: (Variable) batch_size x len_tgt (wanna tranpose)
            context: (Variable) batch_size x len_src x d_model
            mask_src (Tensor) batch_size x len_src
            buffer (List of tensors) List of batch_size * len_tgt-1 * d_model for self-attention recomputing
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src
            
        """
        context = decoder_state.context.transpose(0, 1)
        buffer = decoder_state.buffer
        src = decoder_state.src.transpose(0, 1)
        
        if decoder_state.input_seq is None:
            decoder_state.input_seq = input
        else:
            # concatenate the last input to the previous input sequence
            decoder_state.input_seq = torch.cat([decoder_state.input_seq, input], 0)
        input = decoder_state.input_seq.transpose(0, 1)
        input_ = input[:,-1].unsqueeze(1)
        
        
        output_buffer = list()
            
        batch_size = input_.size(0)
        
        
        
        """ Embedding: batch_size x 1 x d_model """
        emb = self.word_lut(input_)
       
        
        if self.time == 'positional_encoding':
            emb = emb * math.sqrt(self.model_size)
        """ Adding positional encoding """
        if self.time == 'positional_encoding':
            emb = self.time_transformer(emb, t=input.size(1))
        else:
            prev_h = buffer[0] if buffer is None else None
            emb = self.time_transformer(emb, prev_h)
            # output_buffer.append(emb[1])
            buffer[0] = emb[1]
            
        if isinstance(emb, tuple):
            emb = emb[0]
        # emb should be batch_size x 1 x dim
            
        # Preprocess layer: adding dropout
        emb = self.preprocess_layer(emb)
        
        # batch_size x 1 x len_src
        mask_src = src.data.eq(onmt.Constants.PAD).unsqueeze(1)
        
        pad_mask_src = torch.autograd.Variable(src.data.ne(onmt.Constants.PAD))
        
        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.Constants.PAD).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        # mask_tgt = self.mask[:len_tgt, :len_tgt].unsqueeze(0).repeat(batch_size, 1, 1)
        mask_tgt = torch.gt(mask_tgt, 0)
        mask_tgt = mask_tgt[:, -1, :].unsqueeze(1)
                
        output = emb.contiguous()
        
        pad_mask_tgt = torch.autograd.Variable(input.data.ne(onmt.Constants.PAD)) # batch_size x len_src
        pad_mask_src = torch.autograd.Variable(1 - mask_src.squeeze(1))
        
        
        for i, layer in enumerate(self.layer_modules):
            
            buffer_ = buffer[i] if buffer is not None else None
            assert(output.size(1) == 1)
            output, coverage, buffer_ = layer.step(output, context, mask_tgt, mask_src, 
                                        pad_mask_tgt=None, pad_mask_src=None, buffer=buffer_) # batch_size x len_src x d_model
            
            output_buffer.append(buffer_)
        
        buffer = torch.stack(output_buffer)
        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.    
        output = self.postprocess_layer(output)
        
        decoder_state._update_state(buffer)    
        
        return output, coverage
    
  
        
class Transformer(NMTModel):
    """Main model in 'Attention is all you need' """
    
        
    def forward(self, input, grow=False):
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
        
        context, src_mask = self.encoder(src, grow=grow)
        
        output, coverage = self.decoder(tgt, context, src, grow=grow)
        
        output = output.transpose(0, 1) # transpose to have time first, like RNN models
        
        return output
        
    def create_decoder_state(self, src, context, beamSize=1):
        
        from onmt.modules.ParallelTransformer.Models import ParallelTransformerEncoder, ParallelTransformerDecoder
        from onmt.modules.StochasticTransformer.Models import StochasticTransformerEncoder, StochasticTransformerDecoder
        
        if isinstance(self.decoder, TransformerDecoder) or isinstance(self.decoder, StochasticTransformerDecoder) :
            decoder_state = TransformerDecodingState(src, context, beamSize=beamSize)
        elif isinstance(self.decoder, ParallelTransformerDecoder):
            from onmt.modules.ParallelTransformer.Models import ParallelTransformerDecodingState
            decoder_state = ParallelTransformerDecodingState(src, context, beamSize=beamSize)
        return decoder_state


class TransformerDecodingState(DecoderState):
    
    def __init__(self, src, context, beamSize=1):
        
        #if audio only toake one dimesion since only used for mask
        if(src.dim() == 3):
            self.src = src.narrow(2,0,1).squeeze(2)
        else:
            self.src = src
        
        #~ context = 
        
        self.context = context.transpose(0, 1)
        self.context = Variable(self.context.data.repeat(1, beamSize, 1))
        self.beamSize = beamSize
        
        self.buffer = None
        self.input_seq = None
        
    def _update_state(self, buffer):
        
        self.buffer = buffer
        
    def _update_beam(self, beam, b, remainingSents, idx):
        
        for tensor in [self.src, self.input_seq]  :
                    
            t_, br = tensor.size()
            sent_states = tensor.view(t_, self.beamSize, remainingSents)[:, :, idx]
            
            if isinstance(tensor, Variable):
                sent_states.data.copy_(sent_states.data.index_select(
                            1, beam[b].getCurrentOrigin()))
            else:
                sent_states.copy_(sent_states.index_select(
                            1, beam[b].getCurrentOrigin()))
                            
                            
        nl, br_, t_, d_ = self.buffer.size()
                    
        sent_states = self.buffer.view(nl, self.beamSize, remainingSents, t_, d_)[:, :, idx, :, :]
        
        sent_states.data.copy_(sent_states.data.index_select(
                            1, beam[b].getCurrentOrigin()))
    
    # in this section, the sentences that are still active are
    # compacted so that the decoder is not run on completed sentences
    def _prune_complete_beam(self, activeIdx, remainingSents):
        
        model_size = self.context.size(-1)
        
        def updateActive(t):
            # select only the remaining active sentences
            view = t.data.view(-1, remainingSents, model_size)
            newSize = list(t.size())
            newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
            return Variable(view.index_select(1, activeIdx)
                            .view(*newSize))
        
        def updateActive2D(t):
            if isinstance(t, Variable):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents)
                newSize = list(t.size())
                newSize[-1] = newSize[-1] * len(activeIdx) // remainingSents
                return Variable(view.index_select(1, activeIdx)
                                .view(*newSize))
            else:
                view = t.view(-1, remainingSents)
                newSize = list(t.size())
                newSize[-1] = newSize[-1] * len(activeIdx) // remainingSents
                new_t = view.index_select(1, activeIdx).view(*newSize)
                                
                return new_t
        
        def updateActive4D(t):
            # select only the remaining active sentences
            nl, br_, t_, d_ = t.size()
            view = t.data.view(nl, -1, remainingSents, t_, model_size)
            newSize = list(t.size())
            newSize[1] = newSize[1] * len(activeIdx) // remainingSents
            return Variable(view.index_select(2, activeIdx)
                            .view(*newSize)) 
        
        self.context = updateActive(self.context)
        
        self.input_seq = updateActive2D(self.input_seq)
        
        self.src = updateActive2D(self.src)
        
        self.buffer = updateActive4D(self.buffer)




#~ class TrasnformerReconstructor(Reconstructor):
    #~ 
    #~ def forward(self, src, contexts, context_mask):
        #~ 
        #~ """
        #~ Inputs Shapes: 
            #~ src: len_src x batch_size
            #~ context: batch_size x len_tgt x model_size
            #~ context_mask: batch_size x len_tgt
        #~ 
        #~ Outputs Shapes:
            #~ output:      batch_size*(len_src-1) x model_size
            #~ 
            #~ 
        #~ """
        #~ src_input = src[:-1] # exclude last unit from source
        #~ 
        #~ src_input = src_input.transpose(0, 1) # transpose to have batch first
        #~ output, coverage = self.decoder(src, context, context_mask)
        #~ 
        #~ output = output.transpose(0, 1) # transpose to have time first, like RNN models
        #~ 
        #~ return output
        #~ source = source.transpose(0, 1)
        
        
