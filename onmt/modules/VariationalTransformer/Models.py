import numpy as np
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from onmt.modules.Transformer.Layers import PositionalEncoding
from onmt.modules.Transformer.Layers import EncoderLayer, DecoderLayer
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder
from onmt.modules.BaseModel import NMTModel, DecoderState
import onmt
from onmt.modules.WordDrop import embedded_dropout
from collections import defaultdict

from onmt.modules.Linear import XavierLinear as Linear
from onmt.modules.KLDivergence import kl_divergence_normal

torch.set_printoptions(threshold=10000)

def custom_layer(module):
    def custom_forward(*args):
        output = module(*args)
        return output
    return custom_forward


        
        

class VariationalDecoder(TransformerDecoder):
    """A variational 'variation' of the Transformer Decoder
    
    Args:
        opt
        dicts 
        positional encoder
        
    """
    
    def __init__(self, opt, dicts, positional_encoder, encoder_to_share=None):
    
        self.death_rate = opt.death_rate
        self.death_type = 'linear_decay'
        self.layers = opt.layers
        self.opt = opt
        self.ignore_source = opt.var_ignore_source

        # self.death_rates, e_length = expected_length(self.layers, self.death_rate, self.death_type)  
        # build_modules will be called from the inherited constructor
        super().__init__(opt, dicts, positional_encoder, encoder_to_share=encoder_to_share)


        # self.projector = Linear(2 * opt.model_size, opt.model_size)
        self.z_dropout = nn.Dropout(opt.dropout)

    def build_modules(self, encoder_to_share=None):

        if encoder_to_share is None:
            self.layer_modules = nn.ModuleList([DecoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, self.attn_dropout, self.residual_dropout, ignore_source=self.ignore_source) for _ in range(self.layers)])
        else:
            print("* Sharing Encoder and Decoder weights for self attention and feed forward layers ...")
            self.layer_modules = nn.ModuleList()

            for i in range(self.layers):
                decoder_layer = DecoderLayer(self.n_heads, self.model_size, self.dropout, 
                                             self.inner_size, self.attn_dropout, self.residual_dropout, 
                                             ignore_source=self.ignore_source,
                                             encoder_to_share=encoder_to_share.layer_modules[i])
                self.layer_modules.append(decoder_layer)
            
    def forward(self, input, context, src, latent_z=None,  **kwargs):
        """
        Inputs Shapes: 
            input: (Variable) batch_size x len_tgt (wanna tranpose)
            context: (Variable) batch_size x len_src x d_model
            latent_z (variable) batch_size x d_model 
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
        
        
        mask_src = src.eq(onmt.Constants.PAD).unsqueeze(1)
        
        pad_mask_src = src.data.ne(onmt.Constants.PAD)
        
        len_tgt = input.size(1) + 1
        fake_data = input.new(input.size(0), input.size(1) + 1).fill_(onmt.Constants.BOS)
        fake_data[:,1:].copy_(input)
        mask_tgt = fake_data.eq(onmt.Constants.PAD)

        mask_tgt = mask_tgt.unsqueeze(1)  + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0) # size should be B x (T+1) x (T+1)
        
        # T x B x H
        output = emb.transpose(0, 1).contiguous()

        # add dropout to embedding
        output = self.preprocess_layer(output)

        # 1 x B x H
        z = latent_z.unsqueeze(0)

        output_plus_z = torch.cat([z, output], dim=0) # concat to the time dimension

        output = output_plus_z
        
        # the rest will be the same as the Transformer

        for i, layer in enumerate(self.layer_modules):
            
            if len(self.layer_modules) - i <= onmt.Constants.checkpointing and self.training:           
                
                output, coverage = checkpoint(custom_layer(layer), output, context, mask_tgt, mask_src) 
                                                                              # batch_size x len_src x d_model
                
            else:
                output, coverage = layer(output, context, mask_tgt, mask_src) # batch_size x len_src x d_model
            
            
        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.    
        output = self.postprocess_layer(output)

        # the first state is for the latent variable
        output = output[1:,:,:]
            
        
        return output, None

    def step(self, input, decoder_state):
        """
        Inputs Shapes: 
            input: (Variable) batch_size x len_tgt (wanna tranpose)
            context: (Tensor) len_src x batch_size * beam_size x d_model
            mask_src (Tensor) batch_size x len_src
            buffer (List of tensors) List of batch_size * len_tgt-1 * d_model for self-attention recomputing
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src
            
        """
        context = decoder_state.context
        buffers = decoder_state.attention_buffers
        mask_src = decoder_state.src_mask
        latent_z = decoder_state.z
        
        if decoder_state.concat_input_seq == True:
            if decoder_state.input_seq is None:
                decoder_state.input_seq = input
            else:
                # concatenate the last input to the previous input sequence
                decoder_state.input_seq = torch.cat([decoder_state.input_seq, input], 0)
            input = decoder_state.input_seq.transpose(0, 1)
            src = decoder_state.src.transpose(0, 1)
        
        input_ = input[:,-1].unsqueeze(1)
        
        output_buffer = list()
            
        batch_size = input_.size(0)
        
        """ Embedding: batch_size x 1 x d_model """

        # note: we only take into account the embedding at time step t
        emb = self.word_lut(input_)
       
        
        emb = emb * math.sqrt(self.model_size)
        """ Adding positional encoding """

        time_step = input.size(1)
        emb = self.time_transformer(emb, t=time_step)
            
        if isinstance(emb, tuple):
            emb = emb[0]
        # emb should be batch_size x 1 x dim
            
        # Preprocess layer: adding dropout
        emb = self.preprocess_layer(emb)
        
        emb = emb.transpose(0, 1)
        
        # batch_size x 1 x len_src
        if mask_src is None:
            mask_src = src.eq(onmt.Constants.PAD).unsqueeze(1)
        
        len_tgt = input.size(1) + 1
        fake_data = input.new(input.size(0), input.size(1) + 1).fill_(onmt.Constants.BOS)
        fake_data[:,1:].copy_(input)
        mask_tgt = fake_data.eq(onmt.Constants.PAD)

        mask_tgt = mask_tgt.unsqueeze(1)  + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)
        output = emb.contiguous()

        # only concatenate if time step 1
        if time_step > 1 :

            mask_tgt = mask_tgt[:,-1:,:]
        else:
            z = latent_z.unsqueeze(0)

            output = torch.cat([z, output], dim=0) # concat to the time dimension
                
        # FOR DEBUGGING
        # ~ decoder_state._debug_attention_buffer(0)
    
        for i, layer in enumerate(self.layer_modules):
            
            buffer = buffers[i] if i in buffers else None
            # assert(output.size(0) == 1)
            output, coverage, buffer = layer.step(output, context, mask_tgt, mask_src, buffer=buffer) # batch_size x len_src x d_model
            
            decoder_state._update_attention_buffer(buffer, i)

        
        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.    
        output = self.postprocess_layer(output)

        if time_step == 1:
            output = output[1:,:,:]

        return output, coverage

        
class VariationalTransformer(NMTModel):
    """Main model in 'Attention is all you need' """
    
    def __init__(self, encoder, decoder, prior_estimator, posterior_estimator, generator=None, use_prior_training=False):
        super().__init__(encoder, decoder, generator=generator)
        self.prior_estimator = prior_estimator
        self.posterior_estimator = posterior_estimator
        self.use_prior = use_prior_training
        
    def forward(self, batch, sampling=False):
        """
        Inputs Shapes: 
            src: len_src x batch_size
            tgt: len_tgt x batch_size
        
        Outputs Shapes:
            out:      batch_size*len_tgt x model_size
            
            
        """
        src = batch.get('source')
        tgt = batch.get('target_input')
        
        src = src.transpose(0, 1) # transpose to have batch first
        tgt = tgt.transpose(0, 1)
        
        if self.encoder is not None:
            encoder_context, _ = self.encoder(src)
        else:
            encoder_context = None

        encoder_meaning, p_z = self.prior_estimator(src)
        q_z = self.posterior_estimator(encoder_meaning, src, tgt)

        ### reparameterized sample:
        ### z = mean * epsilon + var
        ### epsilon is generated from Normal(0, I)
        if self.training and not self.use_prior:
            z = q_z.rsample()
        else:
            ## during testing we use the prior
            if sampling:
                z = p_z.sample()
            else:
                z = p_z.mean


        z = z.type_as(encoder_meaning)
        
        decoder_output, coverage = self.decoder(tgt, encoder_context, src, z)

        # compute KL between prior and posterior
        kl_divergence = torch.distributions.kl.kl_divergence(q_z, p_z)
        outputs = defaultdict(lambda: None)


        outputs['hiddens'] = decoder_output
        outputs['kl'] = kl_divergence
        outputs['p_z'] = p_z
        outputs['q_z'] = q_z

        return outputs

    def create_decoder_state(self, src, context, mask_src, beamSize=1, type='old', sampling=False):
        
        _, p_z = self.prior_estimator(src.t())

        if sampling:
            z = p_z.sample()
        else:
            z = p_z.mean
        
        decoder_state = VariationalTransformerState(src, context, mask_src, z, beamSize=beamSize)
        return decoder_state

class VariationalTransformerState(DecoderState):
    
    def __init__(self, src, context, src_mask, latent_z, beamSize=1, type='old'):
        
        
        self.beam_size = beamSize
        
        self.input_seq = None
        self.attention_buffers = dict()
        
        self.src = src.repeat(1, beamSize)
        if context is not None:
            self.context = context.repeat(1, beamSize, 1)
        else:
            self.context = None
        self.beamSize = beamSize
        self.src_mask = None
        self.concat_input_seq = True
        self.z = latent_z.repeat(beamSize, 1)    
        
    def _update_attention_buffer(self, buffer, layer):
        
        self.attention_buffers[layer] = buffer # dict of 2 keys (k, v) : T x B x H
        
    def _debug_attention_buffer(self, layer):
        
        if layer not in self.attention_buffers:
            return
        buffer = self.attention_buffers[layer]
        
        for k in buffer.keys():
            print(k, buffer[k].size())
        
    def _update_beam(self, beam, b, remainingSents, idx):
        # here we have to reorder the beam data 
        # 
        for tensor in [self.src, self.input_seq]  :
                    
            t_, br = tensor.size()
            sent_states = tensor.view(t_, self.beamSize, remainingSents)[:, :, idx]
            
            sent_states.copy_(sent_states.index_select(
                            1, beam[b].getCurrentOrigin()))
                
                            
        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]
            if buffer_ is not None:
                for k in buffer_:
                    t_, br_, d_ = buffer_[k].size()
                    sent_states = buffer_[k].view(t_, self.beamSize, remainingSents, d_)[:, :, idx, :]
                    
                    sent_states.data.copy_(sent_states.data.index_select(
                                1, beam[b].getCurrentOrigin()))
    
    
    # in this section, the sentences that are still active are
    # compacted so that the decoder is not run on completed sentences
    # compatible with decoding version 1.0
    def _prune_complete_beam(self, activeIdx, remainingSents):
        
        model_size = self.context.size(-1)
        
        def updateActive(t):
            # select only the remaining active sentences
            view = t.data.view(-1, remainingSents, model_size)
            newSize = list(t.size())
            newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
            return view.index_select(1, activeIdx).view(*newSize)
                            
        

        # expected size: T x B
        def updateActive2D(t):
            view = t.view(-1, remainingSents)
            newSize = list(t.size())
            newSize[-1] = newSize[-1] * len(activeIdx) // remainingSents
            new_t = view.index_select(1, activeIdx).view(*newSize)
                            
            return new_t

        # for batch first 
        def updateActive2DBF(t):

            view = t.data.view(remainingSents, -1)
            newSize = list(t.size())
            newSize[0] = newSize[0] * len(activeIdx) // remainingSents
            new_t = view.index_select(0, activeIdx).view(*newSize)

            return new_t
        
        def updateActive4D(t):
            # select only the remaining active sentences
            nl, t_, br_, d_ = t.size()
            view = t.data.view(nl, -1, remainingSents, model_size)
            newSize = list(t.size())
            newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
            return view.index_select(2, activeIdx).view(*newSize)
        
        if self.context is not None:
            self.context = updateActive(self.context)
        
        self.input_seq = updateActive2D(self.input_seq)
        
        self.src = updateActive2D(self.src)
        
        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]
            if buffer_ is not None:
                for k in buffer_:
                    buffer_[k] = updateActive(buffer_[k])

        self.z = updateActive2DBF(self.z)

    # For the new decoder version only
    # compatible with decoding version 2.0
    def _reorder_incremental_state(self, reorder_state):
        raise NotImplementedError
        # not implemented correctly yet
        if self.context is not None:
            self.context = self.context.index_select(1, reorder_state)
        self.src_mask = self.src_mask.index_select(0, reorder_state)
                            
        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]
            if buffer_ is not None:
                for k in buffer_.keys():
                    t_, br_, d_ = buffer_[k].size()
                    buffer_[k] = buffer_[k].index_select(1, reorder_state) # 1 for time first