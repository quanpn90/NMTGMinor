import numpy as np
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from onmt.modules.Transformer.Layers import PositionalEncoding
from onmt.modules.Transformer.Layers import EncoderLayer, DecoderLayer
from onmt.modules.StochasticTransformer.Layers import StochasticEncoderLayer, StochasticDecoderLayer
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder
from onmt.modules.BaseModel import NMTModel, DecoderState
import onmt
from onmt.modules.WordDrop import embedded_dropout
from torch.autograd import Variable


from onmt.modules.Transformer.Layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing
from onmt.modules.VDTransformer6.Layers import VDEncoderLayer, VDDecoderLayer
from onmt.utils import mean_with_mask
Linear = XavierLinear

torch.set_printoptions(threshold=10000)

def custom_layer(module):
    def custom_forward(*args):
        output = module(*args)
        return output
    return custom_forward


def kl_divergence_list(p_list, q_list):

    kl = list()
    for (p_z_i, q_z_i) in zip(p_list, q_list):
        kl_ = torch.distributions.kl.kl_divergence(q_z_i, p_z_i)
        kl.append(kl_.unsqueeze(0))

    kl = torch.cat(kl, dim=0)

    return kl
class VDDecoder(TransformerDecoder):
    """A variational 'variation' of the Transformer Decoder
    
    Args:
        opt
        dicts 
        positional encoder
        
    """
    
    def __init__(self, opt, dicts, positional_encoder):
    
        self.death_rate = opt.death_rate
        self.death_type = 'linear_decay'
        self.layers = opt.layers
        self.opt = opt

        # self.death_rates, e_length = expected_length(self.layers, self.death_rate, self.death_type)  
        # build_modules will be called from the inherited constructor
        super().__init__(opt, dicts, positional_encoder)


        # self.projector = Linear(2 * opt.model_size, opt.model_size)
        self.z_dropout = nn.Dropout(opt.dropout)

    def build_modules(self):
        
        self.layer_modules = nn.ModuleList([VDDecoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, self.attn_dropout) for _ in range(self.layers)])
            
    def forward(self, input, context, src, context_mean=None, posterior_model=None, priors=None):
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
        
        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.Constants.PAD) # B x T 
        mask_tgt = mask_tgt.unsqueeze(1)  + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)
        
        # T x B x H
        output = emb.transpose(0, 1).contiguous()

        # add dropout to embedding
        output = self.preprocess_layer(output)

        outputs = dict()

        q_z = list()
        baselines = list()
        log_q_z = list()
        # z size: B x L

        for i, layer in enumerate(self.layer_modules):

            
            if posterior_model is not None:

                q_z_i = posterior_model(context_mean, output, src, input) # prediction of layer usage at this step
                # b_i   = baseline_model(context_mean, output, src, input)  # prediction of baseline at this step

                z_ = q_z_i.rsample() # B x H
                z_ = z_.unsqueeze(0).type_as(output) 

                # should be 1 x B x H
                # auto-broacasted to T x B x H
                # baselines.append(b_i)
                q_z.append(q_z_i)
                # log_q_z.append(log_q_z_i)

                if len(self.layer_modules) - i <= onmt.Constants.checkpointing and self.training:           
                    output, coverage = checkpoint(custom_layer(layer), output, context, mask_tgt, mask_src, z_) 
                else:
                    output, coverage = layer(output, context, mask_tgt, mask_src, z_) # batch_size x len_src x d_model
            elif priors is not None:

                assert self.training==False
                z_i = priors[i].rsample()
                z_i = z_i.unsqueeze(0).type_as(output) 
                output, coverage = layer(output, context, mask_tgt, mask_src, z_i)

            else:
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

        outputs['decoder_states'] = output
        outputs['q_z'] = q_z # list of distributions, probs size B x H
        outputs['coverage'] = coverage

        return outputs

    def step(self, input, decoder_state, sampling=False):
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
        # ~ print(input.size())
        # ~ print(mask_src.size())
        # ~ print(context.size())
        
        
        output_buffer = list()
            
        batch_size = input_.size(0)
        
        """ Embedding: batch_size x 1 x d_model """
        emb = self.word_lut(input_)
       
        
        emb = emb * math.sqrt(self.model_size)
        """ Adding positional encoding """
        emb = self.time_transformer(emb, t=input.size(1))
            
        if isinstance(emb, tuple):
            emb = emb[0]
        # emb should be batch_size x 1 x dim
            
        # Preprocess layer: adding dropout
        emb = self.preprocess_layer(emb)
        
        emb = emb.transpose(0, 1)
        
        # batch_size x 1 x len_src
        if mask_src is None:
            mask_src = src.eq(onmt.Constants.PAD).unsqueeze(1)
        
        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.Constants.PAD).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)
        mask_tgt = mask_tgt[:, -1, :].unsqueeze(1)
                
        output = emb.contiguous()

        # z size: B x L 
        if latent_z is not None:
            z_splits = torch.split(latent_z, 1, dim=1)
    
        for i, layer in enumerate(self.layer_modules):

            z_ = z_splits[i].unsqueeze(0) # should be 1 x B x H
                                              # auto-broacasted to T x B x H
            
            buffer = buffers[i] if i in buffers else None
            assert(output.size(0) == 1)
            output, coverage, buffer = layer.step(output, context, mask_tgt, mask_src, z_, buffer=buffer) # batch_size x len_src x d_model
            
            decoder_state._update_attention_buffer(buffer, i)

        
        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.    
        output = self.postprocess_layer(output)

        return output, coverage
        
class VDTransformer(NMTModel):
    """Main model in 'Attention is all you need' """
    
    def __init__(self, encoder, decoder, prior_estimator, posterior_estimator, generator=None, baseline=None):
        super().__init__(encoder, decoder, generator=generator)
        self.prior_estimator = prior_estimator
        self.posterior_estimator = posterior_estimator
        self.baseline = baseline
        
    def forward(self, batch, sampling=True):
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
        
        encoder_context, src_mask = self.encoder(src)

        context_mask = src.eq(onmt.Constants.PAD).transpose(0, 1).unsqueeze(2)
        context_mean = mean_with_mask(encoder_context, context_mask)

        p_z = self.prior_estimator(context_mean, src)

        outputs = dict()
        if self.training:
            decode_outputs = self.decoder(tgt, encoder_context, src, context_mean=context_mean, 
                                          posterior_model=self.posterior_estimator)
            q_z = decode_outputs['q_z']
            # compute KL between prior and posterior
            kl_divergence = kl_divergence_list(q_z, p_z)
            

            outputs['hiddens'] = decode_outputs['decoder_states']
            outputs['kl'] = kl_divergence

            # we need to use the log likelihood of the sentence logP(Y | X, z)
            # to backprop this volume
            outputs['log_q_z'] = None
            outputs['baseline'] = None
            outputs['q_z'] = q_z
            outputs['p_z'] = p_z
        else:
            decode_outputs = self.decoder(tgt, encoder_context, src, priors=p_z)

            outputs['hiddens'] = decode_outputs['decoder_states']
            outputs['p_z'] = p_z

        return outputs

    def create_decoder_state(self, src, encoder_context, src_mask, beamSize, type='old', sampling=False):

        # p_z = self.prior_estimator(encoder_context, src.transpose(0, 1))

        # if sampling:
        # # z = torch.argmax(p_z.probs, dim=-1)
        #     z = p_z.sample()
        # else:
        #     z = torch.argmax(p_z.probs, dim=-1)
        # z = z.type_as(encoder_context)
        decoder_state = VariationalDecodingState(src, encoder_context, src_mask, z, beamSize=beamSize)

        return decoder_state


class VariationalDecodingState(DecoderState):
    
    def __init__(self, src, context, src_mask, z, beamSize=1, type='old'):
        
        
        self.beam_size = beamSize
        
        self.input_seq = None
        self.attention_buffers = dict()

        print(z)
        
        if type == 'old':
            self.src = src.repeat(1, beamSize)
            self.context = context.repeat(1, beamSize, 1) # T x B x H to T x Bxb x H
            self.beamSize = beamSize
            self.src_mask = None
            self.concat_input_seq = True 
            self.z = z.repeat(beamSize, 1) # Bxb x L
        elif type == 'new':
            bsz = context.size(1)
            new_order = torch.arange(bsz).view(-1, 1).repeat(1, self.beam_size).view(-1)
            new_order = new_order.to(context.device)
            self.context = context.index_select(1, new_order)
            self.src_mask = src_mask.index_select(0, new_order)
            self.concat_input_seq = False
            self.z = z.index_select(0, new_order)
        
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
            
            if isinstance(tensor, Variable):
                sent_states.data.copy_(sent_states.data.index_select(
                            1, beam[b].getCurrentOrigin()))
            else:
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

        # update z
        br, l = self.z.size()
        sent_states = self.z.view(self.beamSize, remainingSents, l)[:,idx,:]
        sent_states.copy_(sent_states.index_select(0, beam[b].getCurrentOrigin()))

    
    
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
            return Variable(view.index_select(2, activeIdx)
                            .view(*newSize)) 
        
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
    def _reorder_incremental_state(self, reorder_state):
        self.context = self.context.index_select(1, reorder_state)
        self.src_mask = self.src_mask.index_select(0, reorder_state)
        self.z = self.z.index_select(0, reorder_state)
                            
        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]
            if buffer_ is not None:
                for k in buffer_.keys():
                    t_, br_, d_ = buffer_[k].size()
                    buffer_[k] = buffer_[k].index_select(1, reorder_state) # 1 for time first