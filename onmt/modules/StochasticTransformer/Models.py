import numpy as np
import torch, math
import torch.nn as nn
from onmt.modules.Transformer.Layers import PositionalEncoding
from onmt.modules.Transformer.Layers import EncoderLayer, DecoderLayer
from onmt.modules.StochasticTransformer.Layers import StochasticEncoderLayer, StochasticDecoderLayer
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder
from onmt.modules.BaseModel import NMTModel, Reconstructor
import onmt
from onmt.modules.WordDrop import embedded_dropout
from torch.utils.checkpoint import checkpoint

# ~ from onmt.modules.Checkpoint import checkpoint

from onmt.modules.Transformer.Layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing


def custom_layer(module):
    def custom_forward(*args):
        output = module(*args)
        return output
    return custom_forward   
    
def expected_length(length, death_rate, death_type):
    
    e_length = 0
    death_rates = dict()   
    
    for l in range(length):
        
        if death_type == 'linear_decay':
            survival_rate = 1.0 - (l+1)/length*death_rate
        elif death_type == 'linear_reverse':
            # the bottom layers will die more often
            survival_rate = 1.0 - (length - l )/length*death_rate
        elif death_type == 'uniform':
            survival_rate = 1.0 - death_rate
        else:
            raise NotImplementedError
        
        e_length += survival_rate
        death_rates[l] = 1 - survival_rate
        
    return death_rates, e_length

class StochasticTransformerEncoder(TransformerEncoder):
    """Encoder in 'Attention is all you need'
    
    Args:
        opt: list of options ( see train.py )
        dicts : dictionary (for source language)
        
    """
    
    def __init__(self, opt, embedding, positional_encoder, feature_embedding=None, share=None):
    
        self.death_rate = opt.death_rate
        self.death_type = opt.death_type
        self.layers = opt.layers
        
        # build_modules will be called from the inherited constructor
        
        self.death_rates, e_length = expected_length(self.layers, self.death_rate, self.death_type)  
        
        super(StochasticTransformerEncoder, self).__init__(opt, embedding, positional_encoder, feature_embedding=feature_embedding, share=share)
            
        print("Stochastic Encoder with %.2f expected layers" % e_length) 
       
    def build_modules(self, shared_encoder=None):

        if shared_encoder is not None:
            print("* Shaing encoder parameters with another encoder")
            self.layer_modules = shared_encoder.layer_modules

            self.postprocess_layer = shared_encoder.postprocess_layer
        else:
            self.layer_modules = nn.ModuleList()

            for l in range(self.layers):

                death_r = self.death_rates[l]

                block = StochasticEncoderLayer(self.n_heads, self.model_size, self.dropout,
                                               self.inner_size, self.attn_dropout, self.residual_dropout,
                                               death_rate=death_r)

                self.layer_modules.append(block)
            
    
    def forward(self, input, freeze_embedding=False, return_stack=False, additional_sequence=None, **kwargs):
        """
        Inputs Shapes: 
            input: batch_size x len_src (wanna tranpose)
        
        Outputs Shapes:
            out: batch_size x len_src x d_model
            mask_src 
            
        """

        emb, input = self.embedding_processing(input, freeze_embedding=freeze_embedding,
                                               additional_sequence=additional_sequence)
        
        mask_src = input.eq(onmt.Constants.PAD).unsqueeze(1) # batch_size x 1 x len_src for broadcasting
                
        context = emb.transpose(0, 1).contiguous()
        
        for i, layer in enumerate(self.layer_modules):
            
            # pre-generate coin to use 
            seed = torch.rand(1)

            if self.training:
                coin = (seed[0].item() >= self.death_rates[i])
            else:
                coin = True
            
            if coin:
            
                if len(self.layer_modules) - i <= onmt.Constants.checkpointing and self.training:        
                    context = checkpoint(custom_layer(layer), context, mask_src)

                else:
                    context = layer(context, mask_src)      # batch_size x len_src x d_model
            
        
        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.    
        context = self.postprocess_layer(context)
            
        
        return context, mask_src 
            
    def sample(self, input, **kwargs):
        """
        Inputs Shapes: 
            input: batch_size x len_src (wanna tranpose)
        
        Outputs Shapes:
            out: batch_size x len_src x d_model
            mask_src 
            
        """

        """ Embedding: batch_size x len_src x d_model """
        emb = self.word_lut(input)
        
        """ Scale the emb by sqrt(d_model) """
        
        emb = emb * math.sqrt(self.model_size)
            
        """ Adding positional encoding """
        emb = self.time_transformer(emb)
        
        emb = self.preprocess_layer(emb)
        
        mask_src = input.eq(onmt.Constants.PAD).unsqueeze(1) # batch_size x len_src x 1 for broadcasting
                
        context = emb.transpose(0, 1).contiguous()
        
        for i, layer in enumerate(self.layer_modules):
            
            context = layer.sample(context, mask_src)      # batch_size x len_src x d_model
            
        
        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.    
        context = self.postprocess_layer(context)
             
        return context, mask_src   


class StochasticTransformerDecoder(TransformerDecoder):
    """Encoder in 'Attention is all you need'
    
    Args:
        opt
        dicts 
        
        
    """
    
    def __init__(self, opt, dicts, positional_encoder, encoder_to_share=None):
    
        self.death_rate = opt.death_rate
        self.death_type = opt.death_type
        self.layers = opt.layers
        
        self.death_rates, e_length = expected_length(self.layers, self.death_rate, self.death_type)  
        
        # build_modules will be called from the inherited constructor
        super(StochasticTransformerDecoder, self).__init__(opt, dicts, positional_encoder)
        
        print("Stochastic Decoder with %.2f expected layers" % e_length) 
        
    
    def build_modules(self, encoder_to_share=None):
        
        if encoder_to_share is None:

            self.layer_modules = nn.ModuleList()
                    
            for l in range(self.layers):
                    
                death_r = self.death_rates[l]   
                
                block = StochasticDecoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, self.attn_dropout, death_rate=death_r)
                
                self.layer_modules.append(block)
        else:
            raise NotImplementedError
            
    def forward(self, input, input_attbs, context, src, freeze_embeddings=False, **kwargs):
        """
        Inputs Shapes: 
            input: (Tensor) batch_size x len_tgt (to be transposed)
            context: (Tensor) len_src x batch_size x d_model
        Outputs Shapes:
            out: tgt_len x batch_size x d_model
            coverage: batch_size x len_tgt x len_src

        """

        """ Embedding: batch_size x len_tgt x d_model """

        returns = defaultdict(lambda: None)

        emb = self.embedding_processing(input, input_attbs, freeze_embeddings=freeze_embeddings)
        
        mask_src = src.eq(onmt.Constants.PAD).unsqueeze(1)

        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.Constants.PAD).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)
        
        output = emb.transpose(0, 1).contiguous()
        coverage = None

        for i, layer in enumerate(self.layer_modules):
            
            # pre-generate coin to use
            returns[i] = dict()

            if self.training:
                seed = torch.rand(1)
                coin = (seed[0].item() >= self.death_rates[i])
            else:
                coin = True

            if coin:
                if len(self.layer_modules) - i <= onmt.Constants.checkpointing and self.training:           
                    
                    output_dict = checkpoint(custom_layer(layer), output, context, mask_tgt, mask_src)
                                                                                  # batch_size x len_src x d_model
                    
                else:
                    output_dict = layer(output, context, mask_tgt, mask_src) # batch_size x len_src x d_model

                returns[i]['attn_out'] = output_dict['attn_out']

                # placeholder when new things need to be included to return

                output = output_dict['final_state']
                returns['coverage'] = output_dict['coverage']
            else:
                returns[i]['attn_out'] = None
                returns['coverage'] = None
                    

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.    
        output = self.postprocess_layer(output)

        returns['final_state'] = output

        return returns
            
    def step_sample(self, input, decoder_state):
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
        context = decoder_state.context
        buffers = decoder_state.attention_buffers
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
            buffer[0] = emb[1]
            
        if isinstance(emb, tuple):
            emb = emb[0]
        # emb should be batch_size x 1 x dim
            
        # Preprocess layer: adding dropout
        emb = self.preprocess_layer(emb)
        
        emb = emb.transpose(0, 1)
        
        # batch_size x 1 x len_src
        mask_src = src.data.eq(onmt.Constants.PAD).unsqueeze(1)
        
        
        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.Constants.PAD).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)
        mask_tgt = mask_tgt[:, -1, :].unsqueeze(1)
                
        output = emb.contiguous()
        
        setup = list()
    
        for i, layer in enumerate(self.layer_modules):
            
            buffer = buffers[i] if i in buffers else None
            assert(output.size(0) == 1)
            output, coverage, buffer, coin = layer.step_sample(output, context, mask_tgt, mask_src, buffer=buffer) # batch_size x len_src x d_model
            
            decoder_state._update_attention_buffer(buffer, i)
            
            if coin == True:
                setup.append(i)
        
        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.    
        output = self.postprocess_layer(output)

        return output, coverage, setup