import numpy as np
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from onmt.modules.Transformer.Layers import PositionalEncoding
from onmt.modules.Transformer.Layers import EncoderLayer, DecoderLayer
from onmt.modules.StochasticTransformer.Layers import StochasticEncoderLayer, StochasticDecoderLayer
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder
from onmt.modules.BaseModel import NMTModel, Reconstructor
import onmt
from onmt.modules.WordDrop import embedded_dropout

from onmt.modules.Transformer.Layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing
from onmt.modules.VDTransformer.Layers import VDEncoderLayer, VDDecoderLayer
Linear = XavierLinear

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
    
    def __init__(self, opt, dicts, positional_encoder):
    
        self.death_rate = opt.death_rate
        self.death_type = 'linear_decay'
        self.layers = opt.layers
        self.opt = opt

        # self.death_rates, e_length = expected_length(self.layers, self.death_rate, self.death_type)  
        # build_modules will be called from the inherited constructor
        super().__init__(opt, dicts, positional_encoder)


        self.projector = Linear(2 * opt.model_size, opt.model_size)
        self.z_dropout = nn.Dropout(opt.dropout)
            
    def forward(self, input, context, latent_z, src, **kwargs):
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
        mask_tgt = input.data.eq(onmt.Constants.PAD).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)
        
        # T x B x H
        output = emb.transpose(0, 1).contiguous()

        # add dropout to embedding
        output = self.preprocess_layer(output)


        # 1 x B x H
        # latent_z = (latent_z)
        z = latent_z.unsqueeze(0).expand_as(output)

        output_plus_z = torch.cat([output, z], dim=-1)

        # combine input with latent variable
        output = torch.tanh(self.projector(output_plus_z))

        
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
            
        
        return output, None
        
class VariationalTransformer(NMTModel):
    """Main model in 'Attention is all you need' """
    
    def __init__(self, encoder, decoder, prior_estimator, posterior_estimator, generator=None):
        super().__init__(encoder, decoder, generator=generator)
        self.prior_estimator = prior_estimator
        self.posterior_estimator = posterior_estimator
        
    def forward(self, batch):
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

        p_z = self.prior_estimator(encoder_context, src)
        q_z = self.posterior_estimator(encoder_context, src, tgt)

        ### reparameterized sample:
        ### z = mean * epsilon + var
        ### epsilon is generated from Normal(0, I)
        if self.training:
            z = q_z.rsample()
        else:
            ## during testing, assuming that Y is not available
            ## we should use the mean of the prior
            z = p_z.mean

        z = z.type_as(encoder_context)
        
        decoder_output, coverage = self.decoder(tgt, encoder_context, z, src)

        # compute KL between prior and posterior
        kl_divergence = torch.distributions.kl.kl_divergence(q_z, p_z)
        outputs = dict()

        outputs['hiddens'] = decoder_output
        outputs['kl'] = kl_divergence

        return outputs


#     def create_decoder_state(self, src, context, beamSize=1):
        
#         from onmt.modules.Transformer.Models import TransformerDecodingState
        
#         decoder_state = TransformerDecodingState(src, context, beamSize=beamSize)
#         return decoder_state
