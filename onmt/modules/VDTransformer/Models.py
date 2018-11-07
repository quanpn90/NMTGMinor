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

def custom_layer(module):
    def custom_forward(*args):
        output = module(*args)
        return output
    return custom_forward
    


# VD = VARIATIONAL DEPTH ENCODER / DECODER

"""
    The actor network receives the hidden state from the Transformer
    and predicts a bernoulli distribution of skipping the layer or not.
    
    In the backward pass, we will take the reward (log likelihood of the word of the current layer)
    normalized by the baseline, and multiply with the log_prob of action to get the 
"""
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

class InferenceNetwork(nn.Module):
    
    def __init__(self, opt):
        
        super().__init__()
        self.opt = opt
        self.input_size = opt.model_size
        self.inner_size = opt.inner_size
        self.dropout = opt.dropout
        self.in_norm = nn.LayerNorm(self.input_size)
        self.fc_1 = Linear(self.input_size, self.inner_size)
        self.fc_3 = Linear(self.inner_size, 1) # output a distribution over actions (1 = move on, 0 = skip)

        # DEBUGGING ONLY WHICH FREEZES THIS NETWORK
        # for p in self.parameters():
            # p.requires_grad = False

        # THIS NETWORK ONLY RECEIVES GRADIENT VIA POLICY GRADIENT
        
        
    def forward(self, input, argmax=False):
        
        batch_size, len_input = input.size(0), input.size(1)
        # print(input.size())
        
        input = self.in_norm(input)
        
        # apply a single feed forward neural net
        # or should I use a transformer here ?  
        input = F.relu(self.fc_1(input))
        input = F.dropout(input, p=self.dropout, training=self.training, inplace=True)
        input = torch.sigmoid(self.fc_3(input).float()) # sigmoid at the end for probability / Bernoulli distribution
        
        # size should be B x T x 1 (or T x B if time first)

        dist = torch.distributions.bernoulli.Bernoulli(probs=input)
        if argmax == False:
            action = dist.sample() # should be (B x T) x 1 (zero / one)
            log_probs = dist.log_prob(action)
        else:
            with torch.no_grad():
                all_probs = torch.cat([1 - dist.probs, dist.probs], dim=-1)
                action = torch.argmax(all_probs, dim=-1, keepdim=True).float()
            log_probs = dist.log_prob(action)


        output = dict()
        
        # reshape into 3D tensor
        output['dist'] = dist
        output['action'] = action
        output['log_probs'] = log_probs
        
        return output
    
        # we should have layer normalization here
        
        # output should be logistic on top of feed forward net
        
        
"""
   The baseline receives the same input (or should we have some ahead information) ?
   to produce a prediction of the reward ( c / ppl )
"""        


class BaselineNetwork(nn.Module):
    
    def __init__(self, opt):
        
        super().__init__()
        self.opt = opt
        self.input_size = opt.model_size
        self.inner_size = opt.inner_size
        self.dropout = opt.dropout
        self.in_norm = nn.LayerNorm(self.input_size)
        
        # detach to avoid feedback Loop (if specified)
        self.detach = True
        self.fc_1 = Linear(self.input_size, self.inner_size)
        self.fc_3 = Linear(self.input_size, 1) # output a single probabilty for this layer
        
        
    def forward(self, input):
        """
            input : B x T x H (layer before Transformer)
        """
        
        if self.detach: 
            input_ = input.detach()
        else:
            input_ = input
            
        input = self.in_norm(input_)
        
        input = F.relu(self.fc_1(input))
        input = F.dropout(input, p=self.dropout, training=self.training, inplace=True)
        input = F.softplus(self.fc_3(input)) # softplus at the end for positive output ? why don't we use ReLU ?
    
        return input



class VDEncoder(TransformerEncoder):
    """Encoder in 'Attention is all you need'
    
    Args:
        opt: list of options ( see train.py )
        dicts : dictionary (for source language)
        
    """
    
    def __init__(self, opt, dicts, positional_encoder):
    
        self.death_rate = opt.death_rate
        self.death_type = 'linear_decay'
        self.layers = opt.layers

        
        
        # build_modules will be called from the inherited constructor
        
        self.death_rates, e_length = expected_length(self.layers, self.death_rate, self.death_type)  
        
        super().__init__(opt, dicts, positional_encoder)

        # for score function estimator
        self.layer_generator = InferenceNetwork(opt)

        # priors = torch.Tensor(1).fill_(self.death_rate)
        # self.register_buffer('priors', priors)
        
        # print("Stochastic Encoder with %.2f expected layers" % e_length) 
       
    def build_modules(self):
        
        self.layer_modules = nn.ModuleList()
        
        for l in range(self.layers):                
            
            block = VDEncoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, self.attn_dropout)
            
            self.layer_modules.append(block)
            
    
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
        
        emb = emb * math.sqrt(self.model_size)
            
        """ Adding positional encoding """
        emb = self.time_transformer(emb)
        
        emb = self.preprocess_layer(emb)
        
        mask_src = input.eq(onmt.Constants.PAD).unsqueeze(1) # batch_size x 1 x len_src for broadcasting
        
        #~ pad_mask = input.ne(onmt.Constants.PAD)) # batch_size x len_src
        
        context = emb.transpose(0, 1).contiguous()

        outputs = dict()

        inference_outputs = dict()
        
        for i, layer in enumerate(self.layer_modules):
            
            # pre-generate coin to use 
            # seed = torch.rand(1)

            # run the inference network on the input to predict ignore the layer or not
            inference_output = self.layer_generator(context)
            # print(inference_output)
            coin = inference_output["action"] # should have size batch * time * 1 ?
            coin_probs = inference_output["log_probs"] # batch x time x 1
            dist = inference_output["dist"] # q(z|x)

            inference_outputs[i] = dict()

            # now we compute the kl_divergence
            priors_prob = 1.0 - self.death_rates[i]
            priors = coin_probs.new(*coin_probs.size()).fill_(priors_prob)
            priors.requires_grad = False #
            priors = torch.distributions.bernoulli.Bernoulli(priors)

            # print(dist.probs.size(), priors.probs.size())
            kl = torch.distributions.kl.kl_divergence(dist, priors) # KL(q||p)

            inference_outputs['log_probs'] = coin_probs
            inference_outputs['kl'] = kl

            layer_mask = coin.type_as(context)
            
            if len(self.layer_modules) - i <= onmt.Constants.checkpointing and self.training:        
                context = checkpoint(custom_layer(layer), context, mask_src, layer_mask)

            else:
                context = layer(context, mask_src, layer_mask)      # batch_size x len_src x d_model
            
        
        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.    
        context = self.postprocess_layer(context)
        
        # we need to store
        outputs['context'] = context
        outputs['mask_src'] = mask_src
        outputs['inference'] = inference_outputs 

        # the loss will have two terms
        # logq * logp (score function estimator)
        # kl divergence (just need to sum)

        
        return outputs


class VDDecoder(TransformerDecoder):
    """Encoder in 'Attention is all you need'
    
    Args:
        opt
        dicts 
        
        
    """
    
    def __init__(self, opt, dicts, positional_encoder):
    
        self.death_rate = opt.death_rate
        self.death_type = 'linear_decay'
        self.layers = opt.layers

        self.death_rates, e_length = expected_length(self.layers, self.death_rate, self.death_type)  
        # build_modules will be called from the inherited constructor
        super().__init__(opt, dicts, positional_encoder)
        
        # for score function estimator - This should be lambda
        self.layer_generator = InferenceNetwork(opt)
        
        # baseline to reduce variance (if need to)
        # self.baseline = BaselineNetwork(opt)
        

        # build prior distribution 
        # the prior distribution is taken from the stochastic network
        # 1 because the prior is the same everywhere across the network ?
        # priors = torch.Tensor(1).fill_(self.death_rate)
        # self.register_buffer('priors', priors)
        
    
    def build_modules(self):
        
        self.layer_modules = nn.ModuleList()
        
        for l in range(self.layers):
            
            block = VDDecoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, self.attn_dropout)
            
            self.layer_modules.append(block)
            
    def forward(self, input, context, src, **kwargs):
        # For the generator network 

        """ Embedding: batch_size x len_src x d_model """
        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        if self.time == 'positional_encoding':
            emb = emb * math.sqrt(self.model_size)
        """ Adding positional encoding """
        emb = self.time_transformer(emb)
        if isinstance(emb, tuple):
            emb = emb[0]
        emb = self.preprocess_layer(emb)
        
        mask_src = src.eq(onmt.Constants.PAD).unsqueeze(1)
        
        pad_mask_src = src.data.ne(onmt.Constants.PAD)
        
        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.Constants.PAD).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)
        
        hidden = emb.transpose(0, 1).contiguous()

        outputs = dict()
        inference_outputs = dict()
        
        for i, layer in enumerate(self.layer_modules):

            # run the inference network on the input to predict ignore the layer or not
            inference_output = self.layer_generator(hidden)            
            coin = inference_output["action"] # should have size batch * time * 1 ?
            coin_probs = inference_output["log_probs"] # batch x time x 1
            dist = inference_output['dist'] # q(z|x)


            inference_outputs[i] = dict()

            # now we compute the kl_divergence  
            priors_prob = 1.0 - self.death_rates[i]
            priors = coin_probs.new(*coin_probs.size()).fill_(priors_prob)
            priors.requires_grad = False #
            priors = torch.distributions.bernoulli.Bernoulli(priors)

            kl = torch.distributions.kl.kl_divergence(dist, priors)

            # for each layer we need to store the log probs and the kl divergence 
            inference_outputs[i]['log_probs'] = coin_probs
            inference_outputs[i]['kl'] = kl
            
            layer_mask = coin.type_as(hidden)

            # from the latent variable (coin), run generator model 
            if len(self.layer_modules) - i <= onmt.Constants.checkpointing and self.training:        
                hidden, coverage = checkpoint(custom_layer(layer), hidden, context, mask_tgt, mask_src, layer_mask)

            else:
                hidden, coverage = layer(hidden, context, mask_tgt, mask_src, layer_mask)      # batch_size x len_src x d_model
            
        
        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.    
        hidden = self.postprocess_layer(hidden)

        # we need to store
        outputs['hidden'] = hidden
        outputs['mask_src'] = mask_src
        outputs['inference'] = inference_outputs 

        # the loss will have two terms
        # logq * logp (score function estimator)
        # kl divergence (just need to sum)
        return outputs
        
class VDTransformer(NMTModel):
    """Main model in 'Attention is all you need' """
    
        
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
        
        decoder_outputs = self.decoder(tgt, encoder_context, src)
        

        output = decoder_outputs['hidden'].transpose(0, 1) # transpose to have time first, like RNN models

        outputs = dict()

        outputs['hidden'] = output
        # outputs['encoder_inference'] = encoder_outputs['inference']
        outputs['decoder_inference'] = decoder_outputs['inference']

        return outputs


    def create_decoder_state(self, src, context, beamSize=1):
        
        from onmt.modules.Transformer.Models import TransformerDecodingState
        
        decoder_state = TransformerDecodingState(src, context, beamSize=beamSize)
        return decoder_state
