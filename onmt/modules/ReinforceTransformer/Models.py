import numpy as np
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from onmt.modules.Transformer.Layers import PositionalEncoding
from onmt.modules.Transformer.Layers import EncoderLayer, DecoderLayer
from onmt.modules.StochasticTransformer.Layers import StochasticEncoderLayer, StochasticDecoderLayer
from onmt.modules.ReinforceTransformer.Layers import ReinforcedStochasticDecoder
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder
from onmt.modules.BaseModel import NMTModel, Reconstructor
import onmt
from onmt.modules.WordDrop import embedded_dropout
from onmt.modules.Checkpoint import checkpoint

from onmt.modules.Transformer.Layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing
Linear = XavierLinear

"""
    The actor network receives the hidden state from the Transformer
    and predicts a bernoulli distribution of skipping the layer or not.
    
    In the backward pass, we will take the reward (log likelihood of the word of the current layer)
    normalized by the baseline, and multiply with the log_prob of action to get the 
"""
class ActorNetwork(nn.Module):
    
    def __init__(self, opt):
        
        super().__init__()
        self.opt = opt
        self.input_size = opt.model_size
        self.inner_size = opt.inner_size
        self.dropout = opt.dropout
        self.in_norm = nn.LayerNorm(self.input_size)
        self.fc_1 = Linear(self.input_size, self.inner_size)
        self.fc_2 = Linear(self.inner_size, self.input_size)
        self.fc_3 = Linear(self.input_size, 2) # output a distribution over actions (1 = move on, 0 = skip)
        
        
    def forward(self, input):
        
        input = self.in_norm(input)
        
        input = F.relu(self.fc_1(input))
        input = F.dropout(input, p=self.dropout, training=self.training, inplace=True)
        
        input = F.relu(self.fc_2(input), inplace=True)
        
        input = F.softmax(self.fc_3(input), dim=-1)
        
        # size should be B x T x 2 (or T x B if time first)
        
        batch_size, len_input = input.size(0), input.size(1)
        
        input = input.view(-1, input.size(-1)) # reshape into 2D tensor

        dist =torch.distributions.categorical.Categorical(probs=input)
        action = dist.sample() # should be (B x T) x 1 (zero / one)
        log_probs = dist.log_prob(action)
        output = dict()
        
        # reshape into 3D tensor
        output['action'] = action.view(batch_size, len_input, 1)
        output['log_probs'] = log_probs(batch_size, len_input, 2)
        
        return output
    
    def argmax(self, input):
        
        return
    
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
        self.detach = False
        self.fc_1 = Linear(self.input_size, self.inner_size)
        self.fc_2 = Linear(self.inner_size, self.input_size)
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
        
        input = F.relu(self.fc_2(input), inplace=True)
        
        input = F.softplus(self.fc_3(input)) # softplus at the end for positive output ? why don't we use ReLU ?
    
    return input

def custom_layer(module):
    def custom_forward(*args):
        output = module(*args)
        return output
    return custom_forward
    
def expected_length(length, death_rate):
    
    e_length = 0
    
    for l in range(length):
        survival_rate = 1.0 - (l+1)/length*death_rate
        
        e_length += survival_rate
        
    return e_length

class ReinforcedStochasticDecoder(TransformerDecoder):
    """Encoder in 'Attention is all you need'
    
    Args:
        opt
        dicts 
        
        
    """
    
    def __init__(self, opt, dicts, positional_encoder):
    
        self.death_rate = opt.death_rate
        
        # build_modules will be called from the inherited constructor
        super().__init__(opt, dicts, positional_encoder)
        
        # ~ e_length = expected_length(self.layers, self.death_rate)    
        
        # ~ print("Stochastic Decoder with %.2f expected layers" % e_length) 
        self.layer_generator = ActorNetwork(opt)
        self.baseline = BaselineNetwork(opt)
        
    
    def build_modules(self):
        
        self.layer_modules = nn.ModuleList()
        
        for l in range(self.layers):
            
            # linearly decay the death rate
            death_r = ( l + 1 ) / self.layers * self.death_rate
            
            block = ReinforcedStochasticDecoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, self.attn_dropout, death_rate=death_r)
            
            self.layer_modules.append(block)
            
    def forward(self, input, context, src, reinforce=False, **kwargs):
        
        if reinforce == True:
            
            raise NotImplementedError
            
        else:
            return super().forward(input, context, src, **kwargs)
            
        
        
class ReinforceTransformer(NMTModel):
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
        
        output, coverage = self.decoder(tgt, context, src)
        
        output = output.transpose(0, 1) # transpose to have time first, like RNN models
        
        return output


    def create_decoder_state(self, src, context, beamSize=1):
        
        from onmt.modules.Transformer.Models import TransformerDecodingState
        
        decoder_state = TransformerDecodingState(src, context, beamSize=beamSize)
        return decoder_state
