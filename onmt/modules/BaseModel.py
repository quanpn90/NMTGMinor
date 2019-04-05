import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import onmt, math


#~ from onmt.modules.Transformer.Layers import XavierLinear

class Generator(nn.Module):

    def __init__(self, hidden_size, output_size):
        
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        #~ self.linear = onmt.modules.Transformer.Layers.XavierLinear(hidden_size, output_size)
        self.linear = nn.Linear(hidden_size, output_size)
        
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        
        torch.nn.init.uniform_(self.linear.weight, -stdv, stdv)
        
        self.linear.bias.data.zero_()

        
    def forward(self, input, log_softmax=True):
        
        # added float to the end 
        # print(input.size())
        logits = self.linear(input).float() 
        
        if log_softmax:
            output = F.log_softmax(logits, dim=-1)
        else:
            output = logits
        return output
        

class NMTModel(nn.Module):

    def __init__(self, encoder, decoder, generator=None):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator        
        
    def tie_weights(self):
        assert self.generator is not None, "The generator needs to be created before sharing weights"
        self.generator[0].linear.weight = self.decoder.word_lut.weight
        
    
    def share_enc_dec_embedding(self):
        self.encoder.word_lut.weight = self.decoder.word_lut.weight
        
    def mark_pretrained(self):
        
        self.encoder.mark_pretrained()
        self.decoder.mark_pretrained()
        
    def load_state_dict(self, state_dict, strict=True):
        
        def condition(param_name):
            
            if 'positional_encoder' in param_name:
                return False
            if 'time_transformer' in param_name and self.encoder.time == 'positional_encoding':
                return False
            if param_name == 'decoder.mask':
                return False
            
            return True
        

        #restore old generated if necessay for loading
        if("generator.linear.weight" in state_dict and type(self.generator) is nn.ModuleList):
            self.generator = self.generator[0]

        #~ filtered_dict = dict()
        
        #~ for
        filtered = {k: v for k, v in state_dict.items() if condition(k)}
        
        #~ for k, v in filtered.items():
            #~ print(k, type(k))
        model_dict = self.state_dict()
        #~ 
        #~ model_dict.update(filtered)
        #~ filtered.update(model_dict)
        for k,v in model_dict.items():
            if k not in filtered:
                filtered[k] = v
        #~

        super().load_state_dict(filtered)   

        if(type(self.generator) is not nn.ModuleList):
            self.generator = nn.ModuleList([self.generator])

        #~ for name, param in state_dict.items():
            #~ 
            #~ 
            #~ if isinstance(param, Parameter):
                #~ # backwards compatibility for serialized parameters
                #~ param = param.data
                #~ 
                #~ 
            #~ else:
                #~ continue
        #~ pretrained_dict = {k: v for k, v in state_dict.items() if v}

        
        
    


class Reconstructor(nn.Module):
    
    def __init__(self, decoder, generator=None):
        super(Reconstructor, self).__init__()
        self.decoder = decoder
        self.generator = generator        
    

class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.
    Modules need to implement this to utilize beam search decoding.
    """
    
    
