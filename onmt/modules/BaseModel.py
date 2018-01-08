import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# import onmt.modules.Transformer.Layers.XavierLinear as Linear

class Generator(nn.Module):

    def __init__(self, hidden_size, output_size):
        
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, input, log_softmax=True):
        
        logits = self.linear(input)
        
        if log_softmax:
            output = F.log_softmax(logits, dim=-1)
        else:
            output = logits
        return output
        

class NMTModel(nn.Module):

    def __init__(self, encoder, decoder, generator):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator        
        
    def tie_weights(self):
        self.generator.linear.weight = self.decoder.word_lut.weight