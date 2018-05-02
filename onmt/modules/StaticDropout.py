import torch
from torch.autograd.function import InplaceFunction, Function
from torch.autograd import Variable
from itertools import repeat
import torch.nn as nn

class StaticDropoutFunction(Function):

    @staticmethod
    def forward(ctx, input, module, train=False):
        
        
        ctx.train = train
        ctx.module = module
        ctx.p = module.p

        if ctx.p == 0 or not ctx.train:
            return input
            
        if torch.numel(module.noise) != torch.numel(input):
            module.gen_noise(input)
            
            
        ctx.noise = module.noise

        output = input * ctx.noise

        return output

    @staticmethod
    def backward(ctx, grad_output):
        #~ print("BACKWARD PASS")
        ctx.module.noise_created = False
        ctx.module.noise = None
        if ctx.p > 0 and ctx.train:
            return grad_output * ctx.noise, None, None
        else:
            return grad_output, None, None

class StaticDropout(nn.Module):

    def __init__(self, p=0.5):
        super(StaticDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.noise_created = False
    
    def gen_noise(self, input):
        self.noise = input.new().resize_as_(input)
        if self.p == 1:
            self.noise.fill_(0)
        else:
            self.noise.bernoulli_(1 - self.p).div_(1 - self.p)
        self.noise = self.noise.expand_as(input)
        self.noise_created = True

    def forward(self, input):
        
        if self.noise_created == False and self.training:
            self.gen_noise(input)
            #~ self.noise = input.new().resize_as_(input)
            #~ if self.p == 1:
                #~ self.noise.fill_(0)
            #~ else:
                #~ self.noise.bernoulli_(1 - self.p).div_(1 - self.p)
            #~ self.noise = self.noise.expand_as(input)
            #~ self.noise_created = True
     
        return StaticDropoutFunction.apply(input, self, self.training)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) + ')'
