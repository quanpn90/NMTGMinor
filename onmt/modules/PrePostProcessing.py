

import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn as nn
import onmt
from onmt.modules.Bottle import Bottle
from onmt.modules.VariationalDropout import VariationalDropout


class PrePostProcessing(nn.Module):
    
    """Applies processing to tensors 
    Args:
        d_model: dimension of model
        p:       dropout probabolity  
        sequence of processing steps: 
            n = normalization
            d = dropout
            a = adding previous input to output (residual)
    """
    
    def __init__(self, d_model, dropout_p, sequence='nda', static=True, elementwise_affine=True):
        super(PrePostProcessing, self).__init__() 
        self.d_model = d_model
        self.dropout_p = dropout_p     
        self.residual_type = onmt.Constants.residual_type
        self.steps = list(sequence)
        self.static = static
        
        if self.residual_type == 'gated':
            # gated residual
            # initialize k with one 
            self.k = nn.Parameter(torch.ones(1))

        if 'n' in self.steps:
            ln = nn.LayerNorm((self.d_model,), elementwise_affine=elementwise_affine)
            self.layer_norm = Bottle(ln)
        if 'd' in self.steps:
            # assert('v' not in self.steps)
            self.dropout = nn.Dropout(self.dropout_p, inplace=False)
        if 'v' in self.steps:
            # assert('d' not in self.steps)
            self.var_dropout = VariationalDropout(self.dropout_p, batch_first=False)
    
    def forward(self, tensor, input_tensor=None, mask=None):
        output = tensor
        for step in self.steps:
            if step == 'n':
                output = self.layer_norm(output, mask=mask)
                output = output
            if step == 'd':
                output = self.dropout(output)
            if step == 'a':
                if input_tensor is not None:
                    if self.residual_type != 'gated':
                        output = output + input_tensor
                    else:
                        output = F.relu(self.k) * output + input_tensor
            if step == 'v':
                output = self.var_dropout(output)

        return output
