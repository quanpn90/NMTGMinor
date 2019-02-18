
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn as nn

class MaxOut(nn.Module):
    def __init__(self, d, m, k):
        super(MaxOut, self).__init__()
        self.d_in, self.d_out, self.pool_size = d, m, k
        self.lin = Linear(d, m * k)

    def forward(self, inputs):
        
        original_size = inputs.size()
        
        inputs = inputs.view(-1, inputs.size(-1))
        
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(dim=max_dim)
        
        m = m.view(*original_size[:-1], m.size(-1))
        
        return m