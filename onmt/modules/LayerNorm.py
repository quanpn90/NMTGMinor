import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class LayerNorm(nn.Module):
    """Applies layer normalization to last dimension
    Args:
        d: dimension of hidden units
    """
    def __init__(self, d):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(d), requires_grad=True)

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + 1e-6) + self.beta