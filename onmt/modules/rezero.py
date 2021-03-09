# Implementation of the ReZERO training strategy

import torch
import torch.nn as nn


class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(1e-3))
        self.fn = fn

    def forward(self, x):
        return x * self.g
