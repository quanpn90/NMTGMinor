
import torch
import torch.nn as nn


class VariationalDropout(nn.Module):
    """
    Dropout generated for a sequence T x B x H or B x T x H
    Generate batch-level dropout and expand by T
    """
    def __init__(self,  dropout=0.5,  batch_first=False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x):

        dropout = self.dropout
        batch_first = self.batch_first

        if not self.training or not dropout:
            return x

        if not batch_first:
            # T x B x H (the Time dimension is 1 for broadcasting)
            m = x.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        else:
            m = x.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)

        mask = m / (1 - dropout)

        return mask * x