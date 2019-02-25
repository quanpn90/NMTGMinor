import math

import torch
import torch.nn.functional as F
from torch import nn

from nmtg.modules.linear import XavierLinear, MaxOut
from nmtg.modules.masking import MaskedFunction


class PrePostProcessing(nn.Module):
    """
    Applies processing to tensors
    Args:
        model_dim:          dimension of model
        dropout:            dropout probability
        elementwise_affine: Passed to LayerNorm
        gated_residuals:    Use gated residuals with a parameter
    sequence of processing steps:
        n = normalization
        d = dropout
        a = adding previous input to output (residual)
    """

    def __init__(self, model_dim, sequence='nda', dropout=0.0,
                 elementwise_affine=True, gated_residuals=False, masking=False):
        super(PrePostProcessing, self).__init__()
        self.masking = masking
        self.gated_residuals = gated_residuals
        self.steps = sequence

        if self.gated_residuals:
            self.k = nn.Parameter(torch.ones(1))

        if 'n' in self.steps:
            layer_norm = nn.LayerNorm([model_dim], elementwise_affine=elementwise_affine)
            self.layer_norm = MaskedFunction(layer_norm)
        if 'd' in self.steps:
            self.dropout = nn.Dropout(dropout, inplace=False)

    def forward(self, tensor, input_tensor=None, mask=None):
        output = tensor
        if not self.masking:
            mask = None

        for step in self.steps:
            if step == 'n':
                output = self.layer_norm(output, mask=mask)
            elif step == 'd':
                output = self.dropout(output)
            elif step == 'a':
                if input_tensor is not None:
                    if self.gated_residuals:
                        output = F.relu(self.k) * output + input_tensor
                    else:
                        output = output + input_tensor

        return output


def get_feed_forward(feed_forward_type, model_dim, feed_forward_dim, feed_forward_dropout,
                     weight_norm):
    if feed_forward_type == 'linear_relu_linear':
        feed_forward = FeedForward(model_dim, feed_forward_dim, feed_forward_dropout,
                                   weight_norm=weight_norm)
    elif feed_forward_type == 'maxout':
        pool_size = int(math.ceil(feed_forward_dim / model_dim))
        feed_forward = MaxOut(model_dim, model_dim, pool_size)
    else:
        raise ValueError('Unrecognized feed forward type "{}"'.format(feed_forward_type))
    return feed_forward


class FeedForward(nn.Module):
    """
    Applies position-wise feed forward to inputs

    Args:
        model_dim:      dimension of model
        hidden_dim:     dimension of feed forward
        dropout:        dropout probability
        weight_norm:    use weight normalization on the weights

    Params:
        layer_1: FC layer from model_dim to hidden_dim
        layer_2: FC layer from hidden_dim to model_dim

    Input Shapes:
        input: batch_size x len x model_dim

    Output Shapes:
        out: batch_size x len x model_dim
    """

    def __init__(self, model_dim, hidden_dim, dropout, weight_norm=False):
        super().__init__()
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.layer_1 = XavierLinear(model_dim, hidden_dim, weight_norm=weight_norm)
        self.layer_2 = XavierLinear(hidden_dim, model_dim, weight_norm=weight_norm)

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        hidden = F.relu(self.layer_1(inputs), inplace=True)
        hidden = self.dropout(hidden)
        out = self.layer_2(hidden)
        return out
