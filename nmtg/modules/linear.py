import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init


def group_linear(linears, input, bias=False):
    weights = [linear.weight for linear in linears]

    weight = torch.cat(weights, dim=0)

    if bias:
        biases = [linear.bias for linear in linears]
        bias_ = torch.cat(biases)
    else:
        bias_ = None

    return F.linear(input, weight, bias_)


class XavierLinear(nn.Linear):
    """
    Linear layer with Xavier initialization and optional weight normalization
    """
    def __init__(self, in_features, out_features, bias=True, weight_norm=False):
        super().__init__(in_features, out_features, bias)
        self.weight_norm = weight_norm
        if weight_norm:
            nn.utils.weight_norm(self, name='weight')

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def extra_repr(self):
        parent_repr = super().extra_repr()
        return parent_repr + ', weight_norm={}'.format(self.weight_norm)


class MaxOut(nn.Module):
    """
    Project the input up `pool_size` times, then take the maximum of the outputs.
    """
    def __init__(self, in_features, out_features, pool_size):
        super().__init__()
        self.in_features = in_features
        self.out_fetures = out_features
        self.pool_size = pool_size
        self.linear = nn.Linear(in_features, out_features * pool_size)

    def forward(self, inputs):
        original_size = inputs.size()

        projected = self.linear(inputs).view(*original_size[:-1], self.out_fetures, self.pool_size)
        out, _ = projected.max(-1)
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, pool_size={}'\
            .format(self.in_features, self.out_fetures, self.pool_size)
