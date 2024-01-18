import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils.weight_norm as WeightNorm
import onmt
import torch.nn.functional as F
# from onmt.modules.swish import Swish
from onmt.modules.dropout import VariationalDropout
from torch.nn import Module
from torch import Tensor

from abc import ABC, abstractmethod
import math


class Projection(Module, ABC):
    """Applies a linear transformation to incoming data."""

    input_dim: int
    output_dim: int

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        :param input_dim:
            The dimensionality of inputs.
        :param output_dim:
            The dimensionality of projected outputs.
        """
        super().__init__()

        self.input_dim, self.output_dim = input_dim, output_dim

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        :param x:
            The input to project. *Shape:* :math:`(*,H_{inp})`, where
            :math:`H_{inp}` is the input dimensionality.

        :returns:
            The projected output. *Shape:* :math:`(*,H_{out})`, where all but
            the last dimension are the same shape as the input and
            :math:`H_{out}` is the output dimensionality.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}"


# different linears for the same input
def group_linear(linears, input, bias=False):
    weights = [linear.weight for linear in linears]

    weight = torch.cat(weights, dim=0)

    if bias:
        biases = [linear.bias for linear in linears]
        bias_ = torch.cat(biases)
    else:
        bias_ = None

    return F.linear(input, weight, bias_)


class XavierLinear(Projection):
    ''' Simple Linear layer with xavier init '''

    def __init__(self, input_dim: int, output_dim: int, bias=True, init_fn=None, weight_norm=False,
                 device=None, dtype=None):
        super(XavierLinear, self).__init__(input_dim, output_dim)
        linear = nn.Linear(input_dim, output_dim, bias=bias, device=device, dtype=dtype)

        self.weight_norm = weight_norm
        self.init_fn = init_fn

        if weight_norm:
            self.linear = WeightNorm(linear, name='weight')
        else:
            self.linear = linear

        # hack
        self.weight = self.linear.weight
        self.bias = self.linear.bias

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        if self.init_fn is not None:
            self.init_fn(self.linear)

            return

        torch.nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))

        if self.linear.bias is not None:
            # We do not calculate the true standard deviation of the uniform
            # distribution (i.e. multiply with sqrt(3)). See
            # https://github.com/pytorch/pytorch/issues/57109#issuecomment-828847575.
            bound = 1 / math.sqrt(self.input_dim) if self.input_dim > 0 else 0

            torch.nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x):
        return self.linear(x)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.linear.in_features) \
               + ', out_features=' + str(self.linear.out_features) \
               + ', bias=' + str(self.linear.bias is not None) \
               + ', weight_norm=' + str(self.weight_norm) + ')'


Linear = XavierLinear


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


class FeedForwardSwish(nn.Module):
    """Applies position-wise feed forward to inputs

    Args:
        d_model: dimension of model
        d_ff:    dimension of feed forward
        p:       dropout probability

    Params:
        fc_1: FC layer from d_model to d_ff
        fc_2: FC layer from d_ff to d_model

    Input Shapes:
        input: batch_size x len x d_model or len x batch_size x d_model

    Output Shapes:
        out: batch_size x len x d_model or len x batch_size x d_model
    """

    def __init__(self, d_model, d_ff, p, variational=False):
        super(FeedForwardSwish, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc_1 = XavierLinear(d_model, d_ff)
        self.fc_2 = XavierLinear(d_ff, d_model)
        self.swish = torch.nn.SilU()

        if variational:
            self.dropout = VariationalDropout(p)
        else:
            self.dropout = nn.Dropout(p)

    def forward(self, input):

        out = self.swish(self.fc_1(input))
        out = self.dropout(out)
        out = self.fc_2(out)
        return out


class FeedForward(nn.Module):
    """Applies position-wise feed forward to inputs

    Args:
        d_model: dimension of model
        d_ff:    dimension of feed forward
        p:       dropout probability

    Params:
        fc_1: FC layer from d_model to d_ff
        fc_2: FC layer from d_ff to d_model

    Input Shapes:
        input: batch_size x len x d_model or len x batch_size x d_model

    Output Shapes:
        out: batch_size x len x d_model or len x batch_size x d_model
    """

    def __init__(self, d_model, d_ff, p, variational=False):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc_1 = Linear(d_model, d_ff, nonlinearity="relu")
        self.fc_2 = Linear(d_ff, d_model)

        if variational:
            self.dropout = VariationalDropout(p)
        else:
            self.dropout = nn.Dropout(p)

    def forward(self, input):

        out = F.relu(self.fc_1(input), inplace=True)
        out = self.dropout(out)
        out = self.fc_2(out)
        return out


# class ChunkFeedForward(nn.Module):
#     """Applies position-wise feed forward to CHUNKs of inputs
#
#         Args:
#             d_model: dimension of model
#             d_ff:    dimension of feed forward
#             p:       dropout probability
#
#         Params:
#             fc_1: FC layer from d_model to d_ff
#             fc_2: FC layer from d_ff to d_model
#
#         Input Shapes:
#             input: batch_size x len x d_model or len x batch_size x d_model
#
#         Output Shapes:
#             out: batch_size x len x d_model or len x batch_size x d_model
#         """

    # def __init__(self, d_model, d_ff, p, **kwargs):
    #     super(ChunkFeedForward, self).__init__()
    #     self.d_model = d_model
    #     self.d_ff = d_ff
    #     self.fc_1 = Linear(d_model, d_ff, nonlinearity="relu")
    #     self.fc_2 = Linear(d_ff, d_model)
    #
    #     i
    #
    # def forward(self, input):
    #
    #     out = F.relu(self.fc_1(input), inplace=True)
    #     out = self.dropout(out)
    #     out = self.fc_2(out)
    #     return out