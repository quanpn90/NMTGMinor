import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils.weight_norm as WeightNorm
import onmt
import torch.nn.functional as F
from onmt.modules.Swish import Swish
from onmt.modules.WordDrop import VariationalDropout


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


class XavierLinear(nn.Module):
    ''' Simple Linear layer with xavier init '''

    def __init__(self, d_in, d_out, bias=True, nonlinearity='linear'):
        super(XavierLinear, self).__init__()
        linear = nn.Linear(d_in, d_out, bias=bias)

        weight_norm = onmt.Constants.weight_norm
        self.weight_norm = weight_norm

        if weight_norm:
            self.linear = WeightNorm(linear, name='weight')
        else:
            self.linear = linear

        init.xavier_uniform_(self.linear.weight)

        if bias:
            self.linear.bias.data.zero_()

    def forward(self, x):
        return self.linear(x)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.linear.in_features) \
               + ', out_features=' + str(self.linear.out_features) \
               + ', bias=' + str(self.linear.bias is not None) \
               + ', weight_norm=' + str(self.weight_norm) + ')'


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
        self.swish = Swish()

        if variational:
            self.dropout = VariationalDropout(p)
        else:
            self.dropout = nn.Dropout(p)

    def forward(self, input):

        out = self.swish(self.fc_1(input))
        out = self.dropout(out)
        out = self.fc_2(out)
        return out