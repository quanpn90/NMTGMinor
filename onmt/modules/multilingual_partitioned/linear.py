import torch
import torch.nn.functional as F
import torch.nn as nn
import math


class MPLinear(torch.nn.Module):
    """
    A linear layer with partitioned weights
    """

    # TODO: write gradcheck testing
    def __init__(self, input_size, output_size, factor_size):

        super().__init__()

        self.factor_size = factor_size
        self.input_size = input_size
        self.output_size = output_size

        self.shared_weight = torch.nn.Parameter(torch.Tensor(output_size * input_size, factor_size))
        self.shared_bias = torch.nn.Parameter(torch.Tensor(output_size, factor_size))

        self.reset_parameters()

    def reset_parameters(self, init='normal'):
        if init == 'normal':
            std_ = math.sqrt(2.0 / (self.input_size + self.output_size))
            torch.nn.init.normal_(self.shared_weight, 0.0, std_)
        else:
            std_ = math.sqrt(6.0 / (self.input_size + self.output_size))
            torch.nn.init.uniform_(self.shared_weight, -std_, std_)

        nn.init.constant_(self.shared_bias, 0.)

        # for batch ensemble we init r_i and s_i with random sign vectors

    def forward(self, input, factor):
        """
        :param input: T x B x H
        :param indices: H  (shared factor for the whole minibatch)
        :return:
        """

        assert factor.ndim == 1 and factor.size(0) == self.factor_size

        weight = torch.mv(self.shared_weight, factor).view(self.output_size, self.input_size)
        bias = torch.mv(self.shared_bias, factor)

        input = F.linear(input, weight, bias)

        return input


# Multilingual Factorized Weight
class MPPositionWiseFeedForward(torch.nn.Module):
    """
    Multilingually Partitioned Position Wise Feedforward model
    """

    def __init__(self, model_size, inner_size, dropout=0., variational=False, activation='relu',
                 factor_size=8, rank_size=-1):

        super().__init__()
        self.input_linear = MPLinear(model_size, inner_size, factor_size)
        self.output_linear = MPLinear(inner_size, model_size, factor_size)
        self.variational = variational
        self.dropout = dropout
        self.activation = activation
        self.factor_size = factor_size

        if rank_size == -1:
            rank_size = factor_size

        self.rank_size = rank_size
        # self.factor_to_rank = nn.Linear(self.factor_size, self.rank_size)

        if self.variational:
            from onmt.modules.dropout import variational_dropout
            self.dropout_function = variational_dropout
        else:
            self.dropout_function = F.dropout

    def forward(self, hidden, factor):

        # factor = self.factor_to_rank(factor)

        hidden = self.input_linear(hidden, factor)
        hidden = F.relu(hidden, inplace=True)
        hidden = self.dropout_function(hidden, p=self.dropout, training=self.training)
        hidden = self.output_linear(hidden, factor)
        return hidden

    def reset_parameters(self, init='normal'):

        self.input_linear.reset_parameters(init)
        self.output_linear.reset_parameters(init)


if __name__ == "__main__":

    bsz = 16
    seq_len = 6
    input_size = 16
    output_size = 32
    ensemble = 72
    rank = 2

    input = torch.randn((seq_len, bsz, input_size), requires_grad=True)
    weight = torch.randn((output_size, input_size), requires_grad=True)
    bias = torch.randn((output_size,), requires_grad=True)
    r = torch.randn((bsz, rank, input_size), requires_grad=True)
    s = torch.randn((bsz, rank, output_size), requires_grad=True)

    function = BatchEnsembleLinearFunction.apply

    input = input.double().cuda()
    weight = weight.double().cuda()
    bias = bias.double().cuda()
    r = r.double().cuda()
    s = s.double().cuda()

    print("Gradchecking ...")
    torch.autograd.gradcheck(function, (input, weight, bias, r, s))