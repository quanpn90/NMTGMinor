import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from onmt.modules.dropout import variational_dropout


class AdaptiveFeedForward(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, model_size, inner_size, factor_size, dropout=0., variational=False, activation='relu',
                 adaptive_type='shared'):
        super().__init__()
        self.model_size = model_size
        self.inner_size = inner_size
        self.factor_size = factor_size
        self.dropout = dropout
        self.bias = True
        self.variational = variational
        self.activation = activation
        self.adaptive_type  = adaptive_type

        self.factor_map = nn.Linear(self.model_size, self.factor_size)

        assert self.activation == 'relu'

        self.in_proj_weight = Parameter(torch.Tensor(inner_size, model_size, factor_size))
        self.out_proj_weight = Parameter(torch.Tensor(model_size, inner_size, factor_size))

        self.in_proj_bias = Parameter(torch.Tensor(inner_size, factor_size))
        self.out_proj_bias = Parameter(torch.Tensor(model_size, factor_size))

        self.reset_parameters()
        try:
            from apex.mlp.mlp import mlp_function
            self.optimized = 1
            self.fast_mlp_func = mlp_function
        except ModuleNotFoundError as e:
            self.optimized = 2

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.in_proj_weight, gain=math.sqrt(2))
        # nn.init.xavier_uniform_(self.out_proj_weight)
        std_ = math.sqrt(2.0 / (self.model_size + self.inner_size))
        nn.init.normal_(self.in_proj_weight, 0.0, std_)
        nn.init.normal_(self.out_proj_weight, 0.0, std_)

        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj_bias, 0.)

    def forward(self, input, factor):

        factor = self.factor_map(factor).squeeze()

        in_proj_weight = torch.mv(self.in_proj_weight.view(-1, self.factor_size), factor)\
            .view(self.in_proj_weight.size(0), self.in_proj_weight.size(1))
        out_proj_weight = torch.mv(self.out_proj_weight.view(-1, self.factor_size), factor)\
            .view(self.out_proj_weight.size(0), self.out_proj_weight.size(1))

        in_proj_bias = torch.mv(self.in_proj_bias, factor)
        out_proj_bias = torch.mv(self.out_proj_bias, factor)

        if self.optimized == 2 or not input.is_cuda:
            hidden = F.linear(input, in_proj_weight, in_proj_bias)
            hidden = torch.relu(hidden)
            if self.variational:
                hidden = variational_dropout(hidden, p=self.dropout, training=self.training)
            else:
                hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            hidden = F.linear(hidden, out_proj_weight, out_proj_bias)
        else:
            # Here weight dropout has to be done instead of dropout because
            # Apex MLP does not support dropout
            weights = [F.dropout(in_proj_weight, p=self.dropout, training=self.training),
                       F.dropout(out_proj_weight, p=self.dropout, training=self.training)]
            biases = [F.dropout(in_proj_bias, p=self.dropout, training=self.training),
                      F.dropout(out_proj_bias, p=self.dropout, training=self.training)]
            seq_len, bsz, hidden_size = input.size(0), input.size(1), input.size(2)
            hidden = self.fast_mlp_func(True, 1, input.view(seq_len*bsz, -1), *weights, *biases)
            hidden = hidden.view(seq_len, bsz, hidden_size)

        return hidden
