import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from onmt.modules.dropout import variational_dropout


class PositionWiseFeedForward(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, model_size, inner_size, dropout=0., variational=False, activation='relu'):
        super().__init__()
        self.model_size = model_size
        self.inner_size = inner_size
        self.dropout = dropout
        self.bias = True
        self.variational = variational
        self.activation = activation

        self.in_proj_weight = Parameter(torch.Tensor(inner_size, model_size))
        self.out_proj_weight = Parameter(torch.Tensor(model_size, inner_size))

        self.in_proj_bias = Parameter(torch.Tensor(inner_size))
        self.out_proj_bias = Parameter(torch.Tensor(model_size))

        self.reset_parameters()
        try:
            from apex.mlp.mlp import mlp_function
            self.optimized = 2
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

    def forward(self, input):

        if self.optimized == 2 or not input.device.is_cuda:
            hidden = F.linear(input, self.in_proj_weight, self.in_proj_bias)
            hidden = torch.relu(hidden)
            if self.variational:
                hidden = variational_dropout(hidden, p=self.dropout, training=self.training)
            else:
                hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            hidden = F.linear(hidden, self.out_proj_weight, self.out_proj_bias)
            hidden = torch.relu(hidden)
        else:
            # Here weight dropout has to be done instead of dropout because
            # Apex MLP does not support dropout

            weights = [F.dropout(self.in_proj_weight, p=self.dropout, training=self.training),
                       F.dropout(self.out_proj_weight, p=self.dropout, training=self.training)]
            biases = [F.dropout(self.in_proj_bias, p=self.dropout, training=self.training),
                      F.dropout(self.out_proj_bias, p=self.dropout, training=self.training)]
            seq_len, bsz, hidden_size = input.size(0), input.size(1), input.size(2)
            hidden = self.fast_mlp_func(True, 1, input.view(seq_len*bsz, -1), *weights, *biases)
            hidden = hidden.view(seq_len, bsz, hidden_size)

        return hidden
