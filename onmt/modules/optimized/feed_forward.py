import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from onmt.modules.dropout import variational_dropout, ReLUDropout
from onmt.modules.swish import SiLU
import onmt


class PositionWiseFeedForward(nn.Module):
    """Two-layer Feed-forward neural network"""

    def __init__(self, model_size, inner_size, dropout=0., variational=False,
                 activation='relu', glu=False, weight_drop=0.0):
        super().__init__()
        self.model_size = model_size
        self.inner_size = inner_size
        self.dropout = dropout
        self.bias = True
        self.variational = variational
        self.activation = activation
        self.glu = glu
        self.weight_drop = weight_drop
        self.autograd = False

        if self.activation == 'relu':
            if self.glu:
                self.act = nn.ReLU(inplace=True)
            else:
                self.act = ReLUDropout(p=self.dropout, variational=self.variational, batch_first=False)
        elif self.activation == 'gelu':
            self.act = nn.GELU()
        elif self.activation in ['silu', 'swish']:
            self.act = SiLU()
        elif self.activation in ['sigmoid']:
            if self.glu:
                self.act = nn.functional.glu
            else:
                print("Sigmoid activation function is recommended to be used with -glu")
                raise NotImplementedError

        self.in_proj_weight = Parameter(torch.Tensor(inner_size * (2 if glu else 1), model_size))
        self.out_proj_weight = Parameter(torch.Tensor(model_size, inner_size))

        self.in_proj_bias = Parameter(torch.Tensor(inner_size * (2 if glu else 1)))
        self.out_proj_bias = Parameter(torch.Tensor(model_size))

        self.reset_parameters()
        self.optimized = 2

        self.fused = False

        # At the moment fused mlp is only supported for non-GLU and ReLU
        if not self.glu and self.activation in ['relu', 'silu', 'swish'] and not self.variational:
            if self.activation == 'relu':
                from onmt.modules.mlp.mlp import mlp_relu_function
                if mlp_relu_function is not None:
                    self.fused_function = mlp_relu_function
                    self.fused = True
            elif self.activation in ['silu', 'swish']:
                from onmt.modules.mlp.mlp import mlp_silu_function
                if mlp_silu_function is not None:
                    self.fused_function = mlp_silu_function
                    self.fused = True
            # elif self.activation == 'gelu':
            #     from onmt.modules.mlp.mlp import mlp_gelu_function
            #     if mlp_gelu_function is not None:
            #         self.fused_function = mlp_gelu_function
            #         self.fused = True

            # self.fused = True
            # self.mlp_relu_function = mlp_relu_function

    def reset_parameters(self, init='normal'):
        if init == 'normal':
            std_ = math.sqrt(2.0 / (self.model_size + self.inner_size))
            nn.init.normal_(self.in_proj_weight, 0.0, std_)
            nn.init.normal_(self.out_proj_weight, 0.0, std_)
        else:
            std_ = math.sqrt(6.0 / (self.model_size + self.inner_size))
            nn.init.uniform_(self.in_proj_weight, -std_, std_)
            nn.init.uniform_(self.out_proj_weight, -std_, std_)

        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.constant_(self.out_proj_bias, 0.0)

    def convert_autograd(self):

        if self.autograd:
            return

        with torch.no_grad():
            self.autograd = True
            self.linear_in = torch.nn.Linear(self.model_size, self.inner_size)
            self.linear_out = torch.nn.Linear(self.inner_size, self.model_size)

            self.linear_in.weight.copy_(self.in_proj_weight)
            self.linear_in.bias.copy_(self.in_proj_bias)
            self.linear_out.weight.copy_(self.out_proj_weight)
            self.linear_out.bias.copy_(self.out_proj_bias)

            del self.in_proj_weight
            del self.in_proj_bias
            del self.out_proj_weight
            del self.out_proj_bias

    def forward(self, input, *args):

        if self.fused and input.is_cuda:
            weights = [self.in_proj_weight, self.out_proj_weight]
            biases = [self.in_proj_bias, self.out_proj_bias]

            seq_len, bsz, hidden_size = input.size(0), input.size(1), input.size(2)

            dropout = self.dropout if self.training else 0.0

            hidden = self.fused_function(dropout, input.view(seq_len * bsz, -1),
                                                       *weights, *biases)
            hidden = hidden.view(seq_len, bsz, hidden_size)

        else:
            if self.autograd:
                hidden = self.linear_in(input)
            else:
                hidden = F.linear(input, self.in_proj_weight, self.in_proj_bias)

            if self.glu and self.activation != 'sigmoid':
                hidden, gate = hidden.chunk(2, dim=-1)
                hidden = self.act(hidden) * gate
            else:  # GLU function
                hidden = self.act(hidden)

            if not (not self.glu and self.activation == 'relu'):
                if self.variational:
                    hidden = variational_dropout(hidden, p=self.dropout, training=self.training,
                                                 inplace=self.activation in ['silu', 'relu', 'swish', 'gelu'])
                else:
                    hidden = F.dropout(hidden, p=self.dropout, training=self.training,
                                       inplace=self.activation in ['silu', 'relu', 'swish', 'gelu'])

            if self.autograd:
                hidden = self.linear_out(hidden)
            else:
                hidden = F.linear(hidden, self.out_proj_weight, self.out_proj_bias)

        return hidden
