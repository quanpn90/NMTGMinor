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
        if not self.glu and self.activation in ['relu']:
            try:
                import fused_mlp
                self.fused = True

                from onmt.modules.mlp.mlp import mlp_function
                self.fast_mlp_func = mlp_function
            except (ModuleNotFoundError, ImportError) as e:
                print(
                    "[INFO] Fused MLP implementation not found. Installation can be found in onmt.modules.mlp/setup.py")
                self.fused = False

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

    def forward(self, input, *args):

        if self.fused and input.is_cuda:
            weights = [self.in_proj_weight,  self.out_proj_weight]
            biases = [self.in_proj_bias, self.out_proj_bias]

            seq_len, bsz, hidden_size = input.size(0), input.size(1), input.size(2)

            if self.activation in ['relu']:
                activation = 1
            else:
                raise NotImplementedError

            hidden = self.fast_mlp_func(1, self.dropout if self.training else 0,
                                        activation, input.view(seq_len * bsz, -1),
                                        *weights, *biases)
            hidden = hidden.view(seq_len, bsz, hidden_size)

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
                                                 inplace=self.activation in ['silu', 'relu', 'swish'])
                else:
                    hidden = F.dropout(hidden, p=self.dropout, training=self.training,
                                       inplace=self.activation in ['silu', 'relu', 'swish'])
            hidden = F.linear(hidden, self.out_proj_weight, self.out_proj_bias)

        return hidden
