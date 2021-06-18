import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.cuda.amp import autocast


class MultilingualLinear(torch.nn.Module):

    def __init__(self, input_size, output_size, n_factors=1, rank=1,
                 use_multiplicative=False,
                 weight_drop=0.0, mfw_activation="none",  no_bias=False):

        super().__init__()

        self.use_multiplicative = use_multiplicative
        self.weight_drop = weight_drop
        self.no_bias = no_bias

        assert (not self.no_bias) or self.use_multiplicative

        self.weight = torch.nn.Parameter(torch.Tensor(output_size, input_size))
        self.bias = torch.nn.Parameter(torch.Tensor(output_size))

        if not self.no_bias:
            self.r = torch.nn.Parameter(torch.Tensor(n_factors, rank, output_size))
            self.s = torch.nn.Parameter(torch.Tensor(n_factors, rank, input_size))

        if use_multiplicative:
            self.rm = torch.nn.Parameter(torch.Tensor(n_factors, 1, output_size))
            self.sm = torch.nn.Parameter(torch.Tensor(n_factors, 1, input_size))

        self.reset_parameters()
        self.mfw_activation = mfw_activation.lower()

    def reset_parameters(self, init='normal'):
        if init == 'normal':
            torch.nn.init.xavier_normal_(self.weight)
        else:
            torch.nn.init.xavier_uniform_(self.weight)

        # for batch ensemble we init r_i and s_i with random sign vectors
        if self.use_multiplicative:
            torch.nn.init.constant_(self.rm, 1.0)
            torch.nn.init.constant_(self.sm, 1.0)

        if not self.no_bias:
            torch.nn.init.normal_(self.r, 0.0, 0.02)
            torch.nn.init.normal_(self.s, 0.0, 0.02)

    def freeze(self):

        if self.use_multiplicative:
            self.rm.requires_grad = False
            self.sm.requires_grad = False

        if not self.no_bias:
            self.r.requires_grad = False
            self.s.requires_grad = False

    def unfreeze(self):

        if self.use_multiplicative:
            self.rm.requires_grad = True
            self.sm.requires_grad = True

        if not self.no_bias:
            self.r.requires_grad = True
            self.s.requires_grad = True

    def get_weight(self, indices, factorize=True):

        weight_ = self.weight

        if indices is None:
            return weight_, self.bias

        if factorize:

            weight_ = F.dropout(self.weight, p=self.weight_drop, training=self.training)

            if indices.size(0) == 1 and len(indices.shape) == 1:

                if self.use_multiplicative:
                    rm = torch.index_select(self.rm, 0, indices).squeeze(0)
                    sm = torch.index_select(self.sm, 0, indices).squeeze(0)
                    weight_ = weight_ * torch.sum(torch.bmm(rm.unsqueeze(-1), sm.unsqueeze(1)), dim=0)

                if self.mfw_activation == "none":
                    weight_ = weight_
                elif self.mfw_activation == "gelu":
                    weight_ = F.gelu(weight_)
                elif self.mfw_activation == "silu":
                    weight_ = F.silu(weight_)
                else:
                    raise NotImplementedError

                if not self.no_bias:
                    r = torch.index_select(self.r, 0, indices).squeeze(0)
                    s = torch.index_select(self.s, 0, indices).squeeze(0)
                    weight_mask = torch.bmm(r.unsqueeze(-1), s.unsqueeze(1))
                    weight_mask = torch.sum(weight_mask, dim=0)
                    weight_ = weight_ + weight_mask

        return weight_, self.bias

    def forward(self, input, indices=None, factorize=True):
        """
        :param factorize:
        :param input: T x B x H
        :param indices: T x B or B
        :return:
        """

        if indices.size(0) == 1 and len(indices.shape) == 1:

            weight_, bias = self.get_weight(indices, factorize=factorize)

            input = F.linear(input, weight_, self.bias)

            return input
        else:
            print(indices.size(), input.size())
            raise NotImplementedError


# Multilingual Factorized Weight
class MFWPositionWiseFeedForward(torch.nn.Module):
    """
    Position Wise Feedforward model with factorized weights
    """

    def __init__(self, model_size, inner_size, dropout=0., variational=False, activation='relu',
                 n_languages=1, rank=1, use_multiplicative=False, weight_drop=0.0, mfw_activation='none',
                 glu=False, no_bias=False):
        super().__init__()

        self.variational = variational
        self.dropout = dropout
        self.activation = activation
        self.n_languages = n_languages
        self.weight_drop = weight_drop
        self.glu = glu
        self.dropout_residual = False
        self.fused = False

        self.input_linear = MultilingualLinear(model_size, inner_size * (2 if glu else 1), n_languages,
                                               rank, use_multiplicative, weight_drop, mfw_activation=mfw_activation,
                                               no_bias=no_bias)
        self.output_linear = MultilingualLinear(inner_size, model_size, n_languages,
                                                rank, use_multiplicative, weight_drop, mfw_activation=mfw_activation,
                                                no_bias=no_bias)

        if self.activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.activation == 'gelu':
            self.act = nn.GELU()
        elif self.activation in ['silu', 'swish']:
            self.act = nn.SiLU(inplace=True)

        if self.variational:
            from onmt.modules.dropout import variational_dropout
            self.dropout_function = variational_dropout
        else:
            self.dropout_function = F.dropout

        # At the moment fused mlp is supported for RELU, SiLU, Swish, GELU and AGELU (approximated GELU)
        if not self.glu and \
                self.activation in ['relu', 'silu', 'swish', 'gelu', 'agelu'] and not self.variational:
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
            elif self.activation == 'gelu':
                from onmt.modules.mlp.mlp import mlp_gelu_function
                if mlp_gelu_function is not None:
                    self.fused_function = mlp_gelu_function
                    self.fused = True
            elif self.activation == 'agelu':
                from onmt.modules.mlp.mlp import mlp_agelu_function
                if mlp_agelu_function is not None:
                    self.fused_function = mlp_agelu_function
                    self.fused = True

    def freeze(self):

        self.input_linear.freeze()
        self.output_linear.freeze()

    def unfreeze(self):

        self.input_linear.unfreeze()
        self.output_linear.unfreeze()

    def forward(self, hidden, indices=None, factorize=True, **kwargs):
        """
        :param factorize:
        :param hidden: tensor [T x B x H]
        :param indices: tensor [1]
        :return:
        """
        if self.fused and hidden.is_cuda:
            in_weight, in_bias = self.input_linear.get_weight(indices, factorize=factorize)
            out_weight, out_bias = self.output_linear.get_weight(indices, factorize=factorize)

            with autocast(enabled=False):
                input = hidden
                weights = [in_weight.half(), out_weight.half()]
                biases = [in_bias.half(), out_bias.half()]

                seq_len, bsz, hidden_size = input.size(0), input.size(1), input.size(2)
                recompute = False

                dropout = self.dropout if self.training else 0.0

                hidden = self.fused_function(dropout, recompute, input.half().view(seq_len * bsz, -1),
                                             *weights, *biases).type_as(input)

                hidden = hidden.view(seq_len, bsz, hidden_size)

            return hidden

        else:
            hidden = self.input_linear(hidden, indices)

            if self.glu:
                hidden, gate = hidden.chunk(2, dim=-1)
                hidden = self.act(hidden) * gate
            else:
                hidden = self.act(hidden)

            hidden = self.dropout_function(hidden, p=self.dropout, training=self.training)
            hidden = self.output_linear(hidden, indices)
            return hidden

    def reset_parameters(self, init='normal'):

        self.input_linear.reset_parameters(init)
        self.output_linear.reset_parameters(init)
