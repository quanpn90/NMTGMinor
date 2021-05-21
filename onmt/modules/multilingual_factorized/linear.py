import torch
import torch.nn.functional as F
import torch.nn as nn


class MultilingualLinear(torch.nn.Module):

    def __init__(self, input_size, output_size, n_factors=1, rank=1,
                 use_multiplicative=False,
                 weight_drop=0.0, mfw_activation="none",  no_bias=False):

        super().__init__()

        self.use_multiplicative = use_multiplicative
        self.weight_drop = weight_drop
        self.no_bias = no_bias

        assert (not self.no_bias) or self.use_multiplicative

        self.weight = torch.nn.Parameter(torch.Tensor(input_size, output_size))
        self.bias = torch.nn.Parameter(torch.Tensor(output_size))

        if not self.no_bias:
            self.r = torch.nn.Parameter(torch.Tensor(n_factors, rank, input_size))
            self.s = torch.nn.Parameter(torch.Tensor(n_factors, rank, output_size))

        if use_multiplicative:
            self.rm = torch.nn.Parameter(torch.Tensor(n_factors, 1, input_size))
            self.sm = torch.nn.Parameter(torch.Tensor(n_factors, 1, output_size))

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

    def forward(self, input, indices=None):
        """
        :param input: T x B x H
        :param indices: T x B or B
        :return:
        """
        bsz = input.size(1)
        seq_len = input.size(0)

        weight_ = F.dropout(self.weight, p=self.weight_drop, training=self.training)

        if indices.size(0) == 1 and len(indices.shape) == 1:


            # weight_mask = torch.sum(torch.einsum('bi,bj->bij', (s, r)), dim=0)
            # weight_mask = torch.bmm(s.unsqueeze(-1), r.unsqueeze(1))
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

            input = F.linear(input, weight_.t(), self.bias)
            # input = torch.addmm(self.bias, input.view(-1, input.size(-1)), weight_)
            # input = input.view(seq_len, bsz, input.size(-1))
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

    def freeze(self):

        self.input_linear.freeze()
        self.output_linear.freeze()

    def unfreeze(self):

        self.input_linear.unfreeze()
        self.output_linear.unfreeze()

    def forward(self, hidden, indices=None):
        """
        :param hidden: tensor [T x B x H]
        :param indices: tensor [1]
        :return:
        """

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