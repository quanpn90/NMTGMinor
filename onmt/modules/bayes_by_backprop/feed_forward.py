import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from onmt.modules.dropout import variational_dropout
from .gaussian import Gaussian, ScaleMixtureGaussian
from .utils import flatten_list, unflatten


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

        # two variables to record the (sum) of priors for all linear variables
        self.log_prior = 0
        self.log_variational_posterior = 0

        in_proj_weight_mu = torch.Tensor(inner_size, model_size)
        in_proj_weight_rho = torch.Tensor(inner_size, model_size)

        out_proj_weight_mu = torch.Tensor(model_size, inner_size)
        out_proj_weight_rho = torch.Tensor(model_size, inner_size)

        in_proj_bias_mu = torch.Tensor(inner_size)
        in_proj_bias_rho = torch.Tensor(inner_size)

        out_proj_bias_mu = torch.Tensor(model_size)
        out_proj_bias_rho = torch.Tensor(model_size)

        mu, self.indices, self.shapes = \
            flatten_list([in_proj_weight_mu, out_proj_weight_mu, in_proj_bias_mu, out_proj_bias_mu])
        rho, _, _ = flatten_list([in_proj_weight_rho, out_proj_weight_rho, in_proj_bias_rho, out_proj_bias_rho])
        self.mu = Parameter(mu)
        self.rho = Parameter(rho)
        self.weight = Gaussian(self.mu, self.rho)
        self.weight_prior = ScaleMixtureGaussian()

        self.reset_parameters()
        try:
            from apex.mlp.mlp import mlp_function
            self.optimized = 2
            self.fast_mlp_func = mlp_function
        except ModuleNotFoundError as e:
            self.optimized = 2

    def reset_parameters(self):
        std_ = math.sqrt(2.0 / (self.model_size + self.inner_size))
        nn.init.normal_(self.mu, 0.0, std_)

        nn.init.normal_(self.rho, -5, 0.1)

    def forward(self, input, sample=False, calculate_log_probs=False):

        calculate_log_probs = calculate_log_probs or self.training
        sample = sample or self.training
        # (MCMC)
        # Sample the weights from the variational posterior distribution q(w)
        sampled_weights, log_variational_posterior = self.weight.sample(sample, calculate_log_probs)

        in_proj_weight, out_proj_weight, in_proj_bias, out_proj_bias = \
            unflatten(sampled_weights, self.indices, self.shapes)

        if self.optimized == 2 or not input.is_cuda:
            hidden = F.linear(input, in_proj_weight, in_proj_bias)
            hidden = F.relu(hidden, inplace=True)
            if self.variational:
                hidden = variational_dropout(hidden, p=self.dropout, training=self.training)
            else:
                hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            hidden = F.linear(hidden, out_proj_weight, out_proj_bias)
        else:
            # Apex MLP does not support dropout so instead we use dropconnect
            # Theoretically they should be the same ^^
            weights = [in_proj_weight,
                       out_proj_weight]
            biases = [in_proj_bias,
                      out_proj_bias]
            seq_len, bsz, hidden_size = input.size(0), input.size(1), input.size(2)
            # True = bias, 1 = relu
            hidden = self.fast_mlp_func(True, 1, input.view(seq_len*bsz, -1), *weights, *biases)
            hidden = hidden.view(seq_len, bsz, hidden_size)

        if calculate_log_probs:
            # KL Divergence between prior and (variational) posterior
            self.log_variational_posterior = log_variational_posterior

            self.log_prior = self.weight_prior.log_prob(sampled_weights)

        return hidden

