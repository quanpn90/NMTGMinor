import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from onmt.modules.dropout import variational_dropout
from .gaussian import Gaussian, ScaleMixtureGaussian


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

        self.in_proj_weight_mu = Parameter(torch.Tensor(inner_size, model_size))
        self.in_proj_weight_rho = Parameter(torch.Tensor(inner_size, model_size))
        self.in_proj_weight = Gaussian(self.in_proj_weight_mu, self.in_proj_weight_rho)
        self.in_proj_weight_prior = ScaleMixtureGaussian()

        self.out_proj_weight_mu = Parameter(torch.Tensor(model_size, inner_size))
        self.out_proj_weight_rho = Parameter(torch.Tensor(model_size, inner_size))
        self.out_proj_weight = Gaussian(self.out_proj_weight_mu, self.out_proj_weight_rho)
        self.out_proj_weight_prior = ScaleMixtureGaussian()

        self.in_proj_bias_mu = Parameter(torch.Tensor(inner_size))
        self.in_proj_bias_rho = Parameter(torch.Tensor(inner_size))
        self.in_proj_bias = Gaussian(self.in_proj_bias_mu, self.in_proj_bias_rho)
        self.in_proj_bias_prior = ScaleMixtureGaussian()

        self.out_proj_bias_mu = Parameter(torch.Tensor(model_size))
        self.out_proj_bias_rho = Parameter(torch.Tensor(model_size))
        self.out_proj_bias = Gaussian(self.out_proj_bias_mu, self.out_proj_bias_rho)
        self.out_proj_bias_prior = ScaleMixtureGaussian()

        self.reset_parameters()
        try:
            from apex.mlp.mlp import mlp_function
            self.optimized = 1
            self.fast_mlp_func = mlp_function
        except ModuleNotFoundError as e:
            self.optimized = 2

    def reset_parameters(self):
        std_ = math.sqrt(2.0 / (self.model_size + self.inner_size))
        nn.init.normal_(self.in_proj_weight_mu, 0.0, std_)
        nn.init.normal_(self.out_proj_weight_mu, 0.0, std_)
        nn.init.normal_(self.in_proj_bias_mu, 0.0, 0.02)
        nn.init.normal_(self.out_proj_bias_mu, 0.0, 0.02)

        nn.init.uniform_(self.in_proj_weight_rho, -5, -4)
        nn.init.uniform_(self.out_proj_weight_rho, -5, -4)
        nn.init.uniform_(self.in_proj_bias_rho, -5, -4)
        nn.init.uniform_(self.out_proj_bias_rho, -5, -4)

    def forward(self, input, sample=False, calculate_log_probs=True):

        # (MCMC)
        # Sample the weights from the variational posterior distribution q(w)
        in_proj_weight, in_proj_weight_logprob = self.in_proj_weight.sample(
            stochastic=(self.training or sample),
            return_log_prob=(self.training or calculate_log_probs))
        out_proj_weight, out_proj_weight_logprob = self.out_proj_weight.sample(
            stochastic=(self.training or sample),
            return_log_prob=(self.training or calculate_log_probs))
        in_proj_bias, in_proj_bias_logprob = self.in_proj_bias.sample(
            stochastic=(self.training or sample),
            return_log_prob=(self.training or calculate_log_probs))
        out_proj_bias, out_proj_bias_logprob = self.out_proj_bias.sample(
            stochastic=(self.training or sample),
            return_log_prob=(self.training or calculate_log_probs))

        if self.optimized == 2 or not input.is_cuda:
            hidden = F.linear(input, self.in_proj_weight, self.in_proj_bias)
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

        # KL Divergence between prior and (variational) posterior
        self.log_variational_posterior = in_proj_weight_logprob + in_proj_bias_logprob + \
                                         out_proj_weight_logprob + out_proj_bias_logprob
        self.log_prior = self.in_proj_weight_prior.log_prob(in_proj_weight) + \
                         self.in_proj_bias_prior.log_prob(in_proj_bias) + \
                         self.out_proj_weight_prior.log_prob(out_proj_weight) + \
                         self.out_proj_bias_prior.log_prob(out_proj_bias)

        return hidden

