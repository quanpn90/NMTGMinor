import math
import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer

PI = 0.5
SIGMA_1 = torch.cuda.FloatTensor([math.exp(-0)])
SIGMA_2 = torch.cuda.FloatTensor([math.exp(-6)])


class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)
        self.sigma = math.log(1 + math.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.rho.size())
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - math.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()


class BayesianWeight(object):
    def __init__(self, params,
                 prior="standard_normal"):

        assert prior in ["standard_normal", "mixture"]

        param_list = list(params)

        # first the parameters and their parameters are flattended into a very large tensor
        total_num_params = sum([param.numel() for param in param_list])
        self.flattened_params = torch.nn.Parameter(param_list[0].new_zeros(total_num_params))
        self.flattened_params.grad = param_list[0].data.new_zeros(total_num_params)

        offset = 0
        for param in param_list:
            numel = param.numel()
            flattened_chunk = self.flattened_params.data[offset:offset + numel]

            # copy the data from param to flattened
            flattened_chunk.view_as(param.data).copy_(param.data)
            param.data = flattened_chunk.view_as(param.data)

            # so that part above works and set the param storage to the flattened chunk
            # now we need to find out why the lower part doesn't work

            param.grad = param.data.new_zeros(param.data.size())
            param.grad.data = self.flattened_params.grad.data[offset:offset + numel].view_as(param.data)
            # param.grad = self.flattened_params.grad[offset:offset + numel].view_as(param)

            # grad and data should have the same size, shouldn't they?
            offset += numel

        # similarly, initialize the mean and std parameters with the flattened tensor
        self.mean = torch.nn.Parameter(self.flattened_params.data.new_zeros(self.flattened_params.size()))
        self.mean.data.copy_(self.flattened_params.data)

        self.mean.grad = self.flattened_params.data.new_zeros(self.flattened_params.size())

        # std is initialized as zero (or should we use small uniforms?)
        self.std = torch.nn.Parameter(self.flattened_params.new_zeros(self.flattened_params.size()))
        self.std.data.uniform_(-5, -4)
        self.std.grad = self.flattened_params.new_zeros(self.flattened_params.size())

        # pre-allocate epsilon for fast generation
        self.eps = self.flattened_params.new_zeros(self.flattened_params.size())

        self.prior_type = prior
        if prior == "mixture":
            self.prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)

        # self.posterior = Gaussian(self.mean, self.std)

        # should we need a generator?
        # self.generator = torch.Generator(device=device)

    def parameters(self):

        return [self.mean, self.std]

    def zero_grad(self):

        self.mean.grad.zero_()
        self.std.grad.zero_()

    def forward(self, training=True):
        # this function should be called before the actual model forward()

        # in the forward pass:
        with torch.no_grad():
            # zero the grad for the weights
            self.flattened_params.grad.data.zero_()
            # self.mean.data.copy_(self.flattened_params.data)

            if training:
                # sample eps for reparameterization trick
                self.eps.normal_()

                var = torch.nn.functional.softplus(self.std).mul_(self.eps)

                # w = mean + eps * var
                self.flattened_params.data.copy_(self.mean.data).add_(var.data)

                # at this point we have generated weights for a bayesian neural net
                # after that: we forward the model with the generated weights
                # and then obtain the gradient w.r.t to the flattened weights
            else:
                # during evaluation we probably use the mean (for deterministically)
                self.flattened_params.data.copy_(self.mean.data)

        return

    def backward(self):
        # performed after the backward pass in the main model

        with torch.no_grad():
            # self.mean.data.copy_(self.flattened_weights.data)

            # accumulate the grad from w to mean
            self.mean.grad.data.add_(self.flattened_params.grad.data)

            # derivatives of the softplus function F(x) is sigmoid(x)
            var_grad = torch.nn.functional.sigmoid(self.std.data).mul_(self.eps).mul_(self.flattened_params.grad.data)

            # accumulate the grad from w to std
            self.std.grad.data.add_(var_grad.data)

    def kl_divergence(self, n_samples=1):
        # this function should be called before synchronization !!! (due to randomness)

        kl_loss = 0

        if self.prior_type == "mixture":

            def calculate_log_posterior(x, mean, var):
                return (-math.log(math.sqrt(2 * math.pi))
                         - torch.log(var)
                         - ((x - mean) ** 2) / (2 * var ** 2)).sum()

            # no closed form estimation is possible
            # monte carlo sampling
            for n in range(n_samples):

                # randomize
                if n_samples > 1 and n > 0:
                    with torch.no_grad():
                        self.eps.normal_()
                var = torch.nn.functional.softplus(self.std).mul_(self.eps)
                x = self.mean + var

                log_prior = self.prior.log_prob(x)
                log_posterior = calculate_log_posterior(x, self.mean, var)

                kl_loss += log_posterior - log_prior

            return kl_loss

        elif self.prior_type == "standard_normal":

            # closed form estimation:
            # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
            # variance =
            # std = self.std
            var = self.std ** 2

            kl_loss = 0.5 * torch.sum(-torch.log(var) + var + self.mean ** 2 - 1)

        return kl_loss
