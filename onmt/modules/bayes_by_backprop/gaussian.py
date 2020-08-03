import torch
import torch.nn.functional as F
import numpy
import math
import torch.nn as nn

log_sqrt_2pi = math.log(math.sqrt(2 * math.pi))


class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        # sigma = log(exp(rho) + 1) = softplus
        return F.softplus(self.rho, beta=1)  # this should be a numerically better option
        # return torch.log1p(torch.exp(self.rho))

    def sample(self, stochastic=False, return_log_prob=False):

        wsize = self.mu.numel()
        sigma = self.sigma
        if stochastic:
            # epsilon = self.normal.sample(self.rho.size()).type_as(self.mu)
            epsilon = torch.rand_like(self.mu)
            var = sigma * epsilon
            # return torch.addcmul(self.mu, self.sigma, epsilon)
            w = self.mu + var
        else:
            w = self.mu
            var = 0
        if not return_log_prob:
            return w, 0
        else:
            sigma = sigma.float()
            log_prob = (- log_sqrt_2pi
                        - torch.log(sigma)
                        - (var ** 2) / (2 * sigma ** 2)).sum()
            # log_prob = (-(var ** 2) / (2 * sigma) - torch.log(sigma) - math.log(math.sqrt(2 * math.pi))).sum()
            return w, log_prob

    def log_prob(self, input):

        sigma = self.sigma.float()
        input = input.float()
        return (math.log(math.sqrt(2 * math.pi))
                - torch.log(sigma)
                - ((input - self.mu) ** 2) / (2 * sigma ** 2)).sum()


class ScaleMixtureGaussian(object):
    def __init__(self, pi=None, sigma1=None, sigma2=None):
        super().__init__()
        from onmt.constants import neg_log_sigma1, neg_log_sigma2, prior_pi

        sigma1 = torch.cuda.FloatTensor([math.exp(-neg_log_sigma1)]) if sigma1 is None else sigma1
        sigma2 = torch.cuda.FloatTensor([math.exp(-neg_log_sigma2)]) if sigma2 is None else sigma2
        pi = prior_pi if pi is None else pi

        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, input):
        # input = input.float()  # for exp better to cast to float
        # print(input.type())
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()
