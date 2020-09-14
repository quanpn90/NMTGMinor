"""An index linear class for generic NCE module"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AliasMultinomial(torch.nn.Module):
    ''' Alias sampling method to speedup multinomial sampling
    The alias method treats multinomial sampling as a combination of uniform sampling and
    bernoulli sampling. It achieves significant acceleration when repeatedly sampling from
    the saved multinomial distribution.
    Attributes:
        - probs: the probability density of desired multinomial distribution
    Refs:
        - https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    '''
    def __init__(self, probs):
        super(AliasMultinomial, self).__init__()

        assert isclose(probs.sum().item(), 1), 'The noise distribution must sum to 1'
        cpu_probs = probs.cpu()
        K = len(probs)

        # such a name helps to avoid the namespace check for nn.Module
        self_prob = [0] * K
        self_alias = [0] * K

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for idx, prob in enumerate(cpu_probs):
            self_prob[idx] = K*prob
            if self_prob[idx] < 1.0:
                smaller.append(idx)
            else:
                larger.append(idx)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self_alias[small] = large
            self_prob[large] = (self_prob[large] - 1.0) + self_prob[small]

            if self_prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self_prob[last_one] = 1

        self.register_buffer('prob', torch.Tensor(self_prob))
        self.register_buffer('alias', torch.LongTensor(self_alias))

    def draw(self, *size):
        """Draw N samples from multinomial
        Args:
            - size: the output size of samples
        """
        max_value = self.alias.size(0)

        kk = self.alias.new(*size).random_(0, max_value).long().view(-1)
        prob = self.prob[kk]
        alias = self.alias[kk]
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob).long()
        oq = kk.mul(b)
        oj = alias.mul(1 - b)

        return (oq + oj).view(size)


class NCELinear(nn.Module):

    def __init__(self, hidden_size, output_size, fix_norm=False,
                 noise_distribution=None, noise_ratio=32, logz=9, shared_noise=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear = nn.Linear(hidden_size, output_size)
        self.weight = self.linear.weight
        self.bias = self.linear.bias
        self.noise_distribution = noise_distribution
        self.fix_norm = fix_norm  # shouldn't use it with large vocab size
        self.logz = logz
        self.noise_ratio = noise_ratio
        self.shared_noise = shared_noise

        # if noise distribution is None then create an uniform distribution
        if self.noise_distribution is None:
            print("[INFO] Create an unigram distribution for NCE.")

    def _compute_sample_logits(self, input, target_idx, noise_idx):
        """
        :param input: [bsz x hidden_size]
        :param target_idx: [bsz]
        :param noise_idx: [bsz x K]
        :return:
        """
        # bsz x hidden_size -> bsz x 1 x hidden_size
        input = input.unsqueeze(1)
        # bsz -> bsz x 1
        target_idx = target_idx.unsqueeze(1)

        indices = torch.cat([target_idx, noise_idx], dim=-1)  # bsz x (K+1)

        # bsz x (K+1) x H
        emb_weights = self.weight.index_select(0, indices.view(-1)).view(*indices.size(), -1)
        emb_bias = self.bias.index_select(0, indices.view(-1)).view(*indices.size())

        # element wise multiplication into [bsz x (1 + K)]
        logits = torch.sum(torch.mul(input, emb_weights), dim=2) + emb_bias

        scores_model_target, scores_model_noise = logits[:, 0].unsqueeze(-1), logits[:, :, 1:]

        return scores_model_target, scores_model_noise

    def _compute_sample_logits_shared(self, input, target_idx, noise_idx):
        """
        :param input: [bsz x hidden_size]
        :param target_idx: [bsz]
        :param noise_idx: [K]
        :return:
        """

        emb_weights = self.weight.index_select(0, target_idx)  # bsz x hidden_size
        emb_bias = self.bias.index_select(0, target_idx)  # bsz

        # [bsz x hidden_size] x [bsz x hidden_size]
        scores_model_target = torch.sum(torch.mul(input, emb_weights), dim=1) + emb_bias  # bsz

        noise_weights = self.weight.index_select(0, noise_idx)  # K x hidden_size
        noise_bias = self.noise.index_select(0, noise_idx)  # K

        # [bsz x hidden_size] \times [hidden_size \times K] -> [bsz x K]
        scores_model_noise = torch.addmm(noise_bias, input, noise_weights.t())

        return scores_model_target.unsqueeze(0), scores_model_noise

    def forward(self, output_dicts):
        """
        :param output_dicts: dictionary
        :return:
        """
        input = output_dicts['hidden']
        target = output_dicts['target']  # for this module we need a target during training
        fix_norm = self.fix_norm

        weights = F.normalize(self.weight, dim=-1) if fix_norm else self.weight
        bias = self.bias

        if self.training:
            raise NotImplementedError

            # reshape input and target to 2D and 1D

            # sample noises from the noise distribution

            # compute scores for targets and noises

            # return scores_model_target, scores_model_noise, logprob_noise_target, logprob_noise_noise

        else:
            logits = F.linear(input, weights, bias)

            output_dicts['logits'] = logits
            # return logits



