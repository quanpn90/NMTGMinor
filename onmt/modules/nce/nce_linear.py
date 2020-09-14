"""An index linear class for generic NCE module"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose

BACKOFF_PROB=1e-10


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
                 noise_distribution=None, noise_ratio=32, shared_noise=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear = nn.Linear(hidden_size, output_size)
        # self.weight = self.linear.weight
        # self.bias = self.linear.bias
        #self.noise_distribution = noise_distribution
        self.fix_norm = fix_norm  # shouldn't use it with large vocab size
        self.noise_ratio = noise_ratio
        self.shared_noise = shared_noise

        noise_distribution.clamp_(min=BACKOFF_PROB)
        self.alias = AliasMultinomial(noise_distribution)
        self.register_buffer('logprob_noise', noise_distribution.log())

        # if noise distribution is None then create an uniform distribution
        # if self.noise_distribution is None:
        #    print("[INFO] Create an unigram distribution for NCE.")

    def sample_noise(self, len_seq, bsz):

        if self.shared_noise:
            noise_size = (self.noise_ratio, )
            noise_samples = self.alias.draw(self.noise_ratio)  #.expand(*noise_size)
            logprob_noise_noise = self.logprob_noise[noise_samples].view_as(noise_samples)  # K

            # noise_samples = noise_samples.expand(*noise_size).contiguous()
            # logprob_noise_noise = logprob_noise_noise.expand(*noise_size).contiguous()
            # return size [K] and [K]
            return noise_samples, logprob_noise_noise

        else:
            noise_size = (len_seq * bsz, self.noise_ratio)

            # [B x K]
            noise_samples = self.alias.draw(*noise_size)

            # [B x K]
            logprob_noise_noise = self.logprob_noise[noise_samples.view(-1)].view_as(noise_samples)

            return noise_samples, logprob_noise_noise

    def _compute_sample_logits(self, input, weight, bias, target_idx, noise_idx):
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
        emb_weights = weight.index_select(0, indices.view(-1)).view(*indices.size(), -1)
        emb_bias = bias.index_select(0, indices.view(-1)).view(*indices.size())

        # element wise multiplication into [bsz x (1 + K)]
        logits = torch.sum(torch.mul(input, emb_weights), dim=2) + emb_bias

        scores_model_target, scores_model_noise = logits[:, 0].unsqueeze(-1), logits[:, 1:]

        # return size [bsz x 1], [bsz x K]
        return scores_model_target, scores_model_noise

    def _compute_sample_logits_shared(self, input, weight, bias, target_idx, noise_idx):
        """
        :param input: [bsz x hidden_size]
        :param target_idx: [bsz]
        :param noise_idx: [K]
        :return:
        """

        emb_weights = weight.index_select(0, target_idx)  # bsz x hidden_size
        emb_bias = bias.index_select(0, target_idx)  # bsz

        # [bsz*len_seq x hidden_size] x [bsz*len_seq x hidden_size]
        scores_model_target = torch.sum(torch.mul(input, emb_weights), dim=1) + emb_bias  # bsz

        # print(noise_idx.size()) should be K
        noise_weights = weight.index_select(0, noise_idx)  # K x hidden_size
        noise_bias = bias.index_select(0, noise_idx)  # K

        # [bsz*len_seq x hidden_size] \times [hidden_size \times K] -> [bsz*len_seq x K]
        scores_model_noise = torch.addmm(noise_bias, input, noise_weights.t())

        # return [bsz*len_seq x 1] and [bsz*len_seq x K]
        return scores_model_target.unsqueeze(1), scores_model_noise

    def forward(self, output_dicts):
        """
        :param output_dicts: dictionary
        :return:
        """
        input = output_dicts['hidden']
        # for this output layer we need a target during training to compute scores for them specifically
        target = output_dicts['target'] if 'target' in output_dicts else None
        fix_norm = self.fix_norm

        # for large vocbulary, this option will increase memory cost by H x V x 2
        weight = F.normalize(self.linear.weight, dim=-1) if fix_norm else self.linear.weight
        bias = self.linear.bias

        if self.training:
            seq_len, bsz = input.size(0), input.size(1)
            # reshape input and target to 2D and 1D
            input = input.view(seq_len * bsz, input.size(-1))
            target = target.view(seq_len * bsz)

            # sample noises from the noise distribution
            noises, logprob_noise_noise = self.sample_noise(seq_len, bsz)

            # logprob_noise_noise = self.logprob_noise[noises.view(-1)].view_as(noises)  # bsz*len_seq x K
            logprob_noise_target = self.logprob_noise[target].unsqueeze(1)  # bsz*len_seq x 1

            if self.shared_noise:
                # compute scores for targets and noises
                scores_model_target, scores_model_noise = self._compute_sample_logits_shared(input, weight, bias,
                                                                                             target, noises)

                # [1 x K] to [bsz*len_seq x K]
                # scores_model_noise = scores_model_noise.expand(scores_model_target.size(0), self.noise_ratio)
                logprob_noise_noise = logprob_noise_noise.\
                    unsqueeze(0).expand(scores_model_target.size(0), self.noise_ratio)
            else:
                scores_model_target, scores_model_noise = self._compute_sample_logits(input, weight, bias,
                                                                                      target, noises)
                # logprob_noise_noise should have size [len_seq*bsz x noise_ratio] already

            output_dicts['logprob_noise_noise'] = logprob_noise_noise
            output_dicts['logprob_noise_target'] = logprob_noise_target
            output_dicts['scores_model_target'] = scores_model_target
            output_dicts['scores_model_noise'] = scores_model_noise
            # return scores_model_target, scores_model_noise, logprob_noise_target, logprob_noise_noise

        else:
            logits = F.linear(input, weight, bias)
            output_dicts['logits'] = logits

        return output_dicts



