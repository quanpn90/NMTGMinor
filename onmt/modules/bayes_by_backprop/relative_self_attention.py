import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from .gaussian import Gaussian, ScaleMixtureGaussian
from .utils import flatten_list, unflatten
from ..optimized.relative_self_attention_func import relative_self_attn_func

# from .fast_self_multihead_attn_func          import fast_self_attn_func
# from .fast_self_multihead_attn_norm_add_func import fast_self_attn_norm_add_func
# from apex.normalization.fused_layer_norm     import FusedLayerNorm


class RelativeSelfMultiheadAttn(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, model_size, num_heads, dropout=0.):
        super().__init__()
        self.model_size = model_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = model_size // num_heads
        assert self.head_dim * num_heads == self.model_size, "model_size must be divisible by num_heads"
        self.bias = True
        self.log_prior = 0
        self.log_variational_posterior = 0

        in_proj_weight_mu = torch.Tensor(3 * model_size, model_size)
        in_proj_weight_rho = torch.Tensor(3 * model_size, model_size)

        out_proj_weight_mu = torch.Tensor(model_size, model_size)
        out_proj_weight_rho = torch.Tensor(model_size, model_size)

        pos_proj_weight_mu = torch.Tensor(model_size, model_size)
        pos_proj_weight_rho = torch.Tensor(model_size, model_size)

        in_proj_bias_mu = torch.Tensor(3*model_size)
        in_proj_bias_rho = torch.Tensor(3*model_size)

        out_proj_bias_mu = torch.Tensor(model_size)
        out_proj_bias_rho = torch.Tensor(model_size)

        pos_proj_bias_mu = torch.Tensor(model_size)
        pos_proj_bias_rho = torch.Tensor(model_size)

        r_w_bias_mu = torch.Tensor(self.num_heads, self.head_dim)
        r_w_bias_rho = torch.Tensor(self.num_heads, self.head_dim)

        r_r_bias_mu = torch.Tensor(self.num_heads, self.head_dim)
        r_r_bias_rho = torch.Tensor(self.num_heads, self.head_dim)

        mu, self.indices, self.shapes = flatten_list([in_proj_weight_mu, out_proj_weight_mu, pos_proj_weight_mu,
                                                      in_proj_bias_mu, out_proj_bias_mu, pos_proj_bias_mu,
                                                      r_w_bias_mu, r_r_bias_mu])

        rho, _, _ = flatten_list([in_proj_weight_rho, out_proj_weight_rho, pos_proj_weight_rho,
                                  in_proj_bias_rho, out_proj_bias_rho, pos_proj_bias_rho,
                                  r_w_bias_rho, r_r_bias_rho])
        self.mu = Parameter(mu)
        self.rho = Parameter(rho)
        self.weight = Gaussian(self.mu, self.rho)
        self.weight_prior = ScaleMixtureGaussian()

        self.reset_parameters()
        self.attn_func = relative_self_attn_func

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.in_proj_weight, gain=math.sqrt(2))
        # nn.init.xavier_uniform_(self.out_proj_weight)
        std_ = math.sqrt(2.0 / (self.model_size + self.model_size))
        nn.init.normal_(self.mu, 0.0, std_)
        nn.init.normal_(self.rho, -5, 0.1)
        # nn.init.uniform_(self.rho, -6, -5)

    def forward(self, input, pos, key_padding_mask=None, attn_mask=None, mems=None,
                incremental=False, incremental_cache=None, sample=False, calculate_log_probs=False):

        calculate_log_probs = calculate_log_probs or self.training
        sample = sample or self.training
        # (MCMC)
        # Sample the weights from the variational posterior distribution q(w)
        sampled_weights, log_variational_posterior = self.weight.sample(sample, calculate_log_probs)

        in_proj_weight, out_proj_weight, pos_proj_weight, \
            in_proj_bias, out_proj_bias, pos_proj_bias, \
            r_w_bias, r_r_bias = unflatten(sampled_weights, self.indices, self.shapes)

        if key_padding_mask is not None:
            assert (attn_mask is None), "ERROR attn_mask and key_padding_mask should not be both defined!"
            mask = key_padding_mask
            if len(mask.shape) == 3:
                mask = mask.squeeze(0).transpose(0, 1)
        elif attn_mask is not None:
            mask = attn_mask
            if len(mask.shape) == 3:
                mask = mask.squeeze(-1)
        else:
            mask = None

        is_training = self.training

        outputs, coverage = self.attn_func(input, pos, attn_mask is not None, is_training, self.num_heads,
                                           in_proj_weight, out_proj_weight, pos_proj_weight,
                                           in_proj_bias, out_proj_bias, pos_proj_bias,
                                           r_w_bias, r_r_bias,
                                           mask, self.dropout,
                                           incremental, incremental_cache, False, False)
        # last False is double precision

        # KL Divergence between prior and (variational) posterior
        if calculate_log_probs:
            self.log_variational_posterior = log_variational_posterior

            self.log_prior = self.weight_prior.log_prob(sampled_weights)

        return outputs, coverage
