import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from .gaussian import Gaussian, ScaleMixtureGaussian


from ..optimized.relative_self_attention_func import relative_self_attn_func

# from .fast_self_multihead_attn_func          import fast_self_attn_func
# from .fast_self_multihead_attn_norm_add_func import fast_self_attn_norm_add_func
# from apex.normalization.fused_layer_norm     import FusedLayerNorm


if hasattr(torch._C, '_jit_set_profiling_executor'):
    torch._C._jit_set_profiling_executor(False)
if hasattr(torch._C, '_jit_set_profiling_mode'):
    torch._C._jit_set_profiling_mode(False)


@torch.jit.script
def jit_dropout_add(x, residual, prob, is_training):
    # type: (Tensor, Tensor, float, bool) -> Tensor
    out = F.dropout(x, p=prob, training=True)
    out = residual + out
    return out


class RelativeSelfMultiheadAttn(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.bias = True
        self.log_prior = 0
        self.log_variational_posterior = 0

        self.in_proj_weight_mu = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_weight_rho = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_weight = Gaussian(self.in_proj_weight_mu, self.in_proj_weight_rho)
        self.in_proj_weight_prior = ScaleMixtureGaussian()

        self.out_proj_weight_mu = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.out_proj_weight_rho = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.out_proj_weight = Gaussian(self.out_proj_weight_mu, self.out_proj_weight_rho)
        self.out_proj_weight_prior = ScaleMixtureGaussian()

        self.pos_proj_weight_mu = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.pos_proj_weight_rho = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.pos_proj_weight = Gaussian(self.pos_proj_weight_mu, self.pos_proj_weight_rho)
        self.pos_proj_weight_prior = ScaleMixtureGaussian()

        self.in_proj_bias_mu = Parameter(torch.Tensor(3*embed_dim))
        self.in_proj_bias_rho = Parameter(torch.Tensor(3*embed_dim))
        self.in_proj_bias = Gaussian(self.in_proj_bias_mu, self.in_proj_bias_rho)
        self.in_proj_bias_prior = ScaleMixtureGaussian()

        self.out_proj_bias_mu = Parameter(torch.Tensor(embed_dim))
        self.out_proj_bias_rho = Parameter(torch.Tensor(embed_dim))
        self.out_proj_bias = Gaussian(self.out_proj_bias_mu, self.out_proj_bias_rho)
        self.out_proj_bias_prior = ScaleMixtureGaussian()

        self.pos_proj_bias_mu = Parameter(torch.Tensor(embed_dim))
        self.pos_proj_bias_rho = Parameter(torch.Tensor(embed_dim))
        self.pos_proj_bias = Gaussian(self.pos_proj_bias_mu, self.pos_proj_bias_rho)
        self.pos_proj_bias_prior = ScaleMixtureGaussian()

        self.r_w_bias_mu = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
        self.r_w_bias_rho = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
        self.r_w_bias = Gaussian(self.r_w_bias_mu, self.r_w_bias_rho)
        self.r_w_bias_prior = ScaleMixtureGaussian()

        self.r_r_bias_mu = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
        self.r_r_bias_rho = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
        self.r_r_bias = Gaussian(self.r_r_bias_mu, self.r_r_bias_rho)
        self.r_r_bias_prior = ScaleMixtureGaussian()

        self.reset_parameters()
        self.attn_func = relative_self_attn_func

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.in_proj_weight, gain=math.sqrt(2))
        # nn.init.xavier_uniform_(self.out_proj_weight)
        std_ = math.sqrt(2.0 / (self.embed_dim + self.embed_dim))
        nn.init.normal_(self.in_proj_weight_mu, 0.0, std_)
        nn.init.normal_(self.out_proj_weight_mu, 0.0, std_)
        nn.init.normal_(self.pos_proj_weight_mu, 0.0, std_)

        nn.init.constant_(self.in_proj_bias_mu, 0.)
        nn.init.constant_(self.out_proj_bias_mu, 0.)
        nn.init.constant_(self.pos_proj_bias_mu, 0.)

        nn.init.normal_(self.r_w_bias_mu, 0.0, std_)
        nn.init.normal_(self.r_r_bias_mu, 0.0, std_)

        nn.init.uniform_(self.in_proj_weight_rho, -5, -4)
        nn.init.uniform_(self.out_proj_weight_rho, -5, -4)
        nn.init.uniform_(self.pos_proj_weight_rho, -5, -4)
        nn.init.uniform_(self.in_proj_bias_rho, -5, -4)
        nn.init.uniform_(self.out_proj_bias_rho, -5, -4)
        nn.init.uniform_(self.pos_proj_bias_rho, -5, -4)
        nn.init.uniform_(self.r_w_bias_rho, -5, -4)
        nn.init.uniform_(self.r_r_bias_rho, -5, -4)

    def forward(self, input, pos, key_padding_mask=None, attn_mask=None, mems=None,
                incremental=False, incremental_cache=None, sample=False, calculate_log_probs=True):

        # (MCMC)
        # Sample the weights from the variational posterior distribution q(w)
        in_proj_weight, in_proj_weight_logprob = self.in_proj_weight.sample(
            stochastic=(self.training or sample),
            return_log_prob=(self.training or calculate_log_probs))
        out_proj_weight, out_proj_weight_logprob = self.out_proj_weight.sample(
            stochastic=(self.training or sample),
            return_log_prob=(self.training or calculate_log_probs))
        pos_proj_weight, pos_proj_weight_logprob = self.pos_proj_weight.sample(
            stochastic=(self.training or sample),
            return_log_prob=(self.training or calculate_log_probs))

        in_proj_bias, in_proj_bias_logprob = self.in_proj_bias.sample(
            stochastic=(self.training or sample),
            return_log_prob=(self.training or calculate_log_probs))
        out_proj_bias, out_proj_bias_logprob = self.out_proj_bias.sample(
            stochastic=(self.training or sample),
            return_log_prob=(self.training or calculate_log_probs))
        pos_proj_bias, pos_proj_bias_logprob = self.pos_proj_bias.sample(
            stochastic=(self.training or sample),
            return_log_prob=(self.training or calculate_log_probs))

        r_w_bias, r_w_bias_logprob = self.r_w_bias.sample(
            stochastic=(self.training or sample),
            return_log_prob=(self.training or calculate_log_probs))
        r_r_bias, r_r_bias_logprob = self.r_r_bias.sample(
            stochastic=(self.training or sample),
            return_log_prob=(self.training or calculate_log_probs))

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
        self.log_variational_posterior = in_proj_weight_logprob + in_proj_bias_logprob + \
                                         out_proj_weight_logprob + out_proj_bias_logprob + \
                                         pos_proj_weight_logprob + pos_proj_bias_logprob + \
                                         r_r_bias_logprob + r_r_bias_logprob

        self.log_prior = self.in_proj_weight_prior.log_prob(in_proj_weight) + \
                         self.in_proj_bias_prior.log_prob(in_proj_bias) + \
                         self.out_proj_weight_prior.log_prob(out_proj_weight) + \
                         self.out_proj_bias_prior.log_prob(out_proj_bias) + \
                         self.pos_proj_weight_prior.log_prob(pos_proj_weight) + \
                         self.pos_proj_bias_prior.log_prob(pos_proj_bias) + \
                         self.r_r_bias_prior.log_prob(r_r_bias) + \
                         self.r_w_bias_prior.log_prob(r_w_bias)

        return outputs, coverage
