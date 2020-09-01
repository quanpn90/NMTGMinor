import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from ..optimized.encdec_attention_func import encdec_attn_func
from .gaussian import Gaussian, ScaleMixtureGaussian
from .utils import flatten_list, unflatten


class EncdecMultiheadAttn(nn.Module):
    """Multi-headed encoder-decoder attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, num_heads, embed_dim, attn_drop=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = attn_drop
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.bias = False
        self.scaling = self.head_dim ** -0.5  # this value is hardcoded in the "fast" implementation

        # two variables to record the (sum) of priors for all linear variables
        self.log_prior = 0
        self.log_variational_posterior = 0

        # Q linear mapping weight

        in_proj_weight_q_mu = torch.Tensor(embed_dim, embed_dim)
        in_proj_weight_q_rho = torch.Tensor(embed_dim, embed_dim)

        # KV Linear mapping weight
        in_proj_weight_kv_mu = torch.Tensor(2 * embed_dim, embed_dim)
        in_proj_weight_kv_rho = torch.Tensor(2 * embed_dim, embed_dim)

        # Output linear mapping weight
        out_proj_weight_mu = torch.Tensor(embed_dim, embed_dim)
        out_proj_weight_rho = torch.Tensor(embed_dim, embed_dim)

        self.mu, self.indices, self.shapes = \
            flatten_list([in_proj_weight_q_mu, in_proj_weight_kv_mu, out_proj_weight_mu])
        self.mu = Parameter(self.mu)
        self.rho, _, _ = flatten_list([in_proj_weight_q_rho, in_proj_weight_kv_rho, out_proj_weight_rho])
        self.rho = Parameter(self.rho)

        self.weight = Gaussian(self.mu, self.rho)
        self.weight_prior = ScaleMixtureGaussian()

        self.attn_func = encdec_attn_func

        self.reset_parameters()
        try:
            # the fast one requires apex and does not work with incremental so careful
            from apex.contrib.multihead_attn.fast_encdec_multihead_attn_func import fast_encdec_attn_func
            self.attn_func_fast = fast_encdec_attn_func
            self.optimized = 1

        except ModuleNotFoundError as e:
            self.optimized = 2
            self.attn_func_fast = None

    def reset_parameters(self):
        # We initialize μ with a Gaussian around 0
        # (just as we would initialize standard weights of a neural network)
        std_ = math.sqrt(2.0 / (self.embed_dim + self.embed_dim))
        nn.init.normal_(self.mu, 0.0, std_)

        # It is important to initialize ρ (and hence σ) to a small value, otherwise learning might not work properly.
        nn.init.normal_(self.rho, -7, 0.1)

    def forward(self, query, key, value, attn_mask=None, incremental=False, incremental_cache=None,
                sample=False, calculate_log_probs=False):

        calculate_log_probs = calculate_log_probs or self.training
        sample = sample or self.training
        assert value is key, "ERROR: Keys and values must be the same."

        is_training = self.training
        time_masking = False
        len_key = key.size(0)

        # (MCMC)
        # Sample the weights from the variational posterior distribution q(w)
        sampled_weights, log_variational_posterior = \
            self.weight.sample(stochastic=sample, return_log_prob=calculate_log_probs)

        in_proj_weight_q, in_proj_weight_kv, out_proj_weight = unflatten(sampled_weights, self.indices, self.shapes)

        # Perform forward with the sampled weights
        if self.optimized == 1 and (self.training and not incremental) and len_key <= 1024 and query.is_cuda:
            if attn_mask is not None:
                if attn_mask.dim() == 3:
                    attn_mask = attn_mask.squeeze(1)
                attn_mask = attn_mask.byte()

            outputs = self.attn_func_fast(time_masking, is_training, self.num_heads,
                                          query.type_as(in_proj_weight_q), key.type_as(in_proj_weight_q),
                                          in_proj_weight_q, in_proj_weight_kv, out_proj_weight,
                                          attn_mask, self.dropout)

            coverage = None

        # during evaluation we use the python binding which is safer ....
        else:
            outputs, coverage, = self.attn_func(time_masking, is_training,
                                                self.num_heads, query, key,
                                                in_proj_weight_q, in_proj_weight_kv,
                                                out_proj_weight, attn_mask, self.dropout,
                                                incremental, incremental_cache)

        if calculate_log_probs:
            # KL Divergence between prior and (variational) posterior
            self.log_variational_posterior = log_variational_posterior
            self.log_prior = self.weight_prior.log_prob(sampled_weights)

        return outputs, coverage
