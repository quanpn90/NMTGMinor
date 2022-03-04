import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
import math
from typing import Dict, Optional, Tuple
import torch
from torch.cuda.amp import custom_fwd, custom_bwd

from onmt.modules.performer import Performer, ProjectionUpdater
from onmt.modules.optimized.self_attention_func import self_attn_func
from onmt.modules.optimized.linear import linear_function


class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    has_fused_layernorm = True


    class FusedLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)


except ImportError:
    has_fused_layernorm = False


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    # if torch.jit.is_scripting():
    #     export = True
    # if not export and torch.cuda.is_available() and has_fused_layernorm:
    #     return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class GumbelVectorQuantizer(nn.Module):
    def __init__(
            self,
            dim,
            num_vars,
            temp,
            groups,
            combine_groups,
            vq_dim,
            time_first,
            activation=nn.GELU(),
            weight_proj_depth=1,
            weight_proj_factor=1,
    ):
        """Vector quantization using gumbel softmax
        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            activation: what activation to use (should be a module). this is only used if weight_proj_depth is > 1
            weight_proj_depth: number of layers (with activation in between) to project input before computing logits
            weight_proj_factor: this is used only if weight_proj_depth is > 1. scales the inner dimensionality of
                                projections by this factor
        """
        super().__init__()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.time_first = time_first

        assert (
                vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        self.vars = nn.Parameter(torch.FloatTensor(1, num_groups * num_vars, var_dim))
        nn.init.uniform_(self.vars)

        if weight_proj_depth > 1:

            def block(input_dim, output_dim):
                return nn.Sequential(nn.Linear(input_dim, output_dim), activation)

            inner_dim = self.input_dim * weight_proj_factor
            self.weight_proj = nn.Sequential(
                *[
                    block(self.input_dim if i == 0 else inner_dim, inner_dim)
                    for i in range(weight_proj_depth - 1)
                ],
                nn.Linear(inner_dim, groups * num_vars),
            )
        else:
            self.weight_proj = nn.Linear(self.input_dim, groups * num_vars)
            nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(self.weight_proj.bias)

        if isinstance(temp, str):
            import ast
            temp = ast.literal_eval(temp)
        assert len(temp) == 3, f"{temp}, {len(temp)}"

        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp
        self.codebook_indices = None

    def set_num_updates(self, num_updates):
        self.curr_temp = max(
            self.max_temp * self.temp_decay ** num_updates, self.min_temp
        )

    def get_codebook_indices(self):
        if self.codebook_indices is None:
            from itertools import product

            p = [range(self.num_vars)] * self.groups
            inds = list(product(*p))
            self.codebook_indices = torch.tensor(
                inds, dtype=torch.long, device=self.vars.device
            ).flatten()

            if not self.combine_groups:
                self.codebook_indices = self.codebook_indices.view(
                    self.num_vars ** self.groups, -1
                )
                for b in range(1, self.groups):
                    self.codebook_indices[:, b] += self.num_vars * b
                self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    def codebook(self):
        indices = self.get_codebook_indices()
        return (
            self.vars.squeeze(0)
                .index_select(0, indices)
                .view(self.num_vars ** self.groups, -1)
        )

    def sample_from_codebook(self, b, n):
        indices = self.get_codebook_indices()
        indices = indices.view(-1, self.groups)
        cb_size = indices.size(0)
        assert (
                n < cb_size
        ), f"sample size {n} is greater than size of codebook {cb_size}"
        sample_idx = torch.randint(low=0, high=cb_size, size=(b * n,))
        indices = indices[sample_idx]

        z = self.vars.squeeze(0).index_select(0, indices.flatten()).view(b, n, -1)
        return z

    def to_codebook_index(self, indices):
        res = indices.new_full(indices.shape[:-1], 0)
        for i in range(self.groups):
            exponent = self.groups - i - 1
            res += indices[..., i] * (self.num_vars ** exponent)
        return res

    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res["x"], res["targets"]

    def forward(self, x, produce_targets=False):

        result = {"num_vars": self.num_vars * self.groups}

        # B x H x T -> B x T x H if not time first
        if not self.time_first:
            x = x.transpose(1, 2)

        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)

        # from fsz -> group * num_vars
        x = self.weight_proj(x)
        x = x.view(bsz * tsz * self.groups, -1)

        # choose the (indices of) max var in num_vars
        _, k = x.max(-1)

        # hard_x has the original size of x
        # 1 for chosen value, 0 for non-chosen value
        hard_x = (
            x.new_zeros(*x.shape)
                .scatter_(-1, k.view(-1, 1), 1.0)
                .view(bsz * tsz, self.groups, -1)
        )

        # mean over the bsz * tsz dimension?
        hard_probs = torch.mean(hard_x.float(), dim=0)
        result["code_perplexity"] = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        # code probabilities for each group
        avg_probs = torch.softmax(
            x.view(bsz * tsz, self.groups, -1).float(), dim=-1
        ).mean(dim=0)
        result["prob_perplexity"] = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        result["temp"] = self.curr_temp

        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(x)
        else:
            x = hard_x

        x = x.view(bsz * tsz, -1)

        vars = self.vars
        if self.combine_groups:
            vars = vars.repeat(1, self.groups, 1)

        if produce_targets:
            result["targets"] = (
                x.view(bsz * tsz * self.groups, -1)
                    .argmax(dim=-1)
                    .view(bsz, tsz, self.groups)
                    .detach()
            )

        # x size: [bsz * tsz * self.groups, self.num_vars]
        # the last dimension is basically distribution over different vars (for each group)
        x = x.unsqueeze(-1) * vars

        # vars is "probably" latent variable embeddings
        x = x.view(bsz * tsz, self.groups, self.num_vars, -1)
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)

        if not self.time_first:
            x = x.transpose(1, 2)  # BTC -> BCT

        result["x"] = x

        return result


class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            dropout=0.0,
            weight_drop=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            self_attention=False,
            encoder_decoder_attention=False,
            favor=False,
            generalized_attention=False,
            nb_features=256,
            **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_p = dropout
        self.weight_drop = weight_drop

        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()
        self.favor = favor
        if self.favor:
            self.performer = Performer(self.head_dim, nb_features, generalized_attention=generalized_attention)
        else:
            self.performer = None

        self.onnx_trace = False
        self.fast_attention = False
        self.is_factorized = False
        self.multiplicative_factorize = False
        self.fast_factorize = False

        from onmt.modules.optimized.fast_mha import fast_bert_mha
        self.fast_bert_mha = fast_bert_mha

    def fix_projection_matrices_(self):
        if self.proj_updater:
            self.proj_updater.feature_redraw_interval = None

    def add_factorized_weights(self, n_languages, rank=4, multiplicative=False, fast=False,
                               sub_factors=0, sub_factor_rank=-1):
        embed_dim = self.embed_dim
        self.is_factorized = True
        self.multiplicative_factorize = multiplicative
        self.fast_factorize = fast
        self.sub_factorized = (sub_factors > 0)

        self.r_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, 3 * embed_dim))
        self.s_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
        self.r_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
        self.s_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))

        std = 0.01 if fast else 0.02
        nn.init.normal_(self.r_i, 0.0, std)
        nn.init.normal_(self.s_i, 0.0, std)
        nn.init.normal_(self.r_o, 0.0, std)
        nn.init.normal_(self.s_o, 0.0, std)

        if multiplicative:
            rank = rank if fast else 1
            self.rm_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, 3 * embed_dim))
            self.sm_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.rm_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.sm_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))

            constant = math.sqrt(1.0 / rank) if fast else 1
            nn.init.constant_(self.rm_i, constant)
            nn.init.constant_(self.sm_i, constant)
            nn.init.constant_(self.rm_o, constant)
            nn.init.constant_(self.sm_o, constant)

        if self.sub_factorized:
            self.sub_r_i = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, 3 * embed_dim))
            self.sub_s_i = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, embed_dim))
            self.sub_r_o = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, embed_dim))
            self.sub_s_o = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, embed_dim))

            std = 0.01 if fast else 0.02
            nn.init.normal_(self.sub_r_i, 0.0, std)
            nn.init.normal_(self.sub_s_i, 0.0, std)
            nn.init.normal_(self.sub_r_o, 0.0, std)
            nn.init.normal_(self.sub_s_o, 0.0, std)

            if multiplicative:
                sub_factor_rank = sub_factor_rank if fast else 1
                self.sub_rm_i = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, 3 * embed_dim))
                self.sub_sm_i = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, embed_dim))
                self.sub_rm_o = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, embed_dim))
                self.sub_sm_o = torch.nn.Parameter(torch.Tensor(sub_factors, sub_factor_rank, embed_dim))

                constant = math.sqrt(1.0 / sub_factor_rank) if fast else 1
                nn.init.constant_(self.sub_rm_i, constant)
                nn.init.constant_(self.sub_sm_i, constant)
                nn.init.constant_(self.sub_rm_o, constant)
                nn.init.constant_(self.sub_sm_o, constant)

    def convert_fast_attention(self):

        # print("Convert from vanilla to fast attention module ...")
        if self.fast_attention:
            return
        self.fast_attention = True
        assert self.qkv_same_dim, "Only works with QKV same dim."
        w_q = self.q_proj.weight.clone()
        w_k = self.k_proj.weight.clone()
        w_v = self.v_proj.weight.clone()
        weights = [w_q, w_k, w_v]
        weight_ = torch.cat(weights, dim=0).contiguous()

        b_q = self.q_proj.bias.clone()
        b_k = self.k_proj.bias.clone()
        b_v = self.v_proj.bias.clone()
        biases = [b_q, b_k, b_v]
        bias_ = torch.cat(biases, dim=0).contiguous()

        head_dim = self.head_dim
        heads = self.num_heads
        input_dim = self.embed_dim

        # when we concatenate the weights, the output has the size 3 * D (3 -> heads -> head_dim)
        # the fast attention module requires (heads -> 3 -> head_dim)
        weight_ = weight_.reshape(3 * head_dim * heads, input_dim).view(3, heads, head_dim, input_dim).transpose(0, 1). \
            reshape(-1, input_dim)

        bias_ = bias_.reshape(3 * head_dim * heads).view(3, heads, head_dim).transpose(0, 1).reshape(-1)

        weight_t = torch.Tensor(3 * input_dim, input_dim)
        bias_t = torch.Tensor(3 * input_dim)
        weight_t.copy_(weight_)
        bias_t.copy_(bias_)
        self.proj_weight = Parameter(weight_t)
        self.proj_bias = Parameter(bias_t)

        self.proj_weight.requires_grad = self.q_proj.weight.requires_grad
        self.proj_bias.requires_grad = self.q_proj.bias.requires_grad
        del self.q_proj, self.k_proj, self.v_proj

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            attn_mask: Optional[Tensor] = None,
            cu_seqlens=None, max_len=None,
            lang=None, atb=None, **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param query:
        :param key:
        :param value:
        :param key_padding_mask:
        :param incremental_state:
        :param attn_mask:
        :param cu_seqlens:
        :param max_len:
        :param lang:
        :param atb:
        :param kwargs:
        :return:
        """

        is_tpu = query.device.type == "xla"

        if not self.favor:

            if not self.fast_attention:
                tgt_len, bsz, embed_dim = query.size()
                src_len = tgt_len
                assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
                assert list(query.size()) == [tgt_len, bsz, embed_dim]
                need_weight = False
                assert key is not None and value is not None

                return F.multi_head_attention_forward(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    torch.empty([0]),
                    torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                    self.bias_k,
                    self.bias_v,
                    self.add_zero_attn,
                    self.dropout_p,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    self.training,
                    key_padding_mask,
                    need_weight,
                    attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj.weight,
                    k_proj_weight=self.k_proj.weight,
                    v_proj_weight=self.v_proj.weight,
                )
            else:

                in_proj_weight = F.dropout(self.proj_weight, self.weight_drop, training=self.training)
                out_proj_weight = F.dropout(self.out_proj.weight, self.weight_drop, training=self.training)

                if self.is_factorized:
                    if self.multiplicative_factorize:
                        # squeeze possible because only 1
                        rm_i = torch.index_select(self.rm_i, 0, lang).squeeze(0)
                        sm_i = torch.index_select(self.sm_i, 0, lang).squeeze(0)
                        rm_o = torch.index_select(self.rm_o, 0, lang).squeeze(0)
                        sm_o = torch.index_select(self.sm_o, 0, lang).squeeze(0)

                        if self.fast_factorize:
                            mul_factor_in = torch.mm(rm_i.t(), sm_i)
                            mul_factor_out = torch.mm(rm_o.t(), sm_o)
                        else:
                            mul_factor_in = torch.bmm(rm_i.unsqueeze(-1), sm_i.unsqueeze(1)).sum(dim=0)
                            mul_factor_out = torch.bmm(rm_o.unsqueeze(-1), sm_o.unsqueeze(1)).sum(dim=0)

                        if self.sub_factorized and atb is not None:
                            # squeeze possible because only 1
                            rm_i = torch.index_select(self.sub_rm_i, 0, atb).squeeze(0)
                            sm_i = torch.index_select(self.sub_sm_i, 0, atb).squeeze(0)
                            rm_o = torch.index_select(self.sub_rm_o, 0, atb).squeeze(0)
                            sm_o = torch.index_select(self.sub_sm_o, 0, atb).squeeze(0)

                            if self.fast_factorize:
                                sub_mul_factor_in = torch.mm(rm_i.t(), sm_i)
                                sub_mul_factor_out = torch.mm(rm_o.t(), sm_o)
                            else:
                                sub_mul_factor_in = torch.bmm(rm_i.unsqueeze(-1), sm_i.unsqueeze(1)).sum(dim=0)
                                sub_mul_factor_out = torch.bmm(rm_o.unsqueeze(-1), sm_o.unsqueeze(1)).sum(dim=0)

                            mul_factor_in.mul_(sub_mul_factor_in)
                            mul_factor_out.mul_(sub_mul_factor_out)

                        in_proj_weight = in_proj_weight * mul_factor_in
                        out_proj_weight = out_proj_weight * mul_factor_out

                    r_i = torch.index_select(self.r_i, 0, lang).squeeze(0)
                    s_i = torch.index_select(self.s_i, 0, lang).squeeze(0)
                    r_o = torch.index_select(self.r_o, 0, lang).squeeze(0)
                    s_o = torch.index_select(self.s_o, 0, lang).squeeze(0)

                    if self.fast_factorize:
                        add_factor_in = torch.mm(r_i.t(), s_i)
                        add_factor_out = torch.mm(r_o.t(), s_o)
                    else:
                        add_factor_in = torch.bmm(r_i.unsqueeze(-1), s_i.unsqueeze(1)).sum(dim=0)
                        add_factor_out = torch.bmm(r_o.unsqueeze(-1), s_o.unsqueeze(1)).sum(dim=0)

                    if self.sub_factorized and atb is not None:

                        r_i = torch.index_select(self.sub_r_i, 0, atb).squeeze(0)
                        s_i = torch.index_select(self.sub_s_i, 0, atb).squeeze(0)
                        r_o = torch.index_select(self.sub_r_o, 0, atb).squeeze(0)
                        s_o = torch.index_select(self.sub_s_o, 0, atb).squeeze(0)

                        if self.fast_factorize:
                            sub_add_factor_in = torch.mm(r_i.t(), s_i)
                            sub_add_factor_out = torch.mm(r_o.t(), s_o)
                        else:
                            sub_add_factor_in = torch.bmm(r_i.unsqueeze(-1), s_i.unsqueeze(1)).sum(dim=0)
                            sub_add_factor_out = torch.bmm(r_o.unsqueeze(-1), s_o.unsqueeze(1)).sum(dim=0)

                        add_factor_in.add_(sub_add_factor_in)
                        add_factor_out.add_(sub_add_factor_out)

                    in_proj_weight = in_proj_weight + add_factor_in
                    out_proj_weight = out_proj_weight + add_factor_out

                # Forward Pass starts here
                if query.ndim == 3:
                    # Call semi-fast attention from CUDA/
                    tgt_len, bsz, embed_dim = query.size()
                    src_len = tgt_len
                    assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
                    assert list(query.size()) == [tgt_len, bsz, embed_dim]

                    inputs = query

                    is_training = self.training
                    low_precision = True

                    outputs, coverage = self_attn_func(False, is_training, self.num_heads, inputs,
                                                       in_proj_weight, out_proj_weight,
                                                       self.proj_bias, self.out_proj.bias,
                                                       key_padding_mask, self.dropout_p,
                                                       False, None,
                                                       False, None,  # incremental and state
                                                       low_precision, True)  # low-precision and return coverage

                    return outputs, coverage

                # Fused attention using packed data (B T H) -> (BxT H) and removing padded positions
                elif query.ndim == 2:
                    assert self.fast_bert_mha is not None
                    assert query.dtype == torch.half
                    assert cu_seqlens is not None
                    assert max_len is not None  # and max_len <= 512
                    sm = torch.cuda.get_device_capability()

                    # Only Ampere supported at the moment
                    assert (sm[0] == 8 and (sm[1] in [0]) and max_len <= 512)

                    total_bsz = query.size(0)
                    qkv = linear_function(query, in_proj_weight, self.proj_bias)  # B x H
                    # B x 3 x H x d

                    # TODO: moving to CUDA to remove overhead?
                    qkv = qkv.view(total_bsz, self.num_heads, 3, self.head_dim).transpose(1, 2).contiguous()

                    context, coverage = self.fast_bert_mha(qkv, cu_seqlens, self.dropout_p, max_len, self.training)

                    context = context.view(-1, self.num_heads * self.head_dim).contiguous()
                    outputs = linear_function(context, out_proj_weight, self.out_proj.bias)

                    return outputs, coverage

        else:
            # using performer attention
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

            bsz, len_q, hidden = q.size(0), q.size(1), q.size(2)
            h, d = self.num_heads, self.head_dim
            len_k, len_v = k.size(1), v.size(1)

            q = q.view(bsz, len_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(bsz * h, len_q, d)
            k = k.view(bsz, len_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(bsz * h, len_k, d)
            v = v.view(bsz, len_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # .reshape(bsz * h, len_v, d)

            # 1 for padded positions, 0 for non-padded positions
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask[:, None, :, None]
                v.masked_fill_(key_padding_mask, 0)

            v = v.reshape(bsz * h, len_v, d)

            out, attn = self.performer(q, k, v)
            # out = out.transpose(1, 2).view(bsz, out.size(-2), -1)
            out = out.reshape(bsz, h, len_q, -1).permute(0, 2, 1, 3).reshape(bsz, len_v, -1)

            out = self.out_proj(out)

            return out, attn


def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
            0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )


def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)


class IndexCopy(torch.autograd.Function):
    """
    This function is kinda similar to rnn pad_packed_sequence
    It remaps nonpadded values for a (N-1)-d tensor into a (N)-d tensor
    """
    @staticmethod
    @custom_fwd
    def forward(ctx, input, non_pad_indices, total_batch_size):
        """
        :param ctx:
        :param input: 2D [bsz x ... ] bsz is the total number of elements after unpadding
        :param non_pad_indices: bsz * seq_len
        :param total_batch_size: (int) bsz * seq_len (before unpadding) > bsz
        :return:
        In the forward pass we create a new zero tensor and copy the inputs into it based on non_pad_indices
        """
        sizes = list(input.size())
        sizes[0] = total_batch_size

        output = input.new_zeros(*sizes)
        output.index_copy_(0, non_pad_indices, input)
        ctx.save_for_backward(non_pad_indices)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grads):
        """
        :param ctx:
        :param output_grads:
        :return:
        In the backward pass we simply
        """
        non_pad_indices, = ctx.saved_tensors

        grad_input = output_grads.index_select(0, non_pad_indices)

        return grad_input, None, None


index_copy = IndexCopy.apply

#
# class ConvDownsampler(nn.Module):
#     def __init__(
#             self,
#             conv_layers: List[Tuple[int, int, int]],  # n_in, n_out, kernel size?
#             dropout: float = 0.0,
#             mode: str = "default",
#             conv_bias: bool = False,
#     ):
#         super().__init__()
#
#         assert mode in {"default", "layer_norm"}
#
#         def block(
#                 n_in,
#                 n_out,
#                 k,
#                 stride,
#                 is_layer_norm=False,
#                 is_group_norm=False,
#                 conv_bias=False,
#         ):
#             def make_conv():
#                 conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
#                 nn.init.kaiming_normal_(conv.weight)
#                 return conv
#
#             assert (
#                            is_layer_norm and is_group_norm
#                    ) == False, "layer norm and group norm are exclusive"
#
#             if is_layer_norm:
#                 return nn.Sequential(
#                     make_conv(),
#                     nn.Dropout(p=dropout),
#                     nn.Sequential(
#                         TransposeLast(),
#                         LayerNorm(dim),
#                         TransposeLast(),
#                     ),
#                     nn.GELU(),
#                 )
#             elif is_group_norm:
#                 return nn.Sequential(
#                     make_conv(),
#                     nn.Dropout(p=dropout),
#                     Fp32GroupNorm(dim, dim, affine=True),
#                     nn.GELU(),
#                 )
#             else:
#                 return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())
#
#         in_d = 1
#         self.conv_layers = nn.ModuleList()
#         for i, cl in enumerate(conv_layers):
#             assert len(cl) == 3, "invalid conv definition: " + str(cl)
#             (dim, k, stride) = cl
#
#             self.conv_layers.append(
#                 block(
#                     in_d,
#                     dim,
#                     k,
#                     stride,
#                     is_layer_norm=mode == "layer_norm",
#                     is_group_norm=mode == "default" and i == 0,
#                     conv_bias=conv_bias,
#                 )
#             )
#             in_d = dim
#
#     def forward(self, x):
#
#         # BxT -> BxCxT (only for waveforms with 1 channel)
#         x = x.unsqueeze(1)
#
#         for conv in self.conv_layers:
#             x = conv(x)
#
#         return x