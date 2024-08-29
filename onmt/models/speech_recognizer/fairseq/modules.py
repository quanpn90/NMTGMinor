import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
import math
from typing import Dict, Optional, Tuple
import torch
from torch.cuda.amp import custom_fwd, custom_bwd

# from onmt.modules.optimized.self_attention_func import self_attn_func, self_attn_compact_func
# from onmt.modules.optimized.relative_self_attention_func import relative_self_attn_func
# from onmt.modules.optimized.linear import linear_function, factorize_linear

from ..fairseq_wav2vec2.fairseq_modules import Fp32LayerNorm, Fp32GroupNorm, \
    LayerNorm, GumbelVectorQuantizer, SamePad, TransposeLast, GradMultiply

has_fused_layernorm = False


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
        self.rotary_position = False
        self.pos_proj_weight = None
        self.relative = False

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

        self.onnx_trace = False
        self.fast_attention = False
        self.is_factorized = False
        self.multiplicative_factorize = False
        self.fast_factorize = False
        self.flex_factorize = False

    def add_factorized_weights(self, n_languages, rank=4, multiplicative=False,
                               fast=False, dyrank=False, flexible=False, **kwargs):

        raise NotImplementedError

    def convert_fast_attention(self):

        return

    def add_relative_attention(self):

        return

    def add_rotary_attention(self):
        return

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
            positions: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
            cu_seqlens=None, max_len=None,
            lang=None, atb=None,
            checkpointing=False, **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param checkpointing:
        :param positions:
        :param query:
        :param key:
        :param value:
        :param key_padding_mask:
        :param attn_mask:
        :param cu_seqlens:
        :param max_len:
        :param lang:
        :param atb:
        :param kwargs:
        :return:
        """

        is_tpu = query.device.type == "xla"
        checkpointing = False  # temporarily not checkpoint atm

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        need_weight = False  # note: this argument will make things very slow
        assert key is not None and value is not None
        assert self.relative is False

        # pytorch automatically handles memory efficient and flash attention

        # key_padding_mask has size [B x T] in which masked positions are 1 and non masked are 0
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.bool()

        return F.multi_head_attention_forward(
            query,  # [T x B x H]
            key,    # [T x B x H]
            value,  # [T x B x H]
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


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in torch < 1.8.0


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


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


