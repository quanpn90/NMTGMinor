###############################################################################
# Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################


import torch
import torch.nn.functional as F

try:
    import flash_attn_cuda

except (ModuleNotFoundError, ImportError) as e:
    flash_attn_cuda = None

from .linear import linear_blaslt


def _flash_attn_forward(qkv, cu_seqlens, dropout_p, max_s, softmax_scale, causal, return_softmax):
    context, softmax_lse, *rest = flash_attn_cuda.fwd(qkv, cu_seqlens, dropout_p, max_s, softmax_scale,
                                                       False, causal, return_softmax, None)
    # if context.isnan().any() or softmax_lse.isnan().any():
    #     breakpoint()
    S_dmask = rest[0] if return_softmax else None
    return context, softmax_lse, S_dmask


def _flash_attn_backward(dout, qkv, out, S_dmask, softmax_lse, cu_seqlens, dropout_p, max_s,
                   softmax_scale, causal):
    dqkv, dp, softmax_d = flash_attn_cuda.bwd(dout, qkv, out, S_dmask, softmax_lse, cu_seqlens, dropout_p,
                                               softmax_scale, max_s, False, causal, None)
    # if dqkv.isnan().any() or softmax_d.isnan().any():
    #     breakpoint()
    return dqkv


class FlashMHAFun(torch.autograd.Function):
    """
    BERT Style Multihead Self Attention (Encoder only)
    Can be used for wav2vec 2.0
    """

    @staticmethod
    def forward(ctx, qkv, cu_seqlens, dropout_p, max_s, softmax_scale, causal):
    # def forward(ctx, qkv, cu_seqlens, p_dropout, max_s, is_training):
        # Save rng_state because the backward pass will regenerate the dropout mask
        rng_state = torch.cuda.get_rng_state() if dropout_p > 0 else None

        # by default scale is 1/sqrt(head_dim)
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        context, softmax_lse, S_dmask = _flash_attn_forward(
            qkv, cu_seqlens, dropout_p, max_s, softmax_scale, causal=causal, return_softmax=False
        )
        ctx.save_for_backward(qkv, context, S_dmask, softmax_lse, cu_seqlens, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_s = max_s
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return context

    @staticmethod
    def backward(ctx, dout):
        qkv, context, S_dmask, softmax_lse, cu_seqlens, rng_state = ctx.saved_tensors

        # restore rng state to recompute dropout
        if rng_state is not None:
            cur_rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng_state)
        # S_dmask is None, temporarily use another tensor just to get it running
        dqkv = _flash_attn_backward(
            dout, qkv, context, context, softmax_lse, cu_seqlens, ctx.dropout_p,
            ctx.max_s, ctx.softmax_scale, ctx.causal
        )
        if rng_state is not None:
            torch.cuda.set_rng_state(cur_rng_state)
        return dqkv, None, None, None, None, None, None



def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        try:
            return torch.cuda.amp.autocast_mode._cast(args, torch.get_autocast_gpu_dtype())
        except AttributeError:
            return torch.cuda.amp.autocast_mode._cast(args, torch.half)


if flash_attn_cuda is not None:
    def flash_bert_mha(*args):
        args = _cast_if_autocast_enabled(*args)
        with torch.cuda.amp.autocast(enabled=False):
            return FlashMHAFun.apply(*args)

else:
    flash_bert_mha = None


