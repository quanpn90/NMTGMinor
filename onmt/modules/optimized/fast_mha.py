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
    import fmhalib
except (ModuleNotFoundError, ImportError) as e:
    fmhalib = None


class FMHAFun(torch.autograd.Function):
    """
    BERT Style Multihead Self Attention (Encoder only)
    Can be used for wav2vec 2.0
    """

    @staticmethod
    def forward(ctx, qkv, cu_seqlens, p_dropout, max_s, is_training):
        batch_size = cu_seqlens.numel() - 1
        if batch_size < 4:
            context, S_dmask = fmhalib.fwd_nl(qkv, cu_seqlens, p_dropout, max_s, is_training, None)
        else:
            context, S_dmask = fmhalib.fwd(qkv, cu_seqlens, p_dropout, max_s, is_training, None)
        ctx.save_for_backward(qkv, S_dmask)
        ctx.cu_seqlens = cu_seqlens
        ctx.p_dropout = p_dropout
        ctx.max_s = max_s
        return context, S_dmask

    @staticmethod
    def backward(ctx, dout, dsoftmax):
        qkv, S_dmask = ctx.saved_tensors
        batch_size = ctx.cu_seqlens.numel() - 1
        if batch_size < 4:
            dqkv, dp, _ = fmhalib.bwd_nl(dout, qkv, S_dmask, ctx.cu_seqlens, ctx.p_dropout, ctx.max_s)
        else:
            dqkv, dp = fmhalib.bwd(dout, qkv, S_dmask, ctx.cu_seqlens, ctx.p_dropout, ctx.max_s)

        return dqkv, None, None, None, None, None, None


def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        try:
            return torch.cuda.amp.autocast_mode._cast(args, torch.get_autocast_gpu_dtype())
        except AttributeError:
            return torch.cuda.amp.autocast_mode._cast(args, torch.half)


if fmhalib is not None:
    def fast_bert_mha(*args):
        args = _cast_if_autocast_enabled(*args)
        with torch.cuda.amp.autocast(enabled=False):
            return FMHAFun.apply(*args)
else:
    fast_bert_mha = None
