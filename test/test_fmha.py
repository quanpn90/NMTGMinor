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
import sys
import torch
import numpy as np
import unittest
import math

import fmhalib as mha
from time import time
from random import randint
from torch.cuda.amp import custom_fwd, custom_bwd


# CONDITION to use fast mha:
# length <= 512 and sm=80


class IndexCopy(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, non_pad_indices, total_batch_size):

        sizes = list(input.size())
        sizes[0] = total_batch_size

        output = input.new_zeros(*sizes)
        output.index_copy_(0, non_pad_indices, input)
        ctx.save_for_backward(non_pad_indices)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grads):

        non_pad_indices,  = ctx.saved_tensors

        grad_input = output_grads.index_select(0, non_pad_indices)

        return grad_input, None, None


def py_mha(qkv, amask, b, s, h, d, high_precision=True):
    qkv = qkv.view(b, s, h, 3, d)
    q = qkv[:, :, :, 0, :].permute(0, 2, 1, 3)
    k = qkv[:, :, :, 1, :].permute(0, 2, 1, 3)
    v = qkv[:, :, :, 2, :].permute(0, 2, 1, 3)
    if high_precision:
        p = torch.matmul(q.float(), k.permute(0, 1, 3, 2).float())
        p_masked = p / math.sqrt(d) + (amask) * -10000.0
        s = torch.softmax(p_masked, -1).to(qkv.dtype)
        ctx = torch.matmul(s, v)
    else:
        p = torch.matmul(q, k.permute(0, 1, 3, 2))
        p_masked = p / math.sqrt(d) + (amask) * -10000.0
        s = torch.softmax(p_masked, -1).to(qkv.dtype)
        ctx = torch.matmul(s, v)

    ctx = ctx.permute(0, 2, 1, 3).contiguous()

    ctx.retain_grad()

    return ctx


class TestFMHA(unittest.TestCase):

    def run_uneven_test(self, s, b):

        s = randint(s-127, s)

        print(f'Test uneven s={s} b={b}')

        torch.manual_seed(12341234)
        torch.cuda.manual_seed(12341234)

        dtype = torch.float16
        device = torch.device('cuda')

        h = 16
        d = 64

        amask = torch.ones(b, s, dtype=dtype, device=device)

        slens = []
        prev_size = -1
        for b_ in range(b):
            if prev_size == -1:
                curr_size = randint(1, s)
                slens.append(curr_size)
                prev_size = curr_size
            else:
                # no sort?
                curr_size = randint(1, s)
                slens.append(curr_size)
                prev_size = curr_size

            amask[b_, :prev_size].fill_(0)  # the first prev_size elements have no mask

        max_s = max(slens)

        non_pad_indices = torch.nonzero(amask.view(-1).ne(1)).squeeze(1)

        a = torch.tensor(np.array([0] + slens), dtype=torch.int32)
        amask = amask.unsqueeze(1).unsqueeze(1)
        seqlens = torch.tensor(slens, dtype=torch.int32, device=device)
        cu_seqlens = torch.cumsum(a, 0).to(dtype=torch.int32, device=device)
        total = cu_seqlens[-1].item()

        # input for python mha?
        # should be identical layout with the current code
        qkv = torch.randn((b, s, h, 3, d), device=device, dtype=dtype)

        def run_fmha_forward(qkv_, non_pad_indices_, cu_seqlens_, max_s_):
            qkv_vs = qkv_.permute(0, 1, 3, 2, 4).contiguous().view(b * s, 3, h, d)
            qkv_vs = qkv_vs.index_select(0, non_pad_indices_)
            if b < 4:
                ctx, S_ = mha.fwd_nl(qkv_vs, cu_seqlens_, 0.0, max_s_, True, None)
            else:
                ctx, S_ = mha.fwd(qkv_vs, cu_seqlens_, 0.0, max_s_, True, None)

            ctx.requires_grad = True
            ctx_out = IndexCopy.apply(ctx, non_pad_indices, b * s)
            ctx_out = ctx_out.view(b, s, h, d)

            return qkv_vs, ctx, ctx_out, S_

        def run_mha_backward(grad, ctx_out_, ctx_, qkv_vs_, non_pad_indices_, cu_seqlens_, max_s_, S__):
            ctx_out.backward(grad, inputs=[ctx_])

            if b < 4:
                dqkv2, _, _ = mha.bwd_nl(ctx.grad, qkv_vs_, S__, cu_seqlens_, 0.0, max_s_)
            else:
                dqkv2, _ = mha.bwd(ctx.grad, qkv_vs_, S__, cu_seqlens_, 0.0, max_s_)

            dqkv2 = dqkv2.permute(0, 2, 1, 3)  # [b*s, 3, h, d]

            return dqkv2

        qkv_vs, ctx, ctx_out, S_ = run_fmha_forward(qkv, non_pad_indices, cu_seqlens, max_s)
        qkv.requires_grad = True

        ctx_ref = py_mha(qkv, amask, b, s, h, d)
        mask = amask.squeeze(1).squeeze(1).bool().unsqueeze(-1).unsqueeze(-1)
        ctx_ref.masked_fill_(mask, 0)

        self.assertTrue(torch.allclose(ctx_ref.float(), ctx_out.float(), atol=1e-2))
        print("output ok.")

        labels = torch.randn_like(ctx_ref)
        diff = ctx_ref - labels
        l = (diff * diff).sum() / b
        l.backward(inputs=[ctx_ref, qkv])

        dw = ctx_ref.grad  # .permute(0, 2, 1, 3)
        dw2 = dw.clone().detach().contiguous()

        dqkv2 = run_mha_backward(dw2, ctx_out, ctx, qkv_vs, non_pad_indices, cu_seqlens, max_s, S_)

        qkv_grad = qkv.grad.view(b * s, h, 3, d)
        qkv_grad = qkv_grad.index_select(0, non_pad_indices)

        if not torch.allclose(qkv_grad.float(), dqkv2.float(), atol=1e-3):
            print(qkv_grad.float() - dqkv2.float())
        self.assertTrue(torch.allclose(qkv_grad.float(), dqkv2.float(), atol=1e-2))

        num_iters = 20

        torch.cuda.synchronize()
        start_time = time()
        for _ in range(num_iters):
            qkv_vs, ctx, ctx_out, S_ = run_fmha_forward(qkv, non_pad_indices, cu_seqlens, max_s)

            dw2 = torch.randn_like(ctx_out)
            dqkv2 = run_mha_backward(dw2, ctx_out, ctx, qkv_vs, non_pad_indices, cu_seqlens, max_s, S_)

        torch.cuda.synchronize()
        stop_time = time()
        print(F"Fused MHA MLP time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
        torch.cuda.profiler.stop()

        torch.cuda.synchronize()
        start_time = time()
        for _ in range(num_iters):
            ctx_ref = py_mha(qkv, amask, b, s, h, d, high_precision=False)

            labels = torch.randn_like(ctx_ref)
            ctx_ref.backward(labels)

        torch.cuda.synchronize()
        stop_time = time()
        print(F"Python MLP time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
        torch.cuda.profiler.stop()

    def run_test(self, s, b):
        s = randint(s - 127, s)

        print(f'Test s={s} b={b}')

        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)

        dtype = torch.float16
        device = torch.device('cuda')

        h = 16
        d = 64

        slens = [s] * b
        a = torch.tensor(np.array([0] + slens), dtype=torch.int32)
        amask = torch.zeros(b, h, s, s, dtype=dtype, device=device)
        seqlens = torch.tensor(slens, dtype=torch.int32, device=device)
        cu_seqlens = torch.cumsum(a, 0).to(dtype=torch.int32, device=device)
        total = cu_seqlens[-1].item()

        # input for python mha?
        qkv = torch.randn((b, s, h, 3, d), device=device, dtype=dtype)

        # input for fmha
        qkv_vs = qkv.permute(0, 1, 3, 2, 4).contiguous().view(b * s, 3, h, d)

        qkv.requires_grad = True

        if b < 4:
            ctx, S_ = mha.fwd_nl(qkv_vs, cu_seqlens, 0.0, s, True, None)
        else:
            ctx, S_ = mha.fwd(qkv_vs, cu_seqlens, 0.0, s, True, None)
        ctx = ctx.view(b, s, h, d)

        ctx_ref = py_mha(qkv, amask, b, s, h, d)
        self.assertTrue(torch.allclose(ctx_ref.float(), ctx.float(), atol=1e-3))

        labels = torch.randn_like(ctx_ref)
        diff = ctx_ref - labels
        l = (diff * diff).sum() / b
        l.backward()

        dw = ctx_ref.grad.permute(0, 2, 1, 3)

        dw2 = dw.permute(0, 2, 1, 3).clone().detach().contiguous()

        if b < 4:
            dqkv2, _, _ = mha.bwd_nl(dw2, qkv_vs, S_, cu_seqlens, 0.0, s)
        else:
            dqkv2, _ = mha.bwd(dw2, qkv_vs, S_, cu_seqlens, 0.0, s)

        dqkv2 = dqkv2.permute(0, 2, 1, 3).view(b, s, h, 3, d)

        # print(qkv.grad.float() - dqkv2.float())
        self.assertTrue(torch.allclose(qkv.grad.float(), dqkv2.float(), atol=1e-2))

        num_iters = 20

        torch.cuda.synchronize()
        start_time = time()
        for _ in range(num_iters):
            if b < 4:
                ctx, S_ = mha.fwd_nl(qkv_vs, cu_seqlens, 0.0, s, True, None)
            else:
                ctx, S_ = mha.fwd(qkv_vs, cu_seqlens, 0.0, s, True, None)

            dw2 = torch.randn_like(ctx)
            if b < 4:
                dqkv2, _, _ = mha.bwd_nl(dw2, qkv_vs, S_, cu_seqlens, 0.0, s)
            else:
                dqkv2, _ = mha.bwd(dw2, qkv_vs, S_, cu_seqlens, 0.0, s)

        torch.cuda.synchronize()
        stop_time = time()
        print(F"Fused MHA MLP time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
        torch.cuda.profiler.stop()

        torch.cuda.synchronize()
        start_time = time()
        for _ in range(num_iters):
            ctx_ref = py_mha(qkv, amask, b, s, h, d, high_precision=False)

            labels = torch.randn_like(ctx_ref)
            ctx_ref.backward(labels)

        torch.cuda.synchronize()
        stop_time = time()
        print(F"Python MLP time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
        torch.cuda.profiler.stop()

    def test_128(self):
        # self.run_test(128, 55)
        # self.run_test(128, 47)
        # self.run_test(128, 90)
        #
        # self.run_uneven_test(128, 55)
        # self.run_uneven_test(128, 47)
        # self.run_uneven_test(128, 90)
        # self.run_test(128, 3)
        self.run_uneven_test(128, 3)

    def test_256(self):  # 129 - 256?
        #
        # self.run_test(256, 32)
        # self.run_test(256, 16)
        # self.run_test(224, 16)
        self.run_test(224, 3)
        #
        # self.run_uneven_test(256, 32)
        # self.run_uneven_test(256, 16)
        # self.run_uneven_test(224, 16)
        self.run_uneven_test(224, 3)

    def test_384(self):
        self.run_test(384, 32)
        self.run_test(384, 16)
        self.run_test(384, 8)
        #
        # self.run_uneven_test(384, 32)
        # self.run_uneven_test(384, 16)
        # self.run_uneven_test(384, 8)
        self.run_test(384, 3)

    def test_512(self):
        # self.run_test(512, 32)
        # self.run_test(512, 2)
        self.run_test(512, 3)
        #
        # self.run_uneven_test(512, 32)
        # self.run_uneven_test(512, 2)
        # self.run_uneven_test(512, 3)
#

if __name__ == '__main__':
    unittest.main()
