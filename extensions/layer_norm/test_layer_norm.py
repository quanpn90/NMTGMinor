from copy import copy, deepcopy
import math
import torch
from torch import nn
import torch.nn.functional as F
import unittest
from time import time
import numpy as np
import random

import fast_layer_norm_cuda



def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        try:
            return torch.cuda.amp.autocast_mode._cast(args, torch.get_autocast_gpu_dtype())
        except AttributeError:
            return torch.cuda.amp.autocast_mode._cast(args, torch.half)


class FastLayerNormFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, epsilon, memory_efficient=False):
        ctx.x_shape = x.shape
        ctx.memory_efficient = memory_efficient

        x = x.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()
        hidden_size = gamma.numel()

        xmat = x.view((-1, hidden_size))
        ymat, mu, rsigma = fast_layer_norm_cuda.ln_fwd(xmat, gamma, beta, epsilon)
        if ctx.memory_efficient:
            ctx.save_for_backward(ymat, gamma, None, rsigma, beta)
        else:
            ctx.save_for_backward(xmat, gamma, mu, rsigma, None)

        return ymat.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()  # this happens!
        x_or_y_mat, gamma, mu, rsigma, beta = ctx.saved_tensors
        dymat = dy.view(x_or_y_mat.shape)
        dxmat, dgamma, dbeta, _, _ = fast_layer_norm_cuda.ln_bwd(dymat, x_or_y_mat, mu, rsigma, gamma, beta,
                                                            ctx.memory_efficient)
        dx = dxmat.view(ctx.x_shape)
        return dx, dgamma, dbeta, None, None


def _fast_layer_norm(x, weight, bias, epsilon, memory_efficient=False):
    args = _cast_if_autocast_enabled(x, weight, bias, epsilon, memory_efficient)
    with torch.cuda.amp.autocast(enabled=False):
        return FastLayerNormFN.apply(*args)


class FastLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5, memory_efficient=False):
        super().__init__()
        self.epsilon = eps
        self.weight = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.bias = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

        self.memory_efficient = memory_efficient

    def reset_parameters(self):
        from torch.nn import init
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x):
        return _fast_layer_norm(x, self.weight, self.bias, self.epsilon, self.memory_efficient)

    def extra_repr(self):
        # TODO add dropout probability
        s = F"Fast Layer Norm w/ Hidden sizes: {self.weight.size(0)} / Memory Efficient: {self.memory_efficient}"
        return s

if __name__ == '__main__':
    class TestLN(unittest.TestCase):

        # def test_creation(self):
        #     MLP(mlp_sizes)

        def test_numeric(self):
            # print("Test numeric 3D ....")
            # for dropout in [0.0, 0.2, 0.5, 0.7]:
            bsz = 128
            seq_len = 512
            hidden_sizes = [
                768,
                1024,
                1280,
                1536,
                2048,
                4096
                # 1536,
                # 2048,
                # 2304,
                # 3072,
                # 3840,
                # 4096,
                # 5120,
                # 6144,
                # 8192,
                # 10240,
                # 12288,
                # 12800,
                # 15360,
                # 16384,
                # 18432,
                # 20480,
                # 24576,
                # 25600,
                # 30720,
                # 32768,
                # 40960,
                # 49152,
                # 65536,
            ]
            for hidden in  hidden_sizes:

                ref_ln = nn.LayerNorm(hidden).cuda()
                fast_ln = FastLayerNorm(hidden).cuda()
                print(fast_ln, ref_ln)

                test_input = torch.empty(seq_len, bsz, hidden, device="cuda").uniform_(-1., 1.).requires_grad_()
                ref_input = test_input.clone().detach().requires_grad_()

                ref_out = ref_ln(ref_input)

                test_out = fast_ln(test_input)

                np.testing.assert_allclose(
                    ref_out.detach().cpu().numpy(),
                    test_out.detach().cpu().numpy(),
                    atol=1e-5, rtol=1e-4)

                test_out.mean().mul(10.).backward()
                ref_out.mean().mul(10.).backward()
                np.testing.assert_allclose(
                    test_input.grad.detach().cpu().numpy(),
                    ref_input.grad.detach().cpu().numpy(),
                    atol=1e-7, rtol=1e-5)
                np.testing.assert_allclose(
                    fast_ln.weight.grad.detach().cpu().numpy(),
                    ref_ln.weight.grad.detach().cpu().numpy(),
                    atol=1e-7, rtol=1e-5)
                np.testing.assert_allclose(
                    fast_ln.bias.grad.detach().cpu().numpy(),
                    ref_ln.bias.grad.detach().cpu().numpy(),
                    atol=1e-7, rtol=1e-5)

        def test_performance_half(self):
            num_iters = 64
            bsz = 128
            seq_len = 512
            print("Testing performance ...")
            hidden_sizes = [
                768,
                1024,
                1280,
                1536,
                2048,
                2304,
                3072,
                3840,
                4096,
                5120,
                6144,
                8192,
                10240,
                12288,
                12800,
                15360,
                # 16384,
                # 18432,
                # 20480,
                # 24576,
                # 25600,
                # 30720,
                # 32768,
                # 40960,
                # 49152,
                # 65536,
            ]
            for hidden in hidden_sizes:
                ref_ln = nn.LayerNorm(hidden).cuda().half()
                fast_ln = FastLayerNorm(hidden).cuda().half()
                print(fast_ln, ref_ln)

                test_input = torch.empty(seq_len, bsz, hidden, device="cuda", dtype=torch.half).uniform_(-1., 1.).requires_grad_()
                ref_input = test_input.clone().detach().requires_grad_()

                # Warm up GPU
                for _ in range(num_iters):
                    ref_out = ref_ln(ref_input)
                    ref_loss = ref_out.mean()
                    ref_ln.zero_grad()
                    ref_loss.backward()
                    test_out = fast_ln(test_input)
                    test_loss = test_out.mean()
                    fast_ln.zero_grad()
                    test_loss.backward()

                torch.cuda.profiler.start()
                torch.cuda.synchronize()
                start_time = time()
                for _ in range(num_iters):
                    ref_out = ref_ln(ref_input)
                    ref_loss = ref_out.mean()
                    ref_ln.zero_grad()
                    ref_loss.backward()
                torch.cuda.synchronize()
                stop_time = time()
                print(F"\nPytorch Layer Norm time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")

                torch.cuda.synchronize()
                start_time = time()
                for _ in range(num_iters):
                    test_out = fast_ln(test_input)
                    test_loss = test_out.mean()
                    fast_ln.zero_grad()
                    test_loss.backward()
                torch.cuda.synchronize()
                stop_time = time()
                print(F"Custom Layer Norm time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
                torch.cuda.profiler.stop()

        def test_numeric_memory_efficient(self):
            # print("Test numeric 3D ....")
            # for dropout in [0.0, 0.2, 0.5, 0.7]:
            bsz = 128
            seq_len = 512
            hidden_sizes = [
                768,
                1024,
                1280,
                1536,
                2048,
                4096
                # 1536,
                # 2048,
                # 2304,
                # 3072,
                # 3840,
                # 4096,
                # 5120,
                # 6144,
                # 8192,
                # 10240,
                # 12288,
                # 12800,
                # 15360,
                # 16384,
                # 18432,
                # 20480,
                # 24576,
                # 25600,
                # 30720,
                # 32768,
                # 40960,
                # 49152,
                # 65536,
            ]
            for hidden in  hidden_sizes:

                ref_ln = nn.LayerNorm(hidden).cuda()
                fast_ln = FastLayerNorm(hidden, memory_efficient=True).cuda()
                print(fast_ln, ref_ln)

                test_input = torch.empty(seq_len, bsz, hidden, device="cuda").uniform_(-1., 1.).requires_grad_()
                ref_input = test_input.clone().detach().requires_grad_()

                ref_out = ref_ln(ref_input)

                test_out = fast_ln(test_input)

                np.testing.assert_allclose(
                    ref_out.detach().cpu().numpy(),
                    test_out.detach().cpu().numpy(),
                    atol=1e-5, rtol=1e-4)

                test_out.mean().mul(10.).backward()
                ref_out.mean().mul(10.).backward()
                np.testing.assert_allclose(
                    test_input.grad.detach().cpu().numpy(),
                    ref_input.grad.detach().cpu().numpy(),
                    atol=1e-7, rtol=1e-5)
                np.testing.assert_allclose(
                    fast_ln.weight.grad.detach().cpu().numpy(),
                    ref_ln.weight.grad.detach().cpu().numpy(),
                    atol=1e-7, rtol=1e-5)
                np.testing.assert_allclose(
                    fast_ln.bias.grad.detach().cpu().numpy(),
                    ref_ln.bias.grad.detach().cpu().numpy(),
                    atol=1e-7, rtol=1e-5)

        def test_performance_half_memefficient(self):
            num_iters = 64
            bsz = 128
            seq_len = 512
            print("Testing performance ...")
            hidden_sizes = [
                768,
                1024,
                1280,
                1536,
                2048,
                2304,
                3072,
                3840,
                4096,
                5120,
                6144,
                8192,
                10240,
                12288,
                12800,
                15360,
                # 16384,
                # 18432,
                # 20480,
                # 24576,
                # 25600,
                # 30720,
                # 32768,
                # 40960,
                # 49152,
                # 65536,
            ]
            for hidden in hidden_sizes:
                ref_ln = nn.LayerNorm(hidden).cuda().half()
                fast_ln = FastLayerNorm(hidden, memory_efficient=True).cuda().half()
                print(fast_ln, ref_ln)

                test_input = torch.empty(seq_len, bsz, hidden, device="cuda", dtype=torch.half).uniform_(-1., 1.).requires_grad_()
                ref_input = test_input.clone().detach().requires_grad_()

                # Warm up GPU
                for _ in range(num_iters):
                    ref_out = ref_ln(ref_input)
                    ref_loss = ref_out.mean()
                    ref_ln.zero_grad()
                    ref_loss.backward()
                    test_out = fast_ln(test_input)
                    test_loss = test_out.mean()
                    fast_ln.zero_grad()
                    test_loss.backward()

                torch.cuda.profiler.start()
                torch.cuda.synchronize()
                start_time = time()
                for _ in range(num_iters):
                    ref_out = ref_ln(ref_input)
                    ref_loss = ref_out.mean()
                    ref_ln.zero_grad()
                    ref_loss.backward()
                torch.cuda.synchronize()
                stop_time = time()
                print(F"\nPytorch Layer Norm time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")

                torch.cuda.synchronize()
                start_time = time()
                for _ in range(num_iters):
                    test_out = fast_ln(test_input)
                    test_loss = test_out.mean()
                    fast_ln.zero_grad()
                    test_loss.backward()
                torch.cuda.synchronize()
                stop_time = time()
                print(F"Custom Layer Norm time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
                torch.cuda.profiler.stop()

    unittest.main()