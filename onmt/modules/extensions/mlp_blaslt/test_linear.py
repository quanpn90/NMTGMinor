from copy import copy, deepcopy
import math
import torch
from torch import nn
import torch.nn.functional as F
import unittest
from time import time
import numpy as np
import random
from torch import Tensor

try:
    from torch.cuda.amp import custom_fwd, custom_bwd
except (ModuleNotFoundError, ImportError) as e:
    from ..optimized.compat import custom_fwd, custom_bwd

try:
    import linear_blaslt
except (ModuleNotFoundError, ImportError) as e:
    linear_blaslt = None
torch.backends.cuda.matmul.allow_tf32 = False

def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        try:
            return torch.cuda.amp.autocast_mode._cast(args, torch.get_autocast_gpu_dtype())
        except AttributeError:
            return torch.cuda.amp.autocast_mode._cast(args, torch.half)


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):

        output = linear_blaslt.forward(input, weight, bias)
        ctx.save_for_backward(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, weight = ctx.saved_tensors

        if weight.requires_grad:
            d_input, d_weight, d_bias = linear_blaslt.backward(input, weight, grad_output)
        else:
            d_input = linear_blaslt.backward_input_only(input, weight, grad_output)
            d_weight, d_bias = None, None

        return d_input, d_weight, d_bias

if linear_blaslt:
    def linear_function(*args):
        args = _cast_if_autocast_enabled(*args)
        with torch.cuda.amp.autocast(enabled=False):
            return LinearFunction.apply(*args)
else:
    linear_function = torch.nn.functional.linear


class Linear(torch.nn.Linear):

    def forward(self, input: Tensor) -> Tensor:

        return linear_function(input, self.weight, self.bias)




if __name__ == '__main__':


    seq_len = 32
    batch_size = 64
    # linear_sizes = [1024, 4096, 1024]
    input_size = 1024
    output_size = 250000
    num_iters = 32


    class TestLinear(unittest.TestCase):

        def test_forward_float(self):

            test_linear = Linear(input_size, output_size).cuda()
            linear = test_linear

            ref_linear = torch.nn.Linear(input_size, output_size).cuda()

            ref_linear.weight.data.copy_(test_linear.weight.data)
            ref_linear.bias.data.copy_(test_linear.bias.data)

            test_input = torch.empty(seq_len, batch_size, input_size,
                                     device="cuda").uniform_(-0.01, 0.01).requires_grad_()

            ref_input = test_input.clone().detach().requires_grad_()

            linear_out = test_linear(test_input)
            ref_out = ref_linear(ref_input)

            grad = torch.randn_like(linear_out)
            np.testing.assert_allclose(
                linear_out.detach().cpu().numpy(),
                ref_out.detach().cpu().numpy(),
                atol=1e-5, rtol=1e-5)

            linear_out.mul_(1).backward(grad)
            ref_out.mul_(1).backward(grad)
            np.testing.assert_allclose(
                test_input.grad.detach().cpu().numpy(),
                ref_input.grad.detach().cpu().numpy(),
                atol=1e-5, rtol=1e-5)

            np.testing.assert_allclose(
                linear.weight.grad.detach().cpu().numpy(),
                ref_linear.weight.grad.detach().cpu().numpy(),
                atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(
                linear.bias.grad.detach().cpu().numpy(),
                ref_linear.bias.grad.detach().cpu().numpy(),
                atol=1e-5, rtol=1e-5)

        def test_forward_float_input_only(self):

            test_linear = Linear(input_size, output_size).cuda()
            linear = test_linear

            ref_linear = torch.nn.Linear(input_size, output_size).cuda()

            for p in test_linear.parameters():
                p.requires_grad = False

            for p in ref_linear.parameters():
                p.requires_grad = False

            ref_linear.weight.data.copy_(test_linear.weight.data)
            ref_linear.bias.data.copy_(test_linear.bias.data)

            test_input = torch.empty(seq_len, batch_size, input_size,
                                     device="cuda").uniform_(-0.01, 0.01).requires_grad_()

            ref_input = test_input.clone().detach().requires_grad_()

            linear_out = test_linear(test_input)
            ref_out = ref_linear(ref_input)

            grad = torch.randn_like(linear_out)
            np.testing.assert_allclose(
                linear_out.detach().cpu().numpy(),
                ref_out.detach().cpu().numpy(),
                atol=1e-5, rtol=1e-5)

            linear_out.mul_(1).backward(grad)
            ref_out.mul_(1).backward(grad)
            np.testing.assert_allclose(
                test_input.grad.detach().cpu().numpy(),
                ref_input.grad.detach().cpu().numpy(),
                atol=1e-5, rtol=1e-5)

            # np.testing.assert_allclose(
            #     linear.weight.grad.detach().cpu().numpy(),
            #     ref_linear.weight.grad.detach().cpu().numpy(),
            #     atol=1e-5, rtol=1e-5)
            # np.testing.assert_allclose(
            #     linear.bias.grad.detach().cpu().numpy(),
            #     ref_linear.bias.grad.detach().cpu().numpy(),
            #     atol=1e-5, rtol=1e-5)
        #
        # def test_precision_half(self):
        #
        #     test_linear = Linear(input_size, output_size).half().cuda()
        #     linear = test_linear
        #
        #     ref_linear = torch.nn.Linear(input_size, output_size).half().cuda()
        #
        #     ref_linear.weight.data.copy_(test_linear.weight.data)
        #     ref_linear.bias.data.copy_(test_linear.bias.data)
        #
        #     test_input = torch.empty(seq_len, batch_size, input_size,
        #                              device="cuda").uniform_(-0.01, 0.01).half().requires_grad_()
        #
        #     ref_input = test_input.clone().detach().requires_grad_()
        #
        #     linear_out = test_linear(test_input)
        #     ref_out = ref_linear(ref_input)
        #
        #     grad = torch.randn_like(linear_out)
        #     np.testing.assert_allclose(
        #         linear_out.detach().cpu().numpy(),
        #         ref_out.detach().cpu().numpy(),
        #         atol=1e-3, rtol=1e-3)
        #
        #     linear_out.mul_(1).backward(grad)
        #     ref_out.mul_(1).backward(grad)
        #     np.testing.assert_allclose(
        #         test_input.grad.detach().cpu().numpy(),
        #         ref_input.grad.detach().cpu().numpy(),
        #         atol=1e-3, rtol=1e-3)
        #
        #     np.testing.assert_allclose(
        #         linear.weight.grad.detach().cpu().numpy(),
        #         ref_linear.weight.grad.detach().cpu().numpy(),
        #         atol=1e-3, rtol=1e-3)
        #     np.testing.assert_allclose(
        #         linear.bias.grad.detach().cpu().numpy(),
        #         ref_linear.bias.grad.detach().cpu().numpy(),
        #         atol=1e-3, rtol=1e-3)

        # def test_numeric(self):
        #     print("Test numeric 3D ....")
        #     for dropout in [0.0, 0.2, 0.5, 0.7]:
        #         linear = linear(linear_sizes, activation='relu', dropout=dropout).cuda()
        #
        #         print(linear)
        #         ref_linear = deepcopy(linear)
        #
        #         for _ in range(1):
        #             bsz = random.randint(8, batch_size // 8) * 8
        #             test_input = torch.empty(seq_len, bsz, linear_sizes[0], device="cuda").uniform_(-1.,
        #                                                                                          1.).requires_grad_()
        #             ref_input = test_input.clone().detach().requires_grad_()
        #             linear_out, dropout_mask = linear(test_input)
        #             ref_out = ref_linear.forward(ref_input, dropout_mask, ref=True)
        #
        #             print(dropout_mask.sum() / dropout_mask.numel(), dropout_mask.numel())
        #             np.testing.assert_allclose(
        #                 linear_out.detach().cpu().numpy(),
        #                 ref_out.detach().cpu().numpy(),
        #                 atol=1e-5, rtol=1e-4)
        #
        #             # Use mean value as scalar loss. Multiply 10 to make it big enough not zero out
        #             linear_out.mean().mul(10.).backward()
        #             ref_out.mean().mul(10.).backward()
        #             np.testing.assert_allclose(
        #                 test_input.grad.detach().cpu().numpy(),
        #                 ref_input.grad.detach().cpu().numpy(),
        #                 atol=1e-7, rtol=1e-5)
        #             np.testing.assert_allclose(
        #                 linear.biases[0].grad.detach().cpu().numpy(),
        #                 ref_linear.biases[0].grad.detach().cpu().numpy(),
        #                 atol=1e-7, rtol=1e-5)




        def test_performance_half(self):
            print("Testing performance ...")

            test_linear = Linear(input_size, output_size).half().cuda()
            linear = test_linear

            ref_linear = torch.nn.Linear(input_size, output_size).half().cuda()

            test_input = torch.empty(
                seq_len, batch_size, input_size, device="cuda", dtype=torch.half).fill_(10.).requires_grad_()
            ref_input = torch.empty(
                seq_len, batch_size, input_size, device="cuda", dtype=torch.half).fill_(10.).requires_grad_()

            # Warm up GPU
            for _ in range(num_iters):
                ref_out = ref_linear(ref_input)
                ref_loss = ref_out.mean()
                ref_linear.zero_grad()
                ref_loss.backward()
                linear_out = linear(test_input)
                test_loss = linear_out.mean()
                linear.zero_grad()
                test_loss.backward()

            torch.cuda.profiler.start()
            torch.cuda.synchronize()
            start_time = time()
            for _ in range(num_iters):
                ref_out = ref_linear(ref_input)
                ref_loss = ref_out.mean()
                ref_linear.zero_grad()
                ref_loss.backward()
            torch.cuda.synchronize()
            stop_time = time()
            print(F"\nPytorch linear time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")

            torch.cuda.synchronize()
            start_time = time()
            for _ in range(num_iters):
                linear_out = linear(test_input)
                test_loss = linear_out.mean()
                linear.zero_grad()
                test_loss.backward()
            torch.cuda.synchronize()
            stop_time = time()
            print(F"C++ linear BLASLT time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
            torch.cuda.profiler.stop()


    unittest.main()