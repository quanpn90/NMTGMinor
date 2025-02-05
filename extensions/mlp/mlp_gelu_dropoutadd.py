from copy import copy, deepcopy
import math
import torch
from torch import nn
import torch.nn.functional as F
import unittest
from time import time
import numpy as np
import random

import silu_cuda

try:
    import apex.amp as amp
    from apex.amp import half_function
except (ModuleNotFoundError, ImportError) as e:
    amp = None
    from ..optimized.compat import half_function

try:
    from torch.cuda.amp import custom_fwd, custom_bwd
except (ModuleNotFoundError, ImportError) as e:
    from ..optimized.compat import custom_fwd, custom_bwd

try:
    import fused_mlp_relu
except (ModuleNotFoundError, ImportError) as e:
    fused_mlp_relu = None

try:
    import fused_mlp_agelu
except (ModuleNotFoundError, ImportError) as e:
    fused_mlp_agelu = None

try:
    import fused_mlp_gelu_dropout_add
except (ModuleNotFoundError, ImportError) as e:
    fused_mlp_gelu_dropout_add = None


#
# class MlpReluFunction(torch.autograd.Function):
#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float16)
#     def forward(ctx, activation, *args):
#         output = fused_mlp.forward(args)
#         ctx.save_for_backward(*args)
#         ctx.outputs = output
#         return output[0]
#
#     @staticmethod
#     @custom_bwd
#     def backward(ctx, grad_o):
#         grads = fused_mlp.backward(grad_o, ctx.outputs, ctx.saved_tensors)
#         del ctx.outputs
#         return (None, *grads)
#
#



class MlpGeLUDropoutAddFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, p, r_p, *args):
        outputs = fused_mlp_gelu_dropout_add.forward(p, r_p, args)
        ctx.save_for_backward(*args)
        ctx.outputs = outputs
        dropout_mask = outputs[-2]
        residual_mask = outputs[-1]
        ctx.p = p
        ctx.r_p = p
        return outputs[0], dropout_mask, residual_mask

    @staticmethod
    @custom_bwd
    def backward(ctx, *grad_o):
        p = ctx.p
        r_p = ctx.r_p
        grads = fused_mlp_gelu_dropout_add.backward(p, r_p, grad_o[0], ctx.outputs, ctx.saved_tensors)
        del ctx.outputs
        return (None, None, *grads)


if fused_mlp_gelu_dropout_add:
    mlp_gelu_dropout_add_function = half_function(MlpGeLUDropoutAddFunction.apply)
else:
    mlp_gelu_dropout_add_function = None


class SwishFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return silu_cuda.forward(inp)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        if not ctx.needs_input_grad[0]: return (None,)
        return silu_cuda.backward(inp, grad_out)


def fast_silu(input):
    return SwishFunction.apply(input)


class FastSiLU(torch.nn.Module):

    def forward(self, input):
        return fast_silu(input)


class AGELUFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x):
        ctx.save_for_backward(x)
        SQRT_M2_PI = 0.7978845608
        COEFF = 0.044715
        return 0.5 * x * (1.0 + torch.tanh(SQRT_M2_PI * (x + COEFF * torch.pow(x, 3))))

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        SQRT_M2_PI = 0.7978845608
        COEFF = 0.044715
        BACKCOEFF = 0.1070322243

        tanh_outf = torch.tanh(SQRT_M2_PI * (x + COEFF * torch.pow(x, 3)))
        retf = 0.5 * x * (1.0 - torch.pow(tanh_outf, 2)) * (SQRT_M2_PI + BACKCOEFF * torch.pow(x, 2)) + 0.5 * (
                1.0 + tanh_outf)
        return grad_out * retf


if __name__ == '__main__':

    class MLP(torch.nn.Module):
        """Launch MLP in C++

        Args:
            mlp_sizes (list of int): MLP sizes. Example: [1024,1024,1024] will create 2 MLP layers with shape 1024x1024
            bias (bool): Default True:
            relu (bool): Default True
        """

        def __init__(self, mlp_sizes, activation='gelu', dropout=0.0, res_dropout=0.0):
            super(MLP, self).__init__()
            self.num_layers = len(mlp_sizes) - 1
            self.mlp_sizes = copy(mlp_sizes)
            self.dropout = dropout
            self.res_dropout = res_dropout

            if activation is 'relu':
                self.activation = 1
            elif activation is 'sigmoid':
                self.activation = 2
            elif activation is 'gelu':
                self.activation = 3
            else:
                raise TypeError("activation must be relu or none.")

            self.weights = []
            self.biases = []
            for i in range(self.num_layers):
                w = torch.nn.Parameter(torch.empty(mlp_sizes[i + 1], mlp_sizes[i]))
                self.weights.append(w)
                name = 'weight_{}'.format(i)
                setattr(self, name, w)
                b = torch.nn.Parameter(torch.empty(mlp_sizes[i + 1]))
                self.biases.append(b)
                name = 'bias_{}'.format(i)
                setattr(self, name, b)

            self.reset_parameters()

        def reset_parameters(self):
            for weight in self.weights:
                dimsum = weight.size(0) + weight.size(1)
                std = math.sqrt(2. / float(dimsum))
                nn.init.normal_(weight, 0., std)
            for bias in self.biases:
                std = math.sqrt(1. / float(bias.size(0)))
                nn.init.normal_(bias, 0., std)

        def forward(self, input, mask=None, residual_mask=None, ref=False):

            if ref:
                return self.forward_ref(input, mask, residual_mask)
            # return mlp_relu_function(self.dropout, input, *self.weights, *self.biases)

            # return mlp_agelu_function(self.dropout, input, *self.weights, *self.biases)
            return mlp_gelu_dropout_add_function(self.dropout, self.res_dropout, input, *self.weights, *self.biases)

        def forward_ref(self, input, mask, res_mask):

            i = 0
            output = input
            for l in range(self.num_layers):
                output = F.linear(output, self.weights[l], self.biases[l])

                dropout_mask = mask[i:i + output.numel()]
                pinv = 1 / (1 - self.dropout)
                if l < self.num_layers - 1:
                    if self.dropout > 0:
                        output = F.gelu(output) * dropout_mask.view(output.size(0), -1) * pinv
                    else:
                        output = F.gelu(output)

                i += output.numel()

            pinv = 1 / (1 - self.res_dropout)
            output = output * res_mask.view(output.size(0), -1) * pinv + input if self.res_dropout > 0 else output + input

            return output

        def extra_repr(self):
            # TODO add dropout probability
            s = F"MLP sizes: {self.mlp_sizes},  activation={self.activation}"
            return s


    batch_size = 24568
    mlp_sizes = [1024, 4096, 1024]
    # mlp_sizes = [4, 7, 4]
    num_iters = 20


    class TestMLP(unittest.TestCase):

        def test_creation(self):
            MLP(mlp_sizes)

        def test_numeric(self):
            mlp = MLP(mlp_sizes, activation='gelu', dropout=0.0, res_dropout=0.0).cuda()

            print(mlp)
            ref_mlp = deepcopy(mlp)

            for _ in range(8):
                bsz = random.randint(2850, batch_size // 8) * 8
                test_input = torch.empty(bsz, mlp_sizes[0], device="cuda").uniform_(-1., 1.).requires_grad_()
                ref_input = test_input.clone().detach().requires_grad_()
                mlp_out, dropout_mask, residual_mask = mlp(test_input)
                ref_out = ref_mlp.forward(ref_input, dropout_mask, residual_mask, ref=True)

                print(dropout_mask.sum() / dropout_mask.numel())
                np.testing.assert_allclose(
                    mlp_out.detach().cpu().numpy(),
                    ref_out.detach().cpu().numpy(),
                    atol=1e-5, rtol=1e-4)

                # Use mean value as scalar loss. Multiply 10 to make it big enough not zero out
                mlp_out.mean().mul(10.).backward()
                ref_out.mean().mul(10.).backward()
                np.testing.assert_allclose(
                    test_input.grad.detach().cpu().numpy(),
                    ref_input.grad.detach().cpu().numpy(),
                    atol=1e-7, rtol=1e-5)
                np.testing.assert_allclose(
                    mlp.biases[0].grad.detach().cpu().numpy(),
                    ref_mlp.biases[0].grad.detach().cpu().numpy(),
                    atol=1e-7, rtol=1e-5)

        def test_no_grad(self):
            mlp = MLP(mlp_sizes).cuda()
            ref_mlp = deepcopy(mlp)

            test_input = torch.empty(batch_size, mlp_sizes[0], device="cuda").uniform_(-1., 1.)
            ref_input = test_input.clone().detach()
            mlp_out, dropout_mask, residual_mask = mlp(test_input)

            ref_out = ref_mlp(ref_input, dropout_mask, residual_mask, ref=True)
            np.testing.assert_allclose(
                mlp_out.detach().cpu().numpy(),
                ref_out.detach().cpu().numpy(),
                atol=1e-7, rtol=1e-5)

        def test_performance_half(self):
            mlp = MLP(mlp_sizes).cuda().half()

            mlp_layers = []
            for i in range(mlp.num_layers):
                linear = nn.Linear(mlp_sizes[i], mlp_sizes[i + 1])
                mlp.weights[i].data.copy_(linear.weight)
                mlp.biases[i].data.copy_(linear.bias)
                mlp_layers.append(linear)
                if i < mlp.num_layers - 1:
                    # mlp_layers.append(nn.ReLU(inplace=True))
                    mlp_layers.append(torch.nn.GELU())
                    mlp_layers.append(nn.Dropout(0.0))
                else:
                    mlp_layers.append(nn.Dropout(0.0))

            ref_mlp = nn.Sequential(*mlp_layers).cuda().half()

            test_input = torch.empty(
                batch_size, mlp_sizes[0], device="cuda", dtype=torch.half).fill_(10.).requires_grad_()
            ref_input = torch.empty(
                batch_size, mlp_sizes[0], device="cuda", dtype=torch.half).fill_(10.).requires_grad_()

            # Warm up GPU
            for _ in range(100):
                ref_out = ref_mlp(ref_input) + ref_input
                ref_loss = ref_out.mean()
                ref_mlp.zero_grad()
                ref_loss.backward()
                mlp_out, _, _ = mlp(test_input)
                test_loss = mlp_out.mean()
                mlp.zero_grad()
                test_loss.backward()

            torch.cuda.profiler.start()
            torch.cuda.synchronize()
            start_time = time()
            for _ in range(num_iters):
                ref_out = ref_mlp(ref_input) + ref_input
                ref_loss = ref_out.mean()
                ref_mlp.zero_grad()
                ref_loss.backward()
            torch.cuda.synchronize()
            stop_time = time()
            print(F"\nPytorch MLP time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")

            torch.cuda.synchronize()
            start_time = time()
            for _ in range(num_iters):
                mlp_out, _, _ = mlp(test_input)
                test_loss = mlp_out.mean()
                mlp.zero_grad()
                test_loss.backward()
            torch.cuda.synchronize()
            stop_time = time()
            print(F"C++ MLP time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
            torch.cuda.profiler.stop()


    unittest.main()

    # test = TestMLP()

    # test.test_creation()
    # test.test_performance_half()
    # test.test_with_bias()
