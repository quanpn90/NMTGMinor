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


class MlpReluFunction(torch.autograd.Function):
    @staticmethod
    # @custom_fwd(cast_inputs=torch.float16)
    @custom_fwd
    def forward(ctx, p, *args):
        store_dropout_mask = True
        output = fused_mlp_relu.forward(p, store_dropout_mask, args)
        ctx.save_for_backward(*args)
        ctx.outputs = output
        dropout_mask = output[-1]
        ctx.p = p
        return output[0], dropout_mask

    @staticmethod
    @custom_bwd
    def backward(ctx, *grad_o):
        p = ctx.p
        grads = fused_mlp_relu.backward(p, grad_o[0], ctx.outputs, ctx.saved_tensors)
        del ctx.outputs
        return (None, *grads)


if fused_mlp_relu:
    mlp_relu_function = half_function(MlpReluFunction.apply)
else:
    mlp_relu_function = None


class MlpReluRecomputeFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, p, *args):
        store_dropout_mask = True
        output = fused_mlp_relu.forward(p, store_dropout_mask, args)
        ctx.save_for_backward(*args)
        dropout_mask = output[-1]
        ctx.dropout_mask = dropout_mask
        ctx.p = p
        return output[0], dropout_mask

    @staticmethod
    @custom_bwd
    def backward(ctx, *grad_o):
        p = ctx.p
        dropout_mask = ctx.dropout_mask
        grads = fused_mlp_relu.backward_recompute(p, grad_o[0], dropout_mask, ctx.saved_tensors)
        del ctx.dropout_mask
        return (None, *grads)


if fused_mlp_relu:
    mlp_relu_recompute_function = half_function(MlpReluRecomputeFunction.apply)
else:
    mlp_relu_recompute_function = None


if __name__ == '__main__':

    class MLP(torch.nn.Module):
        """Launch MLP in C++

        Args:
            mlp_sizes (list of int): MLP sizes. Example: [1024,1024,1024] will create 2 MLP layers with shape 1024x1024
            bias (bool): Default True:
            relu (bool): Default True
        """

        def __init__(self, mlp_sizes, activation='relu', dropout=0.25, recompute=False):
            super(MLP, self).__init__()
            self.num_layers = len(mlp_sizes) - 1
            self.mlp_sizes = copy(mlp_sizes)
            self.dropout = dropout
            self.recompute = recompute

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

        def forward(self, input, mask=None, ref=False):

            if ref:
                return self.forward_ref(input, mask)
            # return mlp_relu_function(self.dropout, input, *self.weights, *self.biases)

            # return mlp_agelu_function(self.dropout, input, *self.weights, *self.biases)
            # return mlp_relu_function(self.dropout, input, *self.weights, *self.biases)
            if self.recompute:
                return mlp_relu_recompute_function(self.dropout, input, *self.weights, *self.biases)
            else:
                return mlp_relu_function(self.dropout, input, *self.weights, *self.biases)

        def forward_ref(self, input, mask):

            i = 0
            output = input
            for l in range(self.num_layers):
                output = F.linear(output, self.weights[l], self.biases[l])

                dropout_mask = mask[i:i + output.numel()]
                pinv = 1 / (1 - self.dropout)
                if l < self.num_layers - 1:
                    # print(mask.size())
                    # output = fast_silu(output) * dropout_mask.view(output.size(0), -1) * pinv
                    # output = GELUFunction.apply(output) * dropout_mask.view(output.size(0), -1) * pinv
                    if self.dropout > 0:
                        output = F.relu(output) * dropout_mask.view(output.size(0), -1) * pinv
                    else:
                        output = F.relu(output)

                i += output.numel()

            return output

        def extra_repr(self):
            # TODO add dropout probability
            s = F"MLP sizes: {self.mlp_sizes},  activation={self.activation}"
            return s


    batch_size = 24568
    mlp_sizes = [512, 4096, 512]
    # mlp_sizes = [4, 7, 4]
    num_iters = 10


    class TestMLP(unittest.TestCase):

        def test_creation(self):
            MLP(mlp_sizes)

        def test_numeric(self):
            mlp = MLP(mlp_sizes, activation='relu').cuda()
            print(mlp)
            ref_mlp = deepcopy(mlp)

            for _ in range(1):
                bsz = random.randint(64, batch_size // 8) * 8
                test_input = torch.empty(bsz, mlp_sizes[0], device="cuda").uniform_(-1., 1.).requires_grad_()
                ref_input = test_input.clone().detach().requires_grad_()
                mlp_out, dropout_mask = mlp(test_input)
                ref_out = ref_mlp.forward(ref_input, dropout_mask, ref=True)

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

        def test_with_bias(self):
            for use_activation in ['relu']:
                mlp = MLP(mlp_sizes, activation=use_activation).cuda()

                ref_mlp = deepcopy(mlp)

                test_input = torch.empty(batch_size, mlp_sizes[0], device="cuda").uniform_(-1., 1.).requires_grad_()
                ref_input = test_input.clone().detach().requires_grad_()
                mlp_out, dropout_mask = mlp(test_input)
                ref_out = ref_mlp(ref_input, dropout_mask, ref=True)
                np.testing.assert_allclose(
                    mlp_out.detach().cpu().numpy(),
                    ref_out.detach().cpu().numpy(),
                    atol=1e-7, rtol=1e-5)

                # Use mean value as scalar loss. Multiply 10 to make it big enough not zero out
                mlp_out.mean().mul(10.).backward()
                ref_out.mean().mul(10.).backward()
                np.testing.assert_allclose(
                    test_input.grad.detach().cpu().numpy(),
                    ref_input.grad.detach().cpu().numpy(),
                    atol=1e-5, rtol=1e-4)

                for l in range(mlp.num_layers):
                    np.testing.assert_allclose(
                        mlp.weights[l].grad.detach().cpu().numpy(),
                        ref_mlp.weights[l].grad.detach().cpu().numpy(),
                        atol=1e-7, rtol=1e-5)
                    np.testing.assert_allclose(
                        mlp.biases[l].grad.detach().cpu().numpy(),
                        ref_mlp.biases[l].grad.detach().cpu().numpy(),
                        atol=1e-7, rtol=1e-5)

        def test_no_grad(self):
            mlp = MLP(mlp_sizes).cuda()
            ref_mlp = deepcopy(mlp)

            test_input = torch.empty(batch_size, mlp_sizes[0], device="cuda").uniform_(-1., 1.)
            ref_input = test_input.clone().detach()
            mlp_out, dropout_mask = mlp(test_input)

            ref_out = ref_mlp(ref_input, dropout_mask, ref=True)
            np.testing.assert_allclose(
                mlp_out.detach().cpu().numpy(),
                ref_out.detach().cpu().numpy(),
                atol=1e-7, rtol=1e-5)

        def test_performance_half(self):
            mlp = MLP(mlp_sizes).cuda().half()

            ref_mlp_fast = MLP(mlp_sizes, recompute=False).cuda().half()

            mlp_layers = []
            for i in range(mlp.num_layers):
                linear = nn.Linear(mlp_sizes[i], mlp_sizes[i + 1])
                mlp.weights[i].data.copy_(linear.weight)
                mlp.biases[i].data.copy_(linear.bias)
                mlp_layers.append(linear)
                if i < mlp.num_layers - 1:
                    # mlp_layers.append(nn.ReLU(inplace=True))
                    mlp_layers.append(torch.nn.ReLU())
                    mlp_layers.append(nn.Dropout(0.25))

            ref_mlp = nn.Sequential(*mlp_layers).cuda().half()

            test_input = torch.empty(
                batch_size, mlp_sizes[0], device="cuda", dtype=torch.half).fill_(10.).requires_grad_()
            ref_input = torch.empty(
                batch_size // 2, mlp_sizes[0], device="cuda", dtype=torch.half).fill_(10.).requires_grad_()

            # Warm up GPU
            for _ in range(100):
                ref_out = ref_mlp(ref_input)
                ref_loss = ref_out.mean()
                ref_mlp.zero_grad()
                ref_loss.backward()
                mlp_out, _ = mlp(test_input)
                test_loss = mlp_out.mean()
                mlp.zero_grad()
                test_loss.backward()

            torch.cuda.profiler.start()
            torch.cuda.synchronize()
            start_time = time()
            for _ in range(num_iters * 2):
                ref_out = ref_mlp(ref_input)
                ref_loss = ref_out.mean()
                ref_mlp.zero_grad()
                ref_loss.backward()
            torch.cuda.synchronize()
            stop_time = time()
            print(F"\nPytorch MLP time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")

            torch.cuda.synchronize()
            start_time = time()
            for _ in range(num_iters):
                mlp_out, _ = mlp(test_input)
                test_loss = mlp_out.mean()
                mlp.zero_grad()
                test_loss.backward()
            torch.cuda.synchronize()
            stop_time = time()
            print(F"C++ MLP recompute time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
            torch.cuda.profiler.stop()

            torch.cuda.synchronize()
            start_time = time()
            for _ in range(num_iters * 2):
                mlp_out, _ = ref_mlp_fast(ref_input)
                test_loss = mlp_out.mean()
                ref_mlp.zero_grad()
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
