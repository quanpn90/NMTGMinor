from copy import copy
import math
import torch
from torch import nn
import unittest
from time import time
import numpy as np

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

# try:
#     import fused_relu_mlp
# except (ModuleNotFoundError, ImportError) as e:
#     fused_relu_mlp = None
#
# try:
#     import fused_gelu_mlp
# except (ModuleNotFoundError, ImportError) as e:
#     fused_gelu_mlp = None

try:
    import fused_mlp
except (ModuleNotFoundError, ImportError) as e:
    fused_mlp = None

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
# class MlpGeluFunction(torch.autograd.Function):
#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float16)
#     def forward(ctx, activation, *args):
#         output = fused_gelu_mlp.forward(args)
#         ctx.save_for_backward(*args)
#         ctx.outputs = output
#         return output[0]
#
#     @staticmethod
#     @custom_bwd
#     def backward(ctx, grad_o):
#         grads = fused_gelu_mlp.backward(grad_o, ctx.outputs, ctx.saved_tensors)
#         del ctx.outputs
#         return (None, *grads)


class MlpFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, activation, *args):
        output = fused_mlp.forward(activation, args)
        ctx.save_for_backward(*args)
        ctx.outputs = output
        ctx.activation = activation
        return output[0]

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_o):
        grads = fused_mlp.backward(ctx.activation, grad_o, ctx.outputs, ctx.saved_tensors)
        del ctx.outputs
        return (None, *grads)


# if fused_relu_mlp is not None:
#     mlp_relu_function = half_function(MlpReluFunction.apply)
# else:
#     mlp_relu_function = None
#
# if fused_gelu_mlp is not None:
#     mlp_gelu_function = half_function(MlpGeluFunction.apply)
# else:
#     mlp_gelu_function = None

if fused_mlp:
    mlp_function = half_function(MlpFunction.apply)
else:
    mlp_function = None

#

if __name__ == '__main__':

    class GELU_(torch.nn.Module):
        def forward(self, x):
            xf = x.float()
            return (0.5 * xf * (1 + torch.tanh(math.sqrt(2 / math.pi) * (xf + 0.044715 * torch.pow(xf, 3))))).type_as(x)

    class MLP(torch.nn.Module):
        """Launch MLP in C++

        Args:
            mlp_sizes (list of int): MLP sizes. Example: [1024,1024,1024] will create 2 MLP layers with shape 1024x1024
            bias (bool): Default True:
            relu (bool): Default True
        """

        def __init__(self, mlp_sizes, activation='relu'):
            super(MLP, self).__init__()
            self.num_layers = len(mlp_sizes) - 1
            self.mlp_sizes = copy(mlp_sizes)

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

        def forward(self, input):

            return mlp_function(self.activation, input, *self.weights, *self.biases)
            # return mlp_function(self.bias, self.dropout_prob, self.activation, input, *self.weights, *self.biases)
            # if self.activation == 1:
            #     return mlp_relu_function(input, *self.weights, *self.biases)
            # elif self.activation == 3:
            #     return mlp_gelu_function(input, *self.weights, *self.biases)

        def extra_repr(self):
            # TODO add dropout probability
            s = F"MLP sizes: {self.mlp_sizes},  activation={self.activation}"
            return s

    batch_size = 24568
    # mlp_sizes = [480, 1024, 1024, 512, 256, 1]
    mlp_sizes = [512, 4096, 512]
    num_iters = 10


    class TestMLP(unittest.TestCase):

        def test_creation(self):
            MLP(mlp_sizes)

        def test_numeric(self):
            mlp = MLP(mlp_sizes, activation='relu').cuda()

            mlp_layers = []
            for i in range(mlp.num_layers):
                linear = nn.Linear(mlp_sizes[i], mlp_sizes[i + 1])
                mlp.weights[i].data.copy_(linear.weight)
                mlp.biases[i].data.copy_(linear.bias)
                mlp_layers.append(linear)
                if i < mlp.num_layers - 1:
                    mlp_layers.append(nn.ReLU(inplace=True))
                    # mlp_layers.append(nn.GELU())

            ref_mlp = nn.Sequential(*mlp_layers).cuda()
            print(ref_mlp)
            print(mlp)

            test_input = torch.empty(batch_size, mlp_sizes[0], device="cuda").uniform_(-1., 1.).requires_grad_()
            ref_input = test_input.clone().detach().requires_grad_()
            mlp_out = mlp(test_input)
            ref_out = ref_mlp(ref_input)
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
                atol=0, rtol=1e-5)
            np.testing.assert_allclose(
                mlp.biases[0].grad.detach().cpu().numpy(),
                ref_mlp[0].bias.grad.detach().cpu().numpy(),
                atol=1e-7, rtol=1e-5)

        def test_with_bias(self):
            for use_activation in ['relu', 'gelu']:
                mlp = MLP(mlp_sizes, activation=use_activation).cuda()

                mlp_layers = []
                for i in range(mlp.num_layers):
                    linear = nn.Linear(mlp_sizes[i], mlp_sizes[i + 1], bias=True)
                    mlp.weights[i].data.copy_(linear.weight)
                    mlp.biases[i].data.copy_(linear.bias)
                    mlp_layers.append(linear)
                    if i < mlp.num_layers - 1:
                        if use_activation == 'relu':
                            mlp_layers.append(nn.ReLU(inplace=True))
                        elif use_activation == 'sigmoid':
                            mlp_layers.append(nn.Sigmoid())
                        elif use_activation == 'gelu':
                            # mlp_layers.append(nn.GELU())
                            mlp_layers.append(GELU_())

                ref_mlp = nn.Sequential(*mlp_layers).cuda()
                print(ref_mlp)

                test_input = torch.empty(batch_size, mlp_sizes[0], device="cuda").uniform_(-1., 1.).requires_grad_()
                ref_input = test_input.clone().detach().requires_grad_()
                mlp_out = mlp(test_input)
                ref_out = ref_mlp(ref_input)
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
                    atol=0, rtol=1)
                np.testing.assert_allclose(
                    mlp.weights[0].grad.detach().cpu().numpy(),
                    ref_mlp[0].weight.grad.detach().cpu().numpy(),
                    atol=1e-7, rtol=1)
                np.testing.assert_allclose(
                    mlp.biases[0].grad.detach().cpu().numpy(),
                    ref_mlp[0].bias.grad.detach().cpu().numpy(),
                    atol=1e-7, rtol=1e-5)

        def test_no_grad(self):
            mlp = MLP(mlp_sizes).cuda()

            mlp_layers = []
            for i in range(mlp.num_layers):
                linear = nn.Linear(mlp_sizes[i], mlp_sizes[i + 1])
                mlp.weights[i].data.copy_(linear.weight)
                mlp.biases[i].data.copy_(linear.bias)
                mlp_layers.append(linear)
                if i < mlp.num_layers - 1:
                    mlp_layers.append(nn.ReLU(inplace=True))

            ref_mlp = nn.Sequential(*mlp_layers).cuda()

            test_input = torch.empty(batch_size, mlp_sizes[0], device="cuda").uniform_(-1., 1.)
            ref_input = test_input.clone().detach()
            mlp_out = mlp(test_input)
            ref_out = ref_mlp(ref_input)
            np.testing.assert_allclose(
                mlp_out.detach().cpu().numpy(),
                ref_out.detach().cpu().numpy(),
                atol=1e-7, rtol=1e-5)

            # Use mean value as scalar loss. Multiply 10 to make it big enough not zero out
            mlp_out.mean().mul(10.).backward()
            ref_out.mean().mul(10.).backward()
            np.testing.assert_allclose(
                mlp.weights[0].grad.detach().cpu().numpy(),
                ref_mlp[0].weight.grad.detach().cpu().numpy(),
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
                    mlp_layers.append(nn.ReLU(inplace=True))

            ref_mlp = nn.Sequential(*mlp_layers).cuda().half()

            test_input = torch.empty(
                batch_size, mlp_sizes[0], device="cuda", dtype=torch.half).fill_(10.).requires_grad_()
            ref_input = torch.empty(
                batch_size, mlp_sizes[0], device="cuda", dtype=torch.half).fill_(10.).requires_grad_()

            # Warm up GPU
            for _ in range(100):
                ref_out = ref_mlp(ref_input)
                ref_loss = ref_out.mean()
                ref_mlp.zero_grad()
                ref_loss.backward()
                mlp_out = mlp(test_input)
                test_loss = mlp_out.mean()
                mlp.zero_grad()
                test_loss.backward()

            torch.cuda.profiler.start()
            torch.cuda.synchronize()
            start_time = time()
            for _ in range(num_iters):
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
                mlp_out = mlp(test_input)
                test_loss = mlp_out.mean()
                mlp.zero_grad()
                test_loss.backward()
            torch.cuda.synchronize()
            stop_time = time()
            print(F"C++ MLP time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
            torch.cuda.profiler.stop()


    unittest.main()

    test = TestMLP()

    # test.test_creation()
    # test.test_performance_half()
    # test.test_with_bias()
