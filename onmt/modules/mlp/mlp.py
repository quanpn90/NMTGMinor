from copy import copy
import math
import torch
from torch import nn
import unittest
from time import time
import numpy as np

try:
    from torch.cuda.amp import custom_fwd, custom_bwd
except (ModuleNotFoundError, ImportError) as e:
    from ..optimized.compat import custom_fwd, custom_bwd

try:
    import fused_mlp_relu
except (ModuleNotFoundError, ImportError) as e:
    fused_mlp_relu = None

try:
    import fused_mlp_silu
except (ModuleNotFoundError, ImportError) as e:
    fused_mlp_silu = None

try:
    import fused_mlp_gelu
except (ModuleNotFoundError, ImportError) as e:
    fused_mlp_gelu = None

try:
    import fused_mlp_agelu
except (ModuleNotFoundError, ImportError) as e:
    fused_mlp_agelu = None

try:
    import fused_mlp_gelu_dropout_add
except (ModuleNotFoundError, ImportError) as e:
    fused_mlp_gelu_dropout_add = None

try:
    import mlp_gelu_blaslt
except (ModuleNotFoundError, ImportError) as e:
    mlp_gelu_blaslt = None


def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        try:
            return torch.cuda.amp.autocast_mode._cast(args, torch.get_autocast_gpu_dtype())
        except AttributeError:
            return torch.cuda.amp.autocast_mode._cast(args, torch.half)


class MlpReluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, recompute, *args):

        # only need to store dropout mask if we need to recompute
        store_dropout_mask = recompute
        output = fused_mlp_relu.forward(p, store_dropout_mask, args)
        ctx.save_for_backward(*args)
        ctx.recompute = recompute

        if not recompute:
            ctx.outputs = output
            ctx.dropout_mask = None
        else:
            ctx.dropout_mask = output[-1]
            ctx.outputs = None

        ctx.p = p
        return output[0]

    @staticmethod
    def backward(ctx, *grad_o):
        p = ctx.p
        if not ctx.recompute:
            grads = fused_mlp_relu.backward(p, grad_o[0], ctx.outputs, ctx.saved_tensors)
            del ctx.outputs
        else:
            grads = fused_mlp_relu.backward_recompute(p, grad_o[0], ctx.dropout_mask, ctx.saved_tensors)
            del ctx.dropout_mask

        return (None, None, *grads)


if fused_mlp_relu:
    def mlp_relu_function(*args):
        args = _cast_if_autocast_enabled(*args)
        with torch.cuda.amp.autocast(enabled=False):
            return MlpReluFunction.apply(*args)
else:
    mlp_relu_function = None


class MlpSiluFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, p, recompute, *args):
        output = fused_mlp_silu.forward(p, args)
        ctx.save_for_backward(*args)
        ctx.outputs = output
        dropout_mask = output[-1]
        ctx.p = p
        return output[0]

    @staticmethod
    @custom_bwd
    def backward(ctx, *grad_o):
        p = ctx.p
        grads = fused_mlp_silu.backward(p, grad_o[0], ctx.outputs, ctx.saved_tensors)
        del ctx.outputs
        return (None, None, *grads)


if fused_mlp_silu:
    # mlp_silu_function = MlpSiluFunction.apply
    def mlp_silu_function(*args):
        args = _cast_if_autocast_enabled(*args)
        with torch.cuda.amp.autocast(enabled=False):
            return MlpSiluFunction.apply(*args)
else:
    mlp_silu_function = None


class MlpGELUFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, p, recompute, *args):

        if mlp_gelu_blaslt is not None:
            output = mlp_gelu_blaslt.forward(p, args)
            if recompute:
                ctx.outputs = (output[0], output[-1])
                del output[1]
                del output[2]
            else:
                ctx.outputs = output
        else:
            output = fused_mlp_gelu.forward(p, args)
            ctx.outputs = output
        ctx.save_for_backward(*args)
        # dropout_mask = output[-1]
        ctx.p = p
        ctx.recompute = recompute
        ctx.requires_grad_weight = args[1].requires_grad
        return output[0]

    @staticmethod
    @custom_bwd
    def backward(ctx, *grad_o):
        p = ctx.p
        recompute = ctx.recompute
        if ctx.requires_grad_weight:
            if mlp_gelu_blaslt is not None:
                grads = mlp_gelu_blaslt.backward(p, recompute, grad_o[0], ctx.outputs, ctx.saved_tensors)
            else:
                grads = fused_mlp_gelu.backward(p, grad_o[0], ctx.outputs, ctx.saved_tensors)
        else:
            if mlp_gelu_blaslt is not None:
                grads = mlp_gelu_blaslt.backward_input_only(p, grad_o[0], ctx.outputs, ctx.saved_tensors)
            else:
                grads = fused_mlp_gelu.backward_input_only(p, grad_o[0], ctx.outputs, ctx.saved_tensors)
            for _ in range(len(ctx.saved_tensors) - 1):
                grads.append(None)

        del ctx.requires_grad_weight
        del ctx.outputs
        del ctx.p
        del ctx.recompute

        return (None, None, *grads)


if fused_mlp_gelu or mlp_gelu_blaslt:
    def mlp_gelu_function(*args):
        args = _cast_if_autocast_enabled(*args)
        with torch.cuda.amp.autocast(enabled=False):
            return MlpGELUFunction.apply(*args)
else:
    mlp_gelu_function = None


class MlpAGELUFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, p, recompute, *args):
        output = fused_mlp_agelu.forward(p, args)
        ctx.save_for_backward(*args)
        ctx.outputs = output
        dropout_mask = output[-1]
        ctx.p = p
        return output[0]

    @staticmethod
    @custom_bwd
    def backward(ctx, *grad_o):
        p = ctx.p
        grads = fused_mlp_agelu.backward(p, grad_o[0], ctx.outputs, ctx.saved_tensors)
        del ctx.outputs
        return (None, None, *grads)


if fused_mlp_agelu:
    def mlp_agelu_function(*args):
        args = _cast_if_autocast_enabled(*args)
        with torch.cuda.amp.autocast(enabled=False):
            return MlpAGELUFunction.apply(*args)
else:
    mlp_agelu_function = None

if __name__ == '__main__':

    from copy import deepcopy
    import torch.nn.functional as F
    import random

    class MLP(torch.nn.Module):
        """Launch MLP in C++

        Args:
            mlp_sizes (list of int): MLP sizes. Example: [1024,1024,1024] will create 2 MLP layers with shape 1024x1024
            bias (bool): Default True:
            relu (bool): Default True
        """

        def fast_gelu_1(x):
            # sqrt(2/pi) = 0.7978845608028654
            return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * torch.pow(x, 3.0))))

        def __init__(self, mlp_sizes, activation='gelu', dropout=0.25):
            super(MLP, self).__init__()
            self.num_layers = len(mlp_sizes) - 1
            self.mlp_sizes = copy(mlp_sizes)
            self.dropout = dropout

            if activation == 'relu':
                self.activation = 1
            elif activation == 'sigmoid':
                self.activation = 2
            elif activation == 'gelu':
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

        def forward(self, input, mask=None, ref=False, blaslt=False):

            if ref:
                return self.forward_ref(input, mask=mask)

            if not blaslt:
                return mlp_gelu_function(self.dropout, False, input, *self.weights, *self.biases)

            # print(input.type(), self.weights[0].type())
            return mlp_gelu_blaslt_function(input, self.weights[0], self.biases[0], self.weights[1], self.biases[1])

        def forward_ref(self, input, mask=None):

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
                    output = F.gelu(output)  * dropout_mask.view(output.size(0), -1) * pinv

                i += output.numel()

            return output

        def extra_repr(self):
            # TODO add dropout probability
            s = F"MLP sizes: {self.mlp_sizes},  activation={self.activation}"
            return s


    batch_size = 24568
    mlp_sizes = [1024, 4096, 1024]
    # mlp_sizes = [4, 7, 4]
    num_iters = 10


    class TestMLP(unittest.TestCase):

        def test_creation(self):
            MLP(mlp_sizes)

        def test_numeric(self):
            mlp = MLP(mlp_sizes, activation='gelu').cuda()

            print(mlp)
            ref_mlp = deepcopy(mlp)

            for _ in range(1):
                bsz = random.randint(2850, batch_size // 8) * 8
                test_input = torch.empty(bsz, mlp_sizes[0], device="cuda").uniform_(-1., 1.).requires_grad_()
                ref_input = test_input.clone().detach().requires_grad_()
                mlp_out = mlp(test_input)
                ref_out = ref_mlp.forward(ref_input, ref=True)

                # print(dropout_mask.sum() / dropout_mask.numel())
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
                mlp_out = mlp(test_input)
                ref_out = ref_mlp(ref_input, ref=True)
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
            mlp_out = mlp(test_input)

            ref_out = ref_mlp(ref_input, ref=True)
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
                    mlp_layers.append(nn.Dropout(0.25))

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

            # torch.cuda.synchronize()
            # start_time = time()
            # for _ in range(num_iters):
            #     mlp_out = mlp(test_input, blaslt=True)
            #     test_loss = mlp_out.mean()
            #     mlp.zero_grad()
            #     test_loss.backward()
            # torch.cuda.synchronize()
            # stop_time = time()
            # print(F"BLASLT MLP time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
            # torch.cuda.profiler.stop()


    unittest.main()
