import math
import torch
import numbers
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F
import importlib

try:
    from torch.cuda.amp import custom_fwd, custom_bwd
except (ModuleNotFoundError, ImportError) as e:
    from .optimized.compat import custom_fwd, custom_bwd


def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        try:
            return torch.cuda.amp.autocast_mode._cast(args, torch.get_autocast_gpu_dtype())
        except AttributeError:
            return torch.cuda.amp.autocast_mode._cast(args, torch.half)


"""
Faster version of Layer Norm from apex (new)
"""

try:
    import fast_layer_norm_cuda
    # print("[INFO] Fast layer norm implementation detected.")
except (ModuleNotFoundError, ImportError) as e:
    fast_layer_norm_cuda = None
    # print("[INFO] Fast layer norm implementation not found.")


class FastLayerNormFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, epsilon):
        x = x.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()
        hidden_size = gamma.numel()
        y, mu, rsigma = fast_layer_norm_cuda.ln_fwd(x, gamma, beta, epsilon)
        ctx.save_for_backward(x, gamma, mu, rsigma)
        ctx.need_weight_grad = gamma.requires_grad
        return y

    @staticmethod
    def backward(ctx, dy):
        # assert dy.is_contiguous()
        dy = dy.contiguous()  # this happens!
        x, gamma, mu, rsigma = ctx.saved_tensors

        dx, dgamma, dbeta, _, _ = fast_layer_norm_cuda.ln_bwd(dy, x, mu, rsigma, gamma)

        # TODO: write bwd function that doesn't need backward
        if not ctx.need_weight_grad:
            dgamma = None
            dbeta = None

        return dx, dgamma, dbeta, None


"""
Fast version of Layer Norm from Apex
"""


def fast_layer_norm_affine(input, weight, bias, normalized_shape, eps=1e-5):
    args = _cast_if_autocast_enabled(input, weight, bias, eps)
    with torch.cuda.amp.autocast(enabled=False):
        return FastLayerNormFN.apply(*args)


class FP32LayerNorm(torch.nn.Module):

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):

        eps = self.eps

        return F.layer_norm(
            input.float(), self.normalized_shape, self.weight, self.bias, eps).type_as(input)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class LayerNorm(torch.nn.Module):
    """
    See LayerNorm for details.

    Note, however, that unlike LayerNorm this norm includes a batch component.
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, fast=True):

        eps = self.eps

        if fast and fast_layer_norm_cuda is not None and input.size(-1) in [768, 1024, 2048, 3072, 4096]:
            return fast_layer_norm_affine(input, self.weight, self.bias, self.normalized_shape, eps)

        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class MultilingualLayerNorm(torch.nn.Module):
    """
    See LayerNorm for details.

    Note, however, that unlike LayerNorm this norm includes a batch component.
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, n_languages=1):
        super().__init__()
        self.n_languages = n_languages

        self.fused = False

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(self.n_languages, *self.normalized_shape))
            self.bias = Parameter(torch.Tensor(self.n_languages, *self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, factor):

        eps = self.eps

        if self.elementwise_affine:
            weight = torch.index_select(self.weight, 0, factor).squeeze(0)
            bias = torch.index_select(self.bias, 0, factor).squeeze(0)
        else:
            weight, bias = None, None

        if not input.is_cuda or not fast_fused:
            return F.layer_norm(
                input, self.normalized_shape, weight, bias, eps)
        if self.elementwise_affine:
            if fast_fused and input.is_cuda:
                return fast_layer_norm_affine(input, weight, bias, self.normalized_shape, eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
