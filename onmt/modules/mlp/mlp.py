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


class MlpReluFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, p, *args):
        output = fused_mlp_relu.forward(p, args)
        ctx.save_for_backward(*args)
        ctx.outputs = output
        dropout_mask = output[-1]
        ctx.p = p
        return output[0]

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


class MlpSiluFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, p, *args):
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
        return (None, *grads)


if fused_mlp_silu:
    mlp_silu_function = half_function(MlpSiluFunction.apply)
else:
    mlp_silu_function = None


class MlpGELUFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, p, *args):
        output = fused_mlp_gelu.forward(p, args)
        ctx.save_for_backward(*args)
        ctx.outputs = output
        dropout_mask = output[-1]
        ctx.p = p
        return output[0]

    @staticmethod
    @custom_bwd
    def backward(ctx, *grad_o):
        p = ctx.p
        grads = fused_mlp_gelu.backward(p, grad_o[0], ctx.outputs, ctx.saved_tensors)
        del ctx.outputs
        return (None, *grads)


if fused_mlp_gelu:
    mlp_gelu_function = half_function(MlpGELUFunction.apply)
else:
    mlp_gelu_function = None


class MlpAGELUFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, p, *args):
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
        return (None, *grads)


if fused_mlp_gelu:
    mlp_agelu_function = half_function(MlpAGELUFunction.apply)
else:
    mlp_agelu_function = None
