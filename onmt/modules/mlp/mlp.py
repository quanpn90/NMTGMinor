
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


class MlpReluFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
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
    @custom_bwd
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
    mlp_relu_function = MlpReluFunction.apply
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
    mlp_silu_function = MlpSiluFunction.apply
else:
    mlp_silu_function = None


class MlpGELUFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, p, recompute, *args):
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
        return (None, None, *grads)


if fused_mlp_gelu:
    mlp_gelu_function = MlpGELUFunction.apply
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


if fused_mlp_gelu:
    mlp_agelu_function = MlpAGELUFunction.apply
else:
    mlp_agelu_function = None


class MlpGELUFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, p, recompute, *args):
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
        return (None, None, *grads)


if fused_mlp_gelu:
    mlp_gelu_function = MlpGELUFunction.apply
else:
    mlp_gelu_function = None


class MlpGeLUDropoutAddFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, p, r_p, *args):
        outputs = fused_mlp_gelu_dropout_add.forward(p, r_p, args)
        ctx.save_for_backward(*args)
        ctx.outputs = outputs
        dropout_mask = outputs[-2]
        residual_mask = outputs[-1]
        ctx.p = p
        ctx.r_p = r_p
        return outputs[0]

    @staticmethod
    @custom_bwd
    def backward(ctx, *grad_o):
        p = ctx.p
        r_p = ctx.r_p
        grads = fused_mlp_gelu_dropout_add.backward(p, r_p, grad_o[0], ctx.outputs, ctx.saved_tensors)
        del ctx.outputs
        return (None, None, *grads)


if fused_mlp_gelu_dropout_add:
    mlp_gelu_dropout_add_function = MlpGeLUDropoutAddFunction.apply
else:
    mlp_gelu_dropout_add_function = None