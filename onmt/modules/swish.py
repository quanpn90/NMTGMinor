import torch
import torch.nn as nn

try:
    import apex.amp as amp
    from apex.amp import half_function
except (ModuleNotFoundError, ImportError) as e:
    amp = None
    from .optimized.compat import half_function

try:
    from torch.cuda.amp import custom_fwd, custom_bwd
except (ModuleNotFoundError, ImportError) as e:
    from .optimized.compat import custom_fwd, custom_bwd

try:
    import silu_cuda
except (ModuleNotFoundError, ImportError) as e:
    silu_cuda = None


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


@half_function
def fast_silu(input):
    return SwishFunction.apply(input)


class SiLU(nn.Module):

    def __init__(self, inplace=False):

        super(SiLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):

        # maybe only use during training to avoid kernel problem?
        if silu_cuda is not None and input.is_cuda:
            return fast_silu(input)
        else:
            try:
                output = torch.nn.functional.silu(input, inplace=self.inplace)
            except AttributeError:
                output = input * torch.sigmoid(input)

            return output
