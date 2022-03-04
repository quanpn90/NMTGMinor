import torch
from torch import Tensor

try:
    from torch.cuda.amp import custom_fwd, custom_bwd
except (ModuleNotFoundError, ImportError) as e:
    from .compat import custom_fwd, custom_bwd

try:
    import linear_blaslt
except (ModuleNotFoundError, ImportError) as e:
    linear_blaslt = None


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
    def linear_function(input, weight, bias):
        if bias is None:
            return torch.nn.functional.linear(input, weight, bias)
        else:
            _input, _weight, _bias = _cast_if_autocast_enabled(input, weight, bias)
        with torch.cuda.amp.autocast(enabled=False):
            return LinearFunction.apply(_input, _weight, _bias)
else:
    linear_function = torch.nn.functional.linear


class Linear(torch.nn.Linear):

    def forward(self, input: Tensor) -> Tensor:

        if linear_function is not None and self.bias is not None:
            return linear_function(input, self.weight, self.bias)
        else:
            return torch.nn.functional.linear(input, self.weight, self.bias)
