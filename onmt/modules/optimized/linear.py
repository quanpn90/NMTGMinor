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
            d_input, d_weight, d_bias = linear_blaslt.backward(input, weight, grad_output, True)
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

    def __init__(self, *args, **kwargs):

        super(Linear, self).__init__(*args, **kwargs)


    def forward(self, input: Tensor) -> Tensor:

        if input.is_cuda and linear_function is not None and self.bias is not None:
            return linear_function(input, self.weight, self.bias)
        else:
            return torch.nn.functional.linear(input, self.weight, self.bias)


def factorize_linear(input, weight, bias, rm, sm):

    # here we assume that rm and sm has size [rank x D]

    if input.ndim == 3:

        # assuming input size is [T x B x D]

        bsz, qlen = input.size(1), input.size(0)
        rank = rm.size(0)

        rm = rm.unsqueeze(1).unsqueeze(2)
        sm = sm.unsqueeze(1).unsqueeze(2)

        h = input.unsqueeze(0) * sm

        if rank == 1:
            h = h.squeeze(0)
        else:
            h = h.sum(dim=0)

        h = torch.mm(h.view(qlen * bsz, -1), weight.transpose(0, 1))
        h = h.view(qlen, bsz, -1).unsqueeze(0) * rm

        if rank == 1:
            h = h.squeeze(0)
        else:
            h = h.sum(dim=0)

        h = h + bias.unsqueeze(0).unsqueeze(1)

        return h

    elif input.ndim == 2:

        total_bsz = input.size(0)
        rank = rm.size(0)

        rm = rm.unsqueeze(1)
        sm = sm.unsqueeze(1)

        h = input.unsqueeze(0) * sm

        if rank == 1:
            h = h.squeeze(0)
        else:
            h = h.sum(dim=0)

        h = torch.mm(h, weight.transpose(0, 1))
        h = h.unsqueeze(0) * rm

        if rank == 1:
            h = h.squeeze(0)
        else:
            h = h.sum(dim=0)

        h = h + bias.unsqueeze(0)

        return h

    else:
        raise NotImplementedError

