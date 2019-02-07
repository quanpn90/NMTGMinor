import torch
from torch import nn, Tensor


def masked_function(function, inputs, mask=None):
    """
    Apply a function to the masked part of an input tensor.
    :param function: The function to apply.
    :param inputs: Input tensor, shape (N* x hidden)
    :param mask: Mask, shape (N*) or broadcastable, optional
    :return: The output of applying the function to only the masked part of the tensor,
    but in the original shape
    """
    if mask is None:
        return function(inputs)
    mask = mask.unsqueeze(-1)

    # remember the original shape
    original_shape = inputs.size()

    clean_input = inputs.masked_select(mask)

    # forward pass on the clean input only
    clean_output = function(clean_input)

    # after that, scatter the output (the position where we don't scatter are masked zeros anyways)
    output = inputs.new(*original_shape[:-1], clean_output.size(-1)).zero_()
    output.masked_scatter_(mask, clean_output)
    return output


class MaskedFunction(nn.Module):
    """
    Applies a function only to a masked part of an input tensor.

    Testing has determined that this implementation only gives a speedup for very sparse masks
    (at least 50% zeros). Otherwise, running the operation normally may be faster.
    """

    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, inputs: Tensor, mask=None):
        """
        Apply the function.
        :param inputs: Input tensor, shape (batch x time x hidden)
        :param mask: Mask, shape (batch x time), optional
        :return: The output of applying the function to only the masked part of the tensor,
        but in the original shape
        """
        return masked_function(self.function, inputs, mask)
