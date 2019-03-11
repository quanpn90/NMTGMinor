import torch
from torch import nn, Tensor


def masked_function(function, *inputs, mask=None, restore_shape=True):
    """
    Apply a function to the masked part of an input tensor.
    :param function: The function to apply.
    :param inputs: Input tensor, shape (N* x hidden)
    :param mask: Mask, shape (N*) or broadcastable, optional
    :return: The output of applying the function to only the masked part of the tensor,
    but in the original shape
    """
    if mask is None:
        return function(*inputs)
    valid_indices = torch.nonzero(mask.view(-1)).squeeze(1)

    # remember the original shape
    original_shape = inputs[0].size()
    num_items = torch.prod(original_shape[:-1])

    clean_inputs = []
    for inp in inputs:
        flat_input = inp.view(-1, original_shape[-1])
        clean_inputs.append(flat_input.index_select(0, valid_indices))

    # forward pass on the clean input only
    clean_output = function(*clean_inputs)

    if not restore_shape:
        return clean_output

    # after that, scatter the output (the position where we don't scatter are masked zeros anyways)
    flat_output = inputs[0].new_zeros(num_items, clean_output.size(-1))
    flat_output.index_copy_(0, valid_indices, clean_output)

    output = flat_output.view(*original_shape[:-1], clean_output.size(-1))
    return output


class MaskedFunction(nn.Module):
    """
    Applies a function only to a masked part of an input tensor.

    Testing has determined that this implementation only gives a speedup for very sparse masks
    (at least 50% zeros). Otherwise, running the operation normally may be faster.
    """

    def __init__(self, function, restore_shape=True):
        super().__init__()
        self.function = function
        self.restore_shape = restore_shape

    def forward(self, inputs: Tensor, mask=None):
        """
        Apply the function.
        :param inputs: Input tensor, shape (batch x time x hidden)
        :param mask: Mask, shape (batch x time), optional
        :return: The output of applying the function to only the masked part of the tensor,
        but in the original shape
        """
        return masked_function(self.function, inputs, mask=mask, restore_shape=self.restore_shape)
