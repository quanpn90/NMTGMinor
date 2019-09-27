import math
import torch
import torch.nn as nn
from torch.autograd import Variable

"""
Class Bottle:
When working with masked tensors, bottles extract the "true" tensors 
using masks to avoid unnecessary computation
"""

class Bottle(nn.Module):
    
    def __init__(self, function):
        
        super(Bottle, self).__init__()
        self.function = function

    def forward(self, input, mask=None):
        """
        input: batch x time x hidden
        mask: batch x time
        """

        # revert to no-op for variational dropout
        return self.function(input)
        # # remember the original shape
        # original_shape = input.size()
        #
        # # flattned the tensor to 2D
        # flattened_input = input.contiguous().view(-1, input.size(-1))
        # flattened_size = flattened_input.size()
        #
        #
        # dim = original_shape[-1]
        #
        # if mask is not None:
        #     flattened_mask = mask.view(-1)
        #
        #     non_pad_indices = torch.nonzero(flattened_mask).squeeze(1)
        #
        #     clean_input = flattened_input.index_select(0, non_pad_indices )
        # else:
        #     clean_input = flattened_input
        #
        # # forward pass on the clean input only
        # clean_output = self.function(clean_input)
        #
        # if mask is not None:
        #     # after that, scatter the output (the position where we don't scatter are masked zeros anyways)
        #     flattened_output = Variable(flattened_input.data.new(*flattened_size[:-1], clean_output.size(-1)).zero_())
        #     flattened_output.index_copy_(0, non_pad_indices, clean_output)
        # else:
        #     flattened_output = clean_output
        #
        # # restore the tensor original size
        # output = flattened_output.view(*original_shape[:-1], flattened_output.size(-1))
        
        # return output
