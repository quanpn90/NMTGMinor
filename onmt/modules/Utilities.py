import torch
import torch.nn as nn
import torch.nn.functional as F



def mean_with_mask(context, mask):

    # context dimensions: T x B x H
    # mask dimension: T x B x 1 (with unsqueeze)
    # first, we have to mask the context with zeros at the unwanted position

    eps = 0

    context.masked_fill_(mask, eps)
    # then take the sum over the time dimension
    context_sum = torch.sum(context, dim=0, keepdim=False)

    nonzeros = 1 - mask.type_as(context_sum)
    weights = torch.sum(nonzeros, dim=0, keepdim=False)

    mean = context_sum.div_(weights)

    return mean

def mean_with_mask_backpropable(context, mask):

    # context dimensions: T x B x H
    # mask dimension: T x B x 1 (with unsqueeze)
    # first, we have to mask the context with zeros at the unwanted position

    eps = 0

    nonzeros = (1 - mask).type_as(context)

    context = context * nonzeros
    # then take the sum over the time dimension
    context_sum = torch.sum(context, dim=0, keepdim=False)
    
    weights = torch.sum(nonzeros, dim=0, keepdim=False)

    mean = context_sum.div_(weights)

    return mean


def max_with_mask(context, mask):


	# context dimensions: T x B x H
    # mask dimension: T x B x 1 (with unsqueeze)
    # first, we have to mask the context with zeros at the unwanted position

    eps = -float('inf')

    context.masked_fill_(mask, eps)
    
    max_, indices = torch.max(context, dim=0, keepdim=False)

    return max_