import torch
import torch.nn.functional as F
from apex import amp


@amp.half_function
def swish(x):

    fast = True

    try:
        from swish_torch import SwishFunction
        x = SwishFunction.apply(x)

    except ModuleNotFoundError as e:
        x = x * F.sigmoid(x)

    return x


class FastSwish(torch.nn.Module):

    def __init__(self):
        super(FastSwish, self).__init__()

        self.fast = True

    def forward(self, x):

        if x.is_cuda:
            return swish(x)
        else:
            return x * F.sigmoid(x)
