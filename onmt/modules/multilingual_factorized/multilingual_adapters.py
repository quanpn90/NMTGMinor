# Implementation of the multilingual adapter as in Bapna et. al 2019

import torch
from torch.nn import Parameter
import torch.nn.functional as F
import math
from ..optimized.feed_forward import PositionWiseFeedForward
from ..layer_norm import LayerNorm


def xavier_normal(weight, gain=1.0):

    fan_in, fan_out = weight.size(-2), weight.size(-1)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    with torch.no_grad():
        weight.normal_(0, std)


class MultilingualAdapter(torch.nn.Module):

    def __init__(self, model_size, bottleneck_size, n_languages=1, dropout=0.0):
        
        super(MultilingualAdapter, self).__init__()

        self.all_modules = torch.nn.ModuleList()

        for i in range(n_languages):
            layer_norm = LayerNorm(model_size)
            feed_forward = PositionWiseFeedForward(model_size, bottleneck_size, dropout=dropout)
            adapter = torch.nn.Sequential(layer_norm, feed_forward)
            self.all_modules.append(adapter)

    def forward(self, input, lang=None):
        """
        :param input: TxBxN Tensor
        :param lang:  [1] Tensor
        :return:
        """

        assert lang.numel() == 1

        index = lang.item()

        adapter = self.all_modules[index]

        # normalize -> transform -> residual
        return input + adapter(input)


