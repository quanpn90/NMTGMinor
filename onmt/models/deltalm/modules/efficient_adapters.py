import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.modules.layer_norm import LayerNorm
from onmt.modules.optimized.linear import Linear, linear_function

from .utils import get_activation_fn


def get_adapter_keys(lang_pairs, lang):
    if lang == "src":
        adapter_keys = [p.split("-")[0] for p in lang_pairs.split(",")]
    elif lang == "tgt":
        adapter_keys = [p.split("-")[1] for p in lang_pairs.split(",")]
    elif lang == "pair":
        adapter_keys = lang_pairs.split(",")
    else:
        raise ValueError
    # ensure consistent order!
    adapter_keys = sorted(list(set(adapter_keys)))
    return adapter_keys


class Adapter(nn.Module):
    def __init__(
            self,
            input_size: int,
            bottleneck_size: int,
            activation_fn: str,
            static_layernorm: bool,
    ):
        """
        Implements an Adapter layer following the architecture of
        Bapna and Firat 2019 - Simple, Scalable Adaptation for Neural Machine Translation
        https://aclanthology.org/D19-1165/
        Args:
            input_size (int): the dimensionality of the input feature vector
            bottleneck_size (int): the dimensionality of the bottleneck vector
            activation_fn (str): the activation function used after the down-projection
            static_layernorm (bool): use LayerNorm without trainable parameters
        """
        super().__init__()

        # reuse the transformer Linear layer to have consistent init with the rest of the model
        self.down = Linear(input_size, bottleneck_size)
        self.up = Linear(bottleneck_size, input_size)
        self.layer_norm = LayerNorm(input_size,
                                    elementwise_affine=not static_layernorm)
        self.activation_fn = get_activation_fn(activation_fn)

        for n, p in self.named_parameters():
            p.adapter = True
            p.label = n

    def forward(self, x: torch.Tensor):
        shortcut = x

        x = self.layer_norm(x)
        x = self.down(x)
        x = self.activation_fn(x)
        x = self.up(x)

        return x + shortcut

class EfficientAdapter(nn.Module):
    def __init__(
            self,
            num_modules: int,
            input_size: int,
            bottleneck_size: int,
            activation_fn: str,
            static_layernorm: bool,
    ):
        """
        Implements an Adapter layer following the architecture of
        Bapna and Firat 2019 - Simple, Scalable Adaptation for Neural Machine Translation
        https://aclanthology.org/D19-1165/
        This particular implementation uses a shared up/down projection matrix
        for all adapters and in the forward-pass it simply indexes the
        corresponding adapter (row). This is a workaround for an efficiency
        bug that occurs in distributed training.
        Args:
            input_size (int): the dimensionality of the input feature vector
            bottleneck_size (int): the dimensionality of the bottleneck vector
            activation_fn (str): the activation function used after the down-projection
            static_layernorm (bool): use LayerNorm without trainable parameters
        """
        super().__init__()

        # reuse the transformer Linear layer to have consistent init with the rest of the model
        self.num_modules = num_modules
        self.static_layernorm = static_layernorm
        self.down_weight = Linear(bottleneck_size * input_size, num_modules, bias=False)
        self.down_bias = Linear(bottleneck_size, num_modules, bias=False)
        self.up_weight = Linear(bottleneck_size * input_size, num_modules, bias=False)
        self.up_bias = Linear(input_size, num_modules, bias=False)
        if not self.static_layernorm:
            self.layer_norm_gammas = Linear(input_size, num_modules)
            self.layer_norm_betas = Linear(input_size, num_modules)

        self.activation = get_activation_fn(activation_fn)

        # ensure normal initialization
        # initialize the parameters of each "adapter" row similar to nn.Linear()
        with torch.no_grad():
            for i in range(num_modules):
                self.down_weight.weight[i] = Linear(input_size, bottleneck_size).weight.view(-1)
                self.up_weight.weight[i] = Linear(bottleneck_size, input_size).weight.view(-1)
                self.down_bias.weight[i].fill_(0)
                self.up_weight.weight[i].fill_(0)

                if not self.static_layernorm:
                    self.layer_norm_gammas.weight[i].fill_(1)
                    self.layer_norm_betas.weight[i].fill_(0)

        for n, p in self.named_parameters():
            p.adapter = True
            p.label = n

    def forward(self, x: torch.Tensor, index: int):
        shortcut = x

        down_w = self.down_weight.weight[index]
        up_w = self.up_weight.weight[index]
        down_b = self.down_bias.weight[index]
        up_b = self.up_bias.weight[index]
        ln_g = None
        ln_b = None
        if not self.static_layernorm:
            # ensure ln_g will have mean of 1, instead of 0
            ln_g = self.layer_norm_gammas.weight[index]
            ln_b = self.layer_norm_betas.weight[index]

        x = F.layer_norm(x, (shortcut.size(-1),), ln_g, ln_b)
        x = linear_function(x, down_w.view(-1, shortcut.size(-1)), down_b)
        x = self.activation(x)
        x = linear_function(x, up_w.view(shortcut.size(-1), -1), up_b)

        return x + shortcut