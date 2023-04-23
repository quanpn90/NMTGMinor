import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation_fn
from onmt.modules.layer_norm import layer_norm_func
from onmt.modules.optimized.linear import Linear as LinearModule


def Linear(in_features, out_features, bias=True):
    m = LinearModule(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class EfficientAdapter(nn.Module):
    def __init__(
            self,
            num_modules: int,
            input_size: int,
            bottleneck_size: int,
            activation_fn: str = "relu",
            static_layernorm: bool = False,
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

        # Fused MLP config
        self.fused = False
        self.fused_function = None
        if activation_fn == 'relu':
            from onmt.modules.mlp.mlp import mlp_relu_function
            if mlp_relu_function is not None:
                self.fused_function = mlp_relu_function
                self.fused = True
        elif activation_fn == 'gelu':
            from onmt.modules.mlp.mlp import mlp_gelu_function
            if mlp_gelu_function is not None:
                self.fused_function = mlp_gelu_function
                self.fused = True

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

        x = layer_norm_func(x,  ln_g, ln_b, (shortcut.size(-1),))

        if self.fused and x.is_cuda:
            dropout_p = 0.0

            weights = [down_w, up_w]
            biases = [down_b, up_b]

            x = self.fused_function(dropout_p, False, x, *weights, *biases)
        else:
            x = F.linear(x, down_w.view(-1, shortcut.size(-1)), down_b)
            x = self.activation(x)
            x = F.linear(x, up_w.view(shortcut.size(-1), -1), up_b)

        return x + shortcut
