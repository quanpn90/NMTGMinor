"""MLP with convolutional gating (cgMLP) definition.

References:
    https://openreview.net/forum?id=RA-zVvZLYIy
    https://arxiv.org/abs/2105.08050

"""

import torch

from espnet2.nets.pytorch_backend.net_utils import get_activation
from espnet2.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class ConvolutionalSpatialGatingUnit(torch.nn.Module):
    """Convolutional Spatial Gating Unit (CSGU)."""

    def __init__(
        self,
        size: int,
        kernel_size: int,
        dropout_rate: float,
        use_linear_after_conv: bool,
        gate_activation: str,
        activation_ckpt: bool = False,
    ):
        super().__init__()

        n_channels = size // 2  # split input channels
        self.norm = LayerNorm(n_channels)
        self.conv = torch.nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size,
            1,
            (kernel_size - 1) // 2,
            groups=n_channels,
        )
        if use_linear_after_conv:
            self.linear = torch.nn.Linear(n_channels, n_channels)
        else:
            self.linear = None

        if gate_activation == "identity":
            self.act = torch.nn.Identity()
        else:
            self.act = get_activation(gate_activation)

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation_ckpt = activation_ckpt

    def espnet_initialization_fn(self):
        torch.nn.init.normal_(self.conv.weight, std=1e-6)
        torch.nn.init.ones_(self.conv.bias)
        if self.linear is not None:
            torch.nn.init.normal_(self.linear.weight, std=1e-6)
            torch.nn.init.ones_(self.linear.bias)

    def forward(self, x, gate_add=None):
        """Forward method

        Args:
            x (torch.Tensor): (N, T, D)
            gate_add (torch.Tensor): (N, T, D/2)

        Returns:
            out (torch.Tensor): (N, T, D/2)
        """

        x_r, x_g = x.chunk(2, dim=-1)

        if self.activation_ckpt:
            x_g = torch.utils.checkpoint.checkpoint(
                self.norm, x_g, use_reentrant=False
            )  # (N, T, D/2)
        else:
            x_g = self.norm(x_g)
        x_g = self.conv(x_g.transpose(1, 2)).transpose(1, 2)  # (N, T, D/2)
        if self.linear is not None:
            x_g = self.linear(x_g)

        if gate_add is not None:
            x_g = x_g + gate_add

        if self.activation_ckpt:
            x_g = torch.utils.checkpoint.checkpoint(self.act, x_g, use_reentrant=False)
        else:
            x_g = self.act(x_g)

        out = x_r * x_g  # (N, T, D/2)
        out = self.dropout(out)

        del x_g, x_r
        return out


class ConvolutionalGatingMLP(torch.nn.Module):
    """Convolutional Gating MLP (cgMLP)."""

    def __init__(
        self,
        size: int,
        linear_units: int,
        kernel_size: int,
        dropout_rate: float,
        use_linear_after_conv: bool,
        gate_activation: str,
        activation_ckpt: bool = False,
    ):
        super().__init__()

        self.channel_proj1 = torch.nn.Sequential(
            torch.nn.Linear(size, linear_units), torch.nn.GELU()
        )
        self.csgu = ConvolutionalSpatialGatingUnit(
            size=linear_units,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            use_linear_after_conv=use_linear_after_conv,
            gate_activation=gate_activation,
            activation_ckpt=activation_ckpt,
        )
        self.channel_proj2 = torch.nn.Linear(linear_units // 2, size)
        self.activation_ckpt = activation_ckpt

    def forward(self, x, mask):
        if isinstance(x, tuple):
            xs_pad, pos_emb = x
        else:
            xs_pad, pos_emb = x, None
        del x

        if self.activation_ckpt:
            xs_pad = torch.utils.checkpoint.checkpoint(
                self.channel_proj1, xs_pad, use_reentrant=False
            )
        else:
            xs_pad = self.channel_proj1(xs_pad)  # size -> linear_units
        xs_pad = self.csgu(xs_pad)  # linear_units -> linear_units/2
        xs_pad = self.channel_proj2(xs_pad)  # linear_units/2 -> size

        if pos_emb is not None:
            out = (xs_pad, pos_emb)
        else:
            out = xs_pad
        return out
