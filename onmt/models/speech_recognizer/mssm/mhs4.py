#!/usr/bin/env python3

from typing import Optional, List, Tuple, Union

import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

# import pykeops
# import pykeops.torch
# from pykeops.torch import LazyTensor

from einops import rearrange, repeat
from opt_einsum import contract
from torch.cuda.amp import autocast

try:
    from .ssm_kernel.ssm_kernel_coefficient import compute_kernel_coefficient
except ImportError:
    from ssm_kernel.ssm_kernel_coefficient import compute_kernel_coefficient


@torch.no_grad()
def bilinear_discretization(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor, T: torch.Tensor
):
    """
    Performs a bilinear transformation of the (diagonal + lowrank) transition A and input matrix B.
    For a given tensor of N different time steps, this function will return N approximations to A and B.

    Parameters:
        A: shape (Q, N, N)
        B: shape (Q, N)
        C: shape (Q, C, H, N)
        D: shape (Q, C, H)
        T: shape (Q, H)

    Returns:
        dA: shape (Q, H, N, N)
        dB: shape (Q, H, N)
        dC: shape (Q, C, H, N)
        dD: shape (Q, C, H)
    """

    # Factor term reused for A and B
    factor = 0.50 * contract("qh,qnm->qhnm", T, A)

    # Get identity (1, N, N)
    identity = torch.eye(A.size(-1)).to(A).unsqueeze(0).unsqueeze(0)

    # Correction term
    correction = torch.linalg.inv(identity - factor)

    # Get bilinear A and B
    dA = contract("qhnm,qhmk->qhnk", correction, identity + factor)
    dB = contract("qhnm,qh,qm->qhn", correction, T, B)
    return dA, dB, C, D


def get_activation(act: str = "gelu"):
    if act == "relu":
        return nn.ReLU()
    if act == "gelu":
        return nn.GELU()
    if act == "swish":
        return nn.SiLU()
    if act == "glu":
        return nn.GLU()
    return nn.Identity()


def gen_noisy_linear_weights(parameter_noise, weight):
    """Get Gaussian noisy linear weights based on given noise level ....
    and the weights themselves. The noise are normalized per channel (dim=1).

    InputArgs:
        parameter_noise: float, noise level, [0.0, 1.0]
        weight: Tensor, a weight tensor of a matrix
    Return:
        noisy_weight: Tensor, same dimension as weight, but with noise added.
    """
    noise = torch.randn_like(weight).to(device=weight.device)
    normalized_noise = noise / torch.norm(noise, dim=1, keepdim=True)

    w_norm = torch.norm(weight, dim=1, keepdim=True).detach()
    scale = parameter_noise * w_norm
    noisy_weight = weight + scale * normalized_noise
    return noisy_weight


class Linear(torch.nn.Linear):
    def __init__(
        self,
        input_dim,
        output_dim,
        bias=True,
        parameter_noise: float = 0.0,
        device=None,
        dtype=None,
    ):
        super(Linear, self).__init__(
            in_features=input_dim,
            out_features=output_dim,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        # mirror torch.nn.linear to set device and detype
        self.parameter_noise = parameter_noise

    def get_noisy_weight(self, weight):
        if self.parameter_noise > 0.0 and self.training:
            return gen_noisy_linear_weights(self.parameter_noise, weight)
        return weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.get_noisy_weight(self.weight), self.bias)


class TiedStateSpaceModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 1,
        channels: int = 1,
        scale: float = 0.50,
        maxlen: int = 256,
        timestep_min: float = 0.010,
        timestep_max: float = 0.160,
        parameter_noise: float = 0.00,
        use_fast_kernel: bool = True,
        create_on_gpu=True
    ):
        super().__init__()

        # Store inputs
        self.input_dim = self.H = input_dim    # input dimensions
        self.hidden_dim = self.N = hidden_dim  # N = number of SSM copies?
        self.num_heads = self.Q = num_heads    # Q = number of heads
        self.channels = self.C = channels
        self.parameter_noise = parameter_noise

        # Create diagonal transition matrix
        self.diagonal = nn.Parameter(
            math.log(scale) + torch.randn(num_heads, hidden_dim)
        )

        if create_on_gpu:
            self.diagonal.data = self.diagonal.data.cuda()
            # print(self.diagonal.device)

        device = self.diagonal.device

        # Create lowrank correction

        self.lowrank = nn.Parameter(torch.randn(num_heads, hidden_dim)).to(device)

        # Create discretization step per channel
        self.timestep = nn.Parameter(
            torch.rand(num_heads, input_dim)
            * (math.log(timestep_max) - math.log(timestep_min))
            + math.log(timestep_min)
        ).to(device)

        # Initialise remaining parameters
        self.register("input_matrix", (num_heads, hidden_dim), dim=1, device=device)
        self.register(
            "output_matrix",
            (num_heads, self.channels, input_dim, hidden_dim),
            dim=hidden_dim,
            device=device
        )
        self.register("skip_matrix", (num_heads, channels, input_dim), dim=1, device=device)

        # Register omega parameter
        self.setup(maxlen, dtype=torch.cfloat, device=self.diagonal.device)

        self.use_fast_kernel = use_fast_kernel

    def register(self, name, size, dim, lr=None, device=None):
        # Random uniform initialization
        weight = torch.rand(*size).to(device)
        weight = (2 * weight - 1) / math.sqrt(dim)

        # Register trainable parameter
        self.register_parameter(name, nn.Parameter(weight))

        # Add learning rate
        optim = {}
        if lr is not None:
            optim["lr"] = lr
        if len(optim) > 0:
            setattr(getattr(self, name), "_optim", optim)  # noqa

    @torch.no_grad()
    def get_correction_factor(self, double=False):

        # Get the parameters which are transformed (do not use noisy training on these params)
        d = self.get_diagonal()  # (Q, N)
        p = self.get_lowrank()  # (Q, N)
        t = self.get_timestep()  # (Q, H)

        identity = torch.eye(self.hidden_dim).to(d).unsqueeze(0).unsqueeze(0)

        # Get continous matrix (H, N, N)
        A = 0.50 * contract("qh,qnm->qhnm", t, self.get_transition(d, p))

        # Get discretized A naively
        # print("Solving dA = solve(identity - A, identity + A)", A.size(), A.type(), A.device)
        dA = torch.linalg.solve(identity - A, identity + A)

        # Correction factor
        # Get identity (1, N, N)
        if double:
            return identity + torch.matrix_power(dA, self.maxlen)
        return identity - torch.matrix_power(dA, self.maxlen)

    @torch.no_grad()
    def setup(self, maxlen, dtype, device, double=False):
        """
        Calculate (and cache) FFT nodes and their "unprocessed" them with the bilinear transform
        This should be called everytime the internal length changes
        """

        # Update internal length
        self.maxlen = maxlen

        # Get the correction matrix (H, N, N)
        correction = self.get_correction_factor(double)

        # Now correct for the length by modifying the output matrix using every input channel
        # Do not call the get_output_matrix to avoid noise injection
        weight = self.output_matrix.data
        weight = contract("qchn,qhnk->qchk", weight, correction).contiguous()
        self.output_matrix.data = weight

        # Double length if a sequence has been encountered with longer than supported length
        if double:
            self.maxlen *= 2

        self.setup_omega_z(dtype, device)

    @torch.no_grad()
    def setup_omega_z(self, dtype, device):

        self.L = self.maxlen
        # Create array on the unit circle
        omega = torch.tensor(
            np.exp(-2j * np.pi / self.maxlen), dtype=dtype, device=device
        )
        omega = omega ** torch.arange(self.maxlen, device=device)

        # Create the bilinear transformation
        z = 2 * (1 - omega) / (1 + omega)

        # Store these for faster computation
        self.register_buffer("omega", torch.view_as_real(omega))

        # define self.z here
        self.register_buffer("z", torch.view_as_real(z))

    @torch.no_grad()
    def setup_linear(self):
        """
        This computes the factors necessary to run the recurrent form efficiently.
        """

        # Update the output matrix for correction
        correction = self.get_correction_factor()
        correction = torch.linalg.inv(correction)

        # Now correct for the length by modifying the output matrix using every head
        # Do not call the get_output_matrix to avoid noise injection
        weight = self.output_matrix.data  # (..., HN) -> (H, ..., N)
        weight = contract("qchn,qhnk->qchk", weight, correction).contiguous()
        self.output_matrix.data = weight

        # Get all quantities
        d = self.get_diagonal()  # (Q, N)
        p = self.get_lowrank()  # (Q, N)
        t = self.get_timestep()  # (Q, H)

        # For the A0 matrix
        d0 = 2 / t.unsqueeze(-1) + d.unsqueeze(-2)
        f0 = repeat(p, "q n -> q h n", h=self.input_dim)
        s0 = 1.0

        # For the A1 matrix
        d1 = 1 / (2 / t.unsqueeze(-1) - d.unsqueeze(-2))
        f1 = d1 * p.unsqueeze(-2)
        s1 = 1 / (1 + contract("qhn,qhn,qhn->qh", f0, d1, f0)).unsqueeze(-1)

        # Compute the discretized states
        dA, dB, dC, dD = bilinear_discretization(
            self.get_transition(),
            self.input_matrix,
            self.output_matrix,
            self.skip_matrix,
            self.get_timestep(),
        )

        self.linear_params = {
            "d0": d0,  # (Q, H, N)
            "d1": d1,  # (Q, H, N)
            "f0": f0,  # (Q, H, N)
            "f1": f1,  # (Q, H, N)
            "s0": s0,  # (1)
            "s1": s1,  # (Q, H, 1)
            "dA": dA,  # (Q, H, N, N)
            "dB": dB,  # (Q, H, N)
            "dC": dC,  # (Q, C, H, N)
            "dD": dD,  # (Q, C, H)
        }

    def get_noisy_weight(self, weight):
        if self.parameter_noise > 0.0 and self.training:
            return gen_noisy_linear_weights(self.parameter_noise, weight)
        return weight

    def get_diagonal(self):
        return -torch.exp(self.diagonal)

    def get_lowrank(self):
        return self.lowrank

    def get_transition(self, d=None, p=None):
        d = d if d is not None else self.get_diagonal()
        p = p if p is not None else self.get_lowrank()
        return torch.diag_embed(d) - contract("qm,qn->qmn", p, p)

    def get_timestep(self):
        return torch.exp(self.timestep)

    def get_input_matrix(self):
        return self.get_noisy_weight(self.input_matrix)  # (Q, H)

    def get_output_matrix(self):
        return self.get_noisy_weight(self.output_matrix)  # (Q, C, H, N)

    def get_skip_matrix(self):
        return self.get_noisy_weight(self.skip_matrix)  # (Q, C, H)

    def get_dwoodbury(self, z, d, invt):
        # Get the bilinear transformation
        z = contract("l,qh->qlh", torch.view_as_complex(z), invt)

        # Compute the term and reuse computations (Q, L, H, N)
        return 1 / (z.unsqueeze(-1) - d.unsqueeze(-2).unsqueeze(-2))

    def compute_slow(self, z, d, t, b, c):
        # Get the diagonal component in the woodbury computation
        # which will be reused in computing the kernel

        # z is forced to be fp32
        # the following prevents fp16 underflow, particularly on t
        if t.dtype == torch.float16:
            t = t.to(z.dtype)
            b = b.to(z.dtype)
            c = c.to(z.dtype)
            d = d.to(z.dtype)

        # Get the memory heavy denominator
        r = self.get_dwoodbury(z, d, 1 / t)  # (Q, L, H, N)

        # Compute kernel coeffs
        kernelcc = contract("qihn,qlhn,qchn->qiclh", b.to(r.dtype), r, c)
        return kernelcc

    def get_kernel(self):

        # Get the parameters which are transformed
        d = self.get_diagonal()  # (Q, N)
        t = self.get_timestep()  # (Q, H)

        # Get the lowrank contribution and input matrix
        p = self.get_lowrank()  # (Q, N)
        b = self.get_input_matrix()  # (Q, H)
        c = self.get_output_matrix()  # (Q, C, H, N)

        # Since we have tied states
        b = repeat(b, "q n -> q 1 h n", h=self.input_dim)  # (Q, 1, H, N)
        p = repeat(p, "q n -> q 1 h n", h=self.input_dim)  # (Q, 1, H, N)

        # For batched operations
        b = torch.cat([b, p], dim=1)  # (Q, 2, H, N)
        c = torch.cat([c, p], dim=1)  # (Q, C + 1, H, N)

        # Get the diagonal component in the woodbury computation
        # which will be reused in computing the kernel
        # r = self.get_dwoodbury(d, 1 / t)  # (Q, L, H, N)

        # Compute kernel coeffs
        # kernelcc = contract("qihn,qlhn,qchn->qiclh", b.to(r.dtype), r, c)

        # Compute kernel coeffs
        # kernelcc = self.compute_slow(self.z, d, t, b, c)
        # print(self.z.type(), d.type(), t.type(), b.type(), c.type())
        kernelcc = compute_kernel_coefficient(self.z, d, t, b, c, fast=self.use_fast_kernel)

        # Compute kernel assuming low rank of 1  (Q, 2, C, L, H) -> (Q, 1, C, L, H)
        unit = 2 / (1 + torch.view_as_complex(self.omega))
        kernel = kernelcc[:, :-1, :-1] - kernelcc[:, -1:, :-1] * kernelcc[
            :, :-1, -1:
        ] / (1 + kernelcc[:, -1:, -1:])
        kernel = kernel.squeeze(1)  # (Q, C, L, H)
        kernel = contract("l,qclh->lqch", unit, kernel)
        kernel = torch.fft.irfft(kernel, n=kernel.size(0), dim=0)
        return kernel.float()

    """
    def get_kernel_lazy(self):

        # Get the parameters which are transformed
        d = self.get_diagonal()  # (Q, N)
        t = self.get_timestep()  # (Q, H)

        # Get the input and output matrix
        b = self.get_input_matrix()  # (Q, N)
        c = self.get_output_matrix()  # (Q, C, H, N)

        # Force values to be fp32
        if t.dtype == torch.float16:
            t = t.to(self.z.dtype)
            b = b.to(self.z.dtype)
            c = c.to(self.z.dtype)
            d = d.to(self.z.dtype)

        # Map to lazy vectors for memory efficient computation
        d = LazyTensor(d.view(self.Q, 1, self.N, 1, 1))
        t = LazyTensor(t.view(self.Q, 1, 1, 1, self.H))
        b = LazyTensor(b.view(self.Q, 1, self.N, 1, 1))
        c = LazyTensor(
            c.view(self.Q, self.C, self.H, 1, self.N).transpose(2, 4).contiguous()
        )

        # Complex Lazy Tensors
        z = torch.view_as_complex(self.z)
        z = LazyTensor(z.view(1, 1, 1, self.L, 1))
        o = 2 / (1 + torch.view_as_complex(self.omega))
        o = LazyTensor(o.view(1, 1, 1, self.L, 1))

        # Compute the kernel (Q, C, N, L, H)
        kernel = o * b * c / (z / t - d)
        kernel = kernel.sum(dim=2)
        kernel = torch.fft.irfft(kernel, n=kernel.size(-2), dim=-2)
        return kernel.permute(2, 0, 1, 3).contiguous().float()
    """

    # do we need masking?
    def forward(self, u: torch.Tensor):

        # Get sequence length (L, B, Q, H)
        length = u.size(0)

        # Double length if needed
        while length > self.maxlen:
            self.setup(
                self.maxlen,
                dtype=torch.cfloat,
                device=self.diagonal.device,
                double=True,
            )

        # print(self.z.dtype)

        # This would be call only once at the beginning of fp16 training
        if self.z.dtype == torch.float16:
            self.setup_omega_z(dtype=torch.cfloat, device=self.diagonal.device)

        # For FP16 conversion
        fp16 = u.dtype == torch.float16

        # Perform state space modelling (L, Q, C, H)
        k = self.get_kernel()[:length]  # get kernel always in fp32?
        # print("kernel type", k.type())
        # k = self.get_kernel_lazy()[:length]
        # Now compute the fourier transform
        # breakpoint()
        # k = k.type_as(u)
        k_f = torch.fft.rfft(k.float(), n=2 * length, dim=0)
        uu = u.to(torch.float32) if fp16 else u
        u_f = torch.fft.rfft(uu, n=2 * length, dim=0)
        x_f = contract("lqch,lbqh->lbqch", k_f, u_f)
        # print("fourier dtype", k_f.type(), u_f.type())

        # Get the output without transformation or skip connection
        x = torch.fft.irfft(x_f, n=2 * length, dim=0)[:length]
        x = x.to(torch.float16) if fp16 else x

        # Get the full output
        return x + contract("qch,lbqh->lbqch", self.get_skip_matrix(), u)


class MHS4(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        projection_dim: Optional[int] = None,
        hidden_dim: int = 64,
        num_heads: int = 1,
        activation: Optional[str] = "gelu",
        channels: int = 1,
        rank: int = 1,
        scale: float = 0.50,
        maxlen: int = 256,
        timestep_min: float = 0.010,
        timestep_max: float = 0.160,
        dropout: float = 0.00,
        use_final_linear: bool = True,
        parameter_noise: float = 0.00,
        use_fast_kernel: bool = True,
        create_on_gpu: bool = True
    ):

        super().__init__()
        
        # Only a rank of 1 is supported
        assert rank == 1

        # Store inputs
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.projection_dim = projection_dim or input_dim // num_heads
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.channels = channels
        self.parameter_noise = parameter_noise

        # GLU activation requires double the channels
        glu = activation == "glu"

        # Increase number of channels for glu
        self.channels *= 2 if glu else 1

        # Input is divisible by number of heads
        assert self.input_dim % self.num_heads == 0

        # Projection layer
        self.projweight, self.projbias = (
            self.init_linear(
                sizew=(self.num_heads, self.projection_dim, input_dim),
                sizeb=(self.num_heads, self.projection_dim),
            )
            if self.num_heads > 1
            else (None, None)
        )

        # SSM Layer
        self.ssm = TiedStateSpaceModel(
            input_dim=self.projection_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            channels=self.channels,
            scale=scale,
            maxlen=maxlen,
            timestep_min=timestep_min,
            timestep_max=timestep_max,
            parameter_noise=parameter_noise,
            use_fast_kernel=use_fast_kernel,
            create_on_gpu=create_on_gpu
        )

        # Dropout and activation following ssm
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Final linear layer weight
        self.out = (
            Linear(
                input_dim=self.projection_dim * self.num_heads,
                output_dim=self.output_dim,
                parameter_noise=parameter_noise,
            )
            if use_final_linear
            else nn.Identity()
        )

    def init_linear(self, sizew, sizeb):
        # Weight matrix
        weight = nn.Parameter(torch.empty(sizew))
        init.kaiming_uniform_(weight, a=math.sqrt(5))

        # Bias vector
        bias = nn.Parameter(torch.empty(sizeb))
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(bias, -bound, bound)
        return weight, bias

    def get_noisy_weight(self, weight):
        if self.parameter_noise > 0.0:
            return gen_noisy_linear_weights(self.parameter_noise, weight)
        return weight

    @torch.no_grad()
    def setup(self, maxlen, dtype, device, double=False):
        self.ssm.setup(maxlen, dtype, device, double=double)

    @torch.no_grad()
    def setup_linear(self):
        self.ssm.setup_linear()

    def projection_linear(self, x):
        # Input of shape (L, B, H) -> (L, B, Q, H)
        if self.projweight is None:
            return x.unsqueeze(-2)

        # Noisy training
        projweight = self.get_noisy_weight(self.projweight)
        projbias = self.get_noisy_weight(self.projbias)

        l, b, n = x.size(0), x.size(1), x.size(2)
        q, k = projweight.size(0), projweight.size(1)

        # this op is cast to fp16
        out1 = torch.mm(x.view(l * b, n), projweight.view(q * k, n).transpose(0, 1).contiguous())
        # this op always outputs float32
        out = out1.view(l, b, q, k).add_(projbias.type_as(out1))

        return out
        # return contract("qkn,lbn->lbqk", projweight, x) + projbias

    def forward(self, u: torch.Tensor):
        # Assumes the input is of shape (L, B, H)

        u = self.projection_linear(u)

        u = self.ssm(u)
        u = rearrange(u, "l b q c h -> l b (q c h)")
        u = self.dropout(self.activation(u))
        u = self.out(u)
        return u


def build_stacked_mh_s4(
    num_layers: int = 1,
    only_activate_last: bool = False,
    input_dim: int = 512,
    intermediate_dim: int = 512,
    output_dim: Optional[int] = None,
    hidden_dim: int = 32,
    num_heads: int = 1,
    activation: str = "gelu",
    channels: int = 1,
    rank: int = 1,
    scale: float = 0.50,
    maxlen: int = 256,
    timestep_min: float = 0.010,
    timestep_max: float = 0.160,
    dropout: float = 0.10,
    remove_final_linear: bool = False,
    parameter_noise: float = 0.00,
    use_fast_kernel: bool = True,
    create_on_gpu = True
):

    # Build all layers sequentially
    layers = []

    # Decide on output dimension
    output_dim = output_dim or input_dim

    # Starting first layer build with activation if single layer or activated when stacked
    use_activation = num_layers == 1 or not only_activate_last

    # Do not use final linear layer if we have multiple heads in stacked mode since there's a following projection
    # This is also to reduce the number of parameters
    use_final_linear = (num_heads == 1) or (num_layers == 1)

    layers.append(
        MHS4(
            input_dim=input_dim,
            output_dim=intermediate_dim if num_layers > 1 else output_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            activation=activation if use_activation else None,
            channels=channels,
            rank=rank,
            scale=scale,
            maxlen=maxlen,
            timestep_min=timestep_min,
            timestep_max=timestep_max,
            dropout=dropout,
            use_final_linear=use_final_linear,
            parameter_noise=parameter_noise,
            use_fast_kernel=use_fast_kernel,
            create_on_gpu=create_on_gpu
        )
    )

    # Intermediate layers
    # Ensure each head dimension is consistent
    assert intermediate_dim % num_heads == 0

    for i in range(num_layers - 2):
        layers.append(
            MHS4(
                input_dim=input_dim
                if (not use_final_linear and i == 0)
                else intermediate_dim,
                output_dim=intermediate_dim,
                projection_dim=intermediate_dim // num_heads,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                activation=activation if use_activation else None,
                channels=channels,
                rank=rank,
                scale=scale,
                maxlen=maxlen,
                timestep_min=timestep_min,
                timestep_max=timestep_max,
                dropout=dropout,
                use_final_linear=use_final_linear,
                parameter_noise=parameter_noise,
                use_fast_kernel=use_fast_kernel,
            )
        )

    # Final layer, requires larger projection layers for higher intermediate projections
    # Ensure that the output is divisible
    assert output_dim % num_heads == 0

    if num_layers > 1:
        layers.append(
            MHS4(
                input_dim=input_dim
                if (not use_final_linear and num_layers == 2)
                else intermediate_dim,
                output_dim=output_dim,
                projection_dim=intermediate_dim // num_heads,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                activation=activation,
                channels=channels,
                rank=rank,
                scale=scale,
                maxlen=maxlen,
                timestep_min=timestep_min,
                timestep_max=timestep_max,
                dropout=dropout,
                use_final_linear=True,
                parameter_noise=parameter_noise,
                use_fast_kernel=use_fast_kernel,
            )
        )

    # Get the final layer and remove its linear layer if needed
    if remove_final_linear:
        assert (
            intermediate_dim == input_dim
        ), "Removing the final linear layer is only allowed when the intermediate dimension matches the input"
        layers[-1].out = nn.Identity()

    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        main_module,
        dropout=0.0,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.dp = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.main_module = main_module

    def forward(
        self, input: Tensor, lengths: Tensor, state: Optional[List[Tensor]] = None
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        # Assume the input takes shape (T, B, D)
        # This makes input -> LayerNorm -> main_module -> dropout -> Residual(+input)
        output = self.ln(input)
        output = self.main_module(output)
        output = self.dp(output)
        output = output + input
        return output, lengths, []


class BidirectionalBasicBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        forward_module,
        backward_module,
        dropout=0.0,
        parameter_noise=0.0,
        residual_norm=True,
    ):
        super().__init__()
        if residual_norm:
            self.ln = nn.LayerNorm(input_dim)
            self.dp = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        else:
            self.ln = self.dp = None

        self.forward_module = forward_module
        self.backward_module = backward_module

        self.linear = Linear(
            input_dim=input_dim * 2,
            output_dim=input_dim,
            parameter_noise=parameter_noise,
        )

    def reverse_padded_sequence(self, input, lengths):
        # return input.flip(dims=[0])
        # Assuming input is of shape BTD
        output = torch.zeros_like(input)
        for i, length in enumerate(lengths):
            output[:length, i] = input[:length, i].flip(0)
        return output
    
    def forward(
        self, input: Tensor, lengths: Tensor, state: Optional[List[Tensor]] = None
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        # Assume the input takes shape (T, B, D)
        if self.ln is not None:
            output = self.ln(input)
        else:
            output = input

        output_flip = self.reverse_padded_sequence(output, lengths)

        # Forward/backward module
        f_output = self.forward_module(output)
        b_output = self.backward_module(output_flip)
        b_output_flip = self.reverse_padded_sequence(b_output, lengths)

        # Concatenation and reduction to correct dim (B, T, D)
        output = torch.cat([f_output, b_output_flip], dim=-1)

        if self.ln is not None:
            output = self.dp(self.linear(output))
            output = output + input
        else:
            output = self.linear(output)

        return output, lengths, []


# For backward compatibility
class mySequentialv2(nn.ModuleList):
    def forward(
        self, input: Tensor, lengths: Tensor, state: Optional[List[Tensor]] = None
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        for module in self._modules.values():
            input, lengths, state = module(input, lengths, state)
        return input, lengths, state


class MHBiS4Layer(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        mssm_num_modules: int = 1, # what is this?
        mssm_num_stacks: int = 2,
        mssm_only_activate_last: bool = False,
        mssm_intermediate_dim: int = 512,
        mssm_hidden_dim: int = 32,
        mssm_num_heads: int = 1,
        mssm_activation: str = "gelu",
        mssm_rank: int = 1,
        mssm_scale: float = 0.50,
        mssm_maxlen: int = 256,
        mssm_timestep_min: float = 0.010,
        mssm_timestep_max: float = 0.160,
        mssm_dropout: float = 0.10,
        mssm_remove_final_linear: bool = False,
        ffn_activation: str = "gelu",
        ffn_dim: int = 2048,
        ffn_dropout: float = 0.10,
        parameter_noise: float = 0.00,
        use_fast_kernel: bool = True,
        s4_only=False,
        create_on_gpu=True
    ):
        super().__init__()

        forward_ssm_modules = [
            build_stacked_mh_s4(
                num_layers=mssm_num_stacks,
                only_activate_last=mssm_only_activate_last,
                input_dim=input_dim,
                intermediate_dim=mssm_intermediate_dim,
                output_dim=input_dim,
                hidden_dim=mssm_hidden_dim,
                num_heads=mssm_num_heads,
                activation=mssm_activation,
                rank=mssm_rank,
                scale=mssm_scale,
                maxlen=mssm_maxlen,
                timestep_min=mssm_timestep_min,
                timestep_max=mssm_timestep_max,
                dropout=mssm_dropout,
                remove_final_linear=mssm_remove_final_linear,
                parameter_noise=parameter_noise,
                use_fast_kernel=use_fast_kernel,
                create_on_gpu=create_on_gpu
            )
            for _ in range(mssm_num_modules)
        ]
        backward_ssm_modules = [
            build_stacked_mh_s4(
                num_layers=mssm_num_stacks,
                only_activate_last=mssm_only_activate_last,
                input_dim=input_dim,
                intermediate_dim=mssm_intermediate_dim,
                output_dim=input_dim,
                hidden_dim=mssm_hidden_dim,
                num_heads=mssm_num_heads,
                activation=mssm_activation,
                rank=mssm_rank,
                scale=mssm_scale,
                maxlen=mssm_maxlen,
                timestep_min=mssm_timestep_min,
                timestep_max=mssm_timestep_max,
                dropout=mssm_dropout,
                remove_final_linear=mssm_remove_final_linear,
                parameter_noise=parameter_noise,
                use_fast_kernel=use_fast_kernel,
                create_on_gpu=create_on_gpu
            )
            for _ in range(mssm_num_modules)
        ]

        self.ssm_block = mySequentialv2(
            [
                BidirectionalBasicBlock(
                    input_dim=input_dim,
                    forward_module=fmodule,
                    backward_module=bmodule,
                    dropout=mssm_dropout,
                    parameter_noise=parameter_noise,
                    residual_norm=not s4_only
                )
                for fmodule, bmodule in zip(forward_ssm_modules, backward_ssm_modules)
            ]
        )

        if not s4_only:
            ffn_module = nn.Sequential(
                Linear(
                    input_dim=input_dim,
                    output_dim=ffn_dim * (2 if ffn_activation == "glu" else 1),
                    parameter_noise=parameter_noise,
                ),
                get_activation(ffn_activation),
                nn.Dropout(ffn_dropout) if ffn_dropout > 0.0 else nn.Identity(),
                Linear(
                    input_dim=ffn_dim,
                    output_dim=input_dim,
                    parameter_noise=parameter_noise,
                ),
            )
            self.ffn_block = BasicBlock(
                input_dim=input_dim, main_module=ffn_module, dropout=ffn_dropout
            )
        else:
            self.ffn_block = None

    def forward(
        self, input: Tensor, lengths: Tensor, state: Optional[List[Tensor]] = None
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        output = input
        output, _, _ = self.ssm_block(output, lengths, state)

        if self.ffn_block is not None:
            output, _, _ = self.ffn_block(output, lengths, state)
        return output, lengths, []


class MHBiS4EncoderLayer(nn.Module):
    def __init__(self, cfg, s4_only=False, create_on_gpu=True):
        super().__init__()
        self.module = self.build_module(cfg, s4_only=s4_only, create_on_gpu=create_on_gpu)

    def build_module(self, cfg, s4_only=False, create_on_gpu=True):
        return MHBiS4Layer(
            input_dim = cfg.encoder_embed_dim,
            mssm_num_modules = 1,
            mssm_num_stacks = cfg.encoder_mssm_num_stacks,
            mssm_only_activate_last = False,
            mssm_intermediate_dim = cfg.encoder_embed_dim,
            mssm_hidden_dim = cfg.encoder_mssm_hidden_dim,
            mssm_num_heads = cfg.encoder_mssm_num_heads,
            mssm_activation = cfg.encoder_mssm_activation,
            mssm_rank = 1,
            mssm_scale = cfg.encoder_mssm_scale,
            mssm_maxlen = cfg.encoder_mssm_maxlen,
            mssm_timestep_min = cfg.encoder_mssm_timestep_min,
            mssm_timestep_max = cfg.encoder_mssm_timestep_max,
            mssm_dropout = cfg.dropout,
            mssm_remove_final_linear = True,
            ffn_activation = cfg.activation_fn,
            ffn_dim = cfg.encoder_ffn_embed_dim,
            ffn_dropout = cfg.relu_dropout or 0,
            parameter_noise = 0.00,
            use_fast_kernel = True  , # Why?
            s4_only=s4_only,
            create_on_gpu=create_on_gpu
        )

    @torch.no_grad()
    def infer_lengths(self, batch, maxlen, encoder_padding_mask: Optional[Tensor]):
        # Assume non padding elements are part of sequence
        lengths = (encoder_padding_mask.ne(1)).sum(-1)
        return lengths.to(int)

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None # relative position encoding
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        bsz, seq_len = x.size(1), x.size(0)
        if encoder_padding_mask is None:
            encoder_padding_mask = x.new_zeros(bsz, seq_len)

        lengths = self.infer_lengths(
            batch = x.size(1),
            maxlen = x.size(0),
            encoder_padding_mask=encoder_padding_mask,
        )
        x, _, _ = self.module(x, lengths)

        return x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        pass


if __name__ == "__main__":

    import json
    from types import SimpleNamespace

    from random import randint

    def json_to_namespace(json_file):
        with open(json_file) as f:
            x = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

        for name in x.__dict__:
            if x.__dict__[name] in ['False', 'True']:
                x.__dict__[name] = (x.__dict__[name] == 'True')
        return x


    cfg = json_to_namespace("mssm_config.json")

    s4_layer = MHBiS4EncoderLayer(cfg, s4_only=True, create_on_gpu=True)

    print(s4_layer)

    s4_layer = s4_layer.cuda()

    t = 512
    b = 16
    h = 1024

    x = torch.randn(*(t, b, h)).cuda()

    mask = torch.ones(*(b, t), dtype=torch.bool)

    for i in range(b):

        l = randint(t//2, t)
        mask[i][0:l].fill_(0)

    x = x.half()
    print(x.size(), x.type())
    with autocast(enabled=True, dtype=torch.float16):
        output = s4_layer(x, mask)

    print(output.size())
    print(output.sum())
    n_params = 0

    for param in s4_layer.parameters():
        n_params += param.numel()

    print(n_params)
    print(n_params * 24 )