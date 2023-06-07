import sys
python = sys.argv[1]=="0"
import time

if not python:
    from torch.utils.cpp_extension import load

    blru = load(name="blru", sources=["blru.cpp","blru_kernel.cu"]) #, verbose=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.cuda.amp import autocast

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="ComplexHalf support is experimental*")

class BLRUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Lambda, Bu, lengths, direction):
        max_len = lengths.max().item()
        arange = torch.arange(int(math.log(max_len,2))+1, device=lengths.device) # log(L)

        two_exp = 2**arange # log(L)
        Lambda_exp = Lambda.unsqueeze(-1).pow(two_exp).to(Bu.dtype) # N x log(L)

        output = blru.forward(Lambda_exp, Bu.contiguous(), lengths, direction)

        variables = [Lambda_exp, output, lengths]
        ctx.direction = direction
        ctx.save_for_backward(*variables)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Why to conj in this function is explained here: https://pytorch.org/docs/stable/notes/autograd.html

        # Basically:
        # grad_output is \partial L/\partial s* (thats what pytorch calculates)
        # => grad_output.conj() is \partial L/\partial s (because the domain of L is R)
        # => outputs is \partial L/\partial z (thats what the backward function calculates)
        # => d_Lambda.conj(), d_Bu.conj() is \partial L/\partial z* (again because the domain of L is R)

        Lambda_exp, output, lengths = ctx.saved_tensors
        outputs = blru.backward(grad_output.conj().contiguous(), Lambda_exp, output, lengths, ctx.direction)
        d_Lambda, d_Bu = outputs
        return d_Lambda.conj(), d_Bu.conj(), None, None

def BLRUFunctionPython(Lambda, Bu, lengths, direction):
    if direction == 2:
        return torch.cat([BLRUFunctionPython(Lambda[:Bu.shape[1]//2],Bu[:,:Bu.shape[1]//2],lengths,0),
                          BLRUFunctionPython(Lambda[Bu.shape[1]//2:],Bu[:,Bu.shape[1]//2:],lengths,1)],1)

    # Bu has shape [B x T x D]

    output = torch.empty_like(Bu)
    for B in range(Bu.shape[0]):
        for N in range(Bu.shape[1]):
            if direction == 0:
                for L in range(lengths[B]):
                    if L == 0:
                        v = Bu[B,N,L]
                    else:
                        v = Lambda[N] * v + Bu[B,N,L]
                    output[B,N,L] = v
            elif direction == 1:
                for L in range(lengths[B]-1,-1,-1):
                    if L == lengths[B]-1:
                        v = Bu[B,N,L]
                    else:
                        v = Lambda[N] * v + Bu[B,N,L]
                    output[B,N,L] = v
            else:
                raise NotImplementedError
    return output

class BLRU(nn.Module):
    def __init__(self, H, N, direction=0, r_min=0, r_max=1, max_phase=2*np.pi):
        super().__init__()

        """Initialize parameters of the LRU layer."""

        # N: state dimension, H: model dimension
        # Initialization of Lambda is complex valued distributed uniformly on ring
        # between r_min and r_max, with phase in [0, max_phase].
        u1 = torch.rand((N,))  # N
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2)))  # N
        u2 = torch.rand((N,))  # N
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))  # N

        # Glorot initialized Input/Output projection matrices
        B = torch.randn(H, N, 2) / ((2 * H) ** 0.5)  # H x N x 2
        self.C = nn.Parameter(torch.randn(2, H, N) / (N ** 0.5))  # 2 x N x H

        with torch.no_grad():
            # Normalization factor
            diag_lambda = torch.exp(-torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log))  # N
            gamma_log = torch.sqrt(1 - torch.abs(diag_lambda) ** 2).unsqueeze(-1)  # N x 1

        self.B = nn.Parameter(B * gamma_log)  # H x N x 2

        self.direction = direction

    def forward(self, u, lengths, python=False):
        """Forward pass of the LRU layer. Output sequence y and input_sequence u are of shape (B, L, H)."""

        Lambda = torch.exp(-torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log))  # N
        Bu = torch.view_as_complex(torch.einsum("blh,hnz->bnlz", u, self.B))  # B x L x N

        if not python:
            x = BLRUFunction.apply(Lambda, Bu, lengths, self.direction).transpose(2, 1)  # B x L x N
        else:
            x = BLRUFunctionPython(Lambda, Bu, lengths, self.direction).transpose(2, 1)  # B x L x N

        y = torch.matmul(x.real, self.C[0]) - torch.matmul(x.imag, self.C[1])  # B x L x H
        return y

    def calc(seq, lengths, python, label):
        with autocast(enabled=True):
            for i in range(24):
                seq = layer(seq, lengths, python=python)

        err = seq - label
        mask = torch.arange(err.shape[1], device=err.device).unsqueeze(0) < lengths.unsqueeze(1)
        loss = (err * err)[mask].sum()
        loss.backward()

        return seq, loss


if __name__ == "__main__":
    device = "cuda"
    torch.manual_seed(42)

    B = 64
    L = 1000
    H = d_model = 1024
    N = d_hidden = 1024
    """B = 3
    L = 20
    H = d_model = 4
    N = d_hidden = 4"""
    direction = 2

    layer = BLRU(d_model, d_hidden, direction).to(device)

    inp = torch.randn(B, L, d_model, device=device, requires_grad=True)
    lengths = torch.randint(1, L+1, (B,), dtype=torch.int32, device=device)
    label = torch.randn_like(inp)

    if python:
        print("PYTHON")
    else:
        print("CUDA")

    n = 0
    t = time.time()
    while True:
        seq, loss = calc(inp, lengths, python, label)
        n += 1

        print("n",n,time.time()-t)
        t = time.time()

        if n==1:
            #print("output",seq)
            print("d_nu_log",layer.nu_log.grad)
            print("d_theta_log",layer.theta_log.grad)
            print("d_B",layer.B.grad)