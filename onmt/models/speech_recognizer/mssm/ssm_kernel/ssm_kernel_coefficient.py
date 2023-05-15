#!/usr/bin/env python3

import torch
from opt_einsum import contract

import os
import pathlib
import ssm_kernel_coefficient_cuda

# from torch.utils.cpp_extension import load
# ssm_kernel_coefficient_binding = load(
#     name="ssm_kernel_coefficient_binding",
#     sources=[
#         os.path.join(
#             pathlib.Path(__file__).parent.resolve(),
#             "ssm_kernel_coefficient_binding_cuda.cu"
#         )
#     ],
#     verbose = True
# )

# pyre-ignore
from ssm_kernel_coefficient_binding import (
    kernel_coefficient_backward_double,
    kernel_coefficient_backward_float,
    kernel_coefficient_forward_double,
    kernel_coefficient_forward_float,
)


def compute_kernel_coefficient(z, d, t, b, c, fast=False):
    if not fast or not z.is_cuda:
        return compute_slow(z, d, t, b, c)
    return compute_fast(z, d, t, b, c)


def get_dwoodbury(z, d, invt):
    # Get the bilinear transformation
    z = contract("l,qh->qlh", torch.view_as_complex(z), invt)
    # Compute the term and reuse computations (Q, L, H, N)
    return 1 / (z.unsqueeze(-1) - d.unsqueeze(-2).unsqueeze(-2))


def compute_slow(z, d, t, b, c):
    # Get the diagonal component in the woodbury computation
    # which will be reused in computing the kernel

    # z is forced to be fp32
    # the following prevents fp16 underflow, particularly on t
    if t.dtype == torch.float16:
        t = t.to(z.dtype)
        b = b.to(z.dtype)
        c = c.to(z.dtype)
        d = d.to(z.dtype)

    r = get_dwoodbury(z, d, 1 / t)  # (Q, L, H, N)
    # Compute kernel coeffs
    kernelcc = contract("qihn,qlhn,qchn->qiclh", b.to(r.dtype), r, c)
    return kernelcc


def compute_fast(z, d, t, b, c):
    # z is forced to be fp32
    # the following prevents fp16 underflow, particularly on t
    if t.dtype == torch.float16:
        t = t.to(z.dtype)
        b = b.to(z.dtype)
        c = c.to(z.dtype)
    zz = contract("l,qh->qlh", torch.view_as_complex(z), 1 / t)  # (Q, L, H)
    bc = contract("qihn,qchn->icqhn", b, c).to(zz.dtype)  # (I, C, Q, H, N)
    I, C, Q, H, N = bc.shape
    bc = bc.view(-1, Q, H, N)
    L = zz.shape[1]
    d = d.to(zz.dtype) # (Q, N)
    coeff = KernelCoefficientFast.apply(bc, zz, d)  # (IC, Q, L, H)
    return coeff.view(I, C, Q, L, H).permute(2, 0, 1, 3, 4)  # (Q, I, C, L, H)


class KernelCoefficientFast(torch.autograd.Function):
    # Compute sum{n} { a[n] / (b[l] - c[n]) }
    @staticmethod
    def forward(ctx, a_n, b_l, c_n):
        if not a_n.is_cuda and b_l.is_cuda and c_n.is_cuda:
            raise NotImplementedError("Only support CUDA tensors")
        ctx.save_for_backward(a_n, b_l, c_n)
        if b_l.dtype == torch.complex64:
            return ssm_kernel_coefficient_cuda.kernel_coefficient_forward_float(a_n, b_l, c_n)
        else:
            return ssm_kernel_coefficient_cuda.kernel_coefficient_forward_double(a_n, b_l, c_n)

    @staticmethod
    def backward(ctx, dout):
        a_n, b_l, c_n = ctx.saved_tensors
        if b_l.dtype == torch.complex64:
            da_n, db_l, dc_n = ssm_kernel_coefficient_cuda.kernel_coefficient_backward_float(a_n, b_l, c_n, dout)
        else:
            da_n, db_l, dc_n = ssm_kernel_coefficient_cuda.kernel_coefficient_backward_double(a_n, b_l, c_n, dout)
        return da_n, db_l, dc_n


if __name__ == "__main__":
    
    # Test
    num_heads = 4
    input_dim = 64
    hid_dim = 32
    seq_len = 256
    dtype=torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)
    b = torch.randn(num_heads, 2, input_dim, hid_dim, device=device, dtype=dtype).requires_grad_(True)
    c = torch.randn(num_heads, 2, input_dim, hid_dim, device=device, dtype=dtype).requires_grad_(True)
    z = torch.randn(seq_len, 2, device=device, dtype=dtype)
    t = torch.randn(num_heads, input_dim, device=device, dtype=dtype).requires_grad_(True)
    d = torch.randn(num_heads, hid_dim, device=device, dtype=dtype).requires_grad_(True)

    zz = z.to(torch.float64)
    dd = d.to(torch.float64)
    tt = t.to(torch.float64)
    bb = b.to(torch.float64)
    cc = c.to(torch.float64)

    ans64 = compute_slow(zz, dd, tt, bb, cc)
    ans = compute_slow(z, d, t, b, c)
    out64 = compute_fast(zz, dd, tt, bb, cc)
    out = compute_fast(z, d, t, b, c)

    err = torch.rand_like(out)

    ans64_dd, ans64_dt, ans64_db, ans64_dc = torch.autograd.grad(
        ans64, (dd, tt, bb, cc), err, retain_graph=True
    )
    ans_dd, ans_dt, ans_db, ans_dc = torch.autograd.grad(
        ans, (d, t, b, c), err, retain_graph=True
    )
    out64_dd, out64_dt, out64_db, out64_dc = torch.autograd.grad(
        out64, (dd, tt, bb, cc), err, retain_graph=True
    )
    out_dd, out_dt, out_db, out_dc = torch.autograd.grad(
        out, (d, t, b, c), err, retain_graph=True
    )

    print()
    print("out: max abs error (ans64, out64)", torch.max(torch.abs(out64 - ans64)))
    print("dd: max abs error (ans64, out64)", torch.max(torch.abs(ans64_dd - out64_dd)))
    print("dt: max abs error (ans64, out64)", torch.max(torch.abs(ans64_dt - out64_dt)))
    print("db: max abs error (ans64, out64)", torch.max(torch.abs(ans64_db - out64_db)))
    print("dc: max abs error (ans64, out64)", torch.max(torch.abs(ans64_dc - out64_dc)))

    print()
    print("out: max abs error (ans64, out)", torch.max(torch.abs(out - ans64)))
    print("dd: max abs error (ans64, out)", torch.max(torch.abs(ans64_dd - out_dd)))
    print("dt: max abs error (ans64, out)", torch.max(torch.abs(ans64_dt - out_dt)))
    print("db: max abs error (ans64, out)", torch.max(torch.abs(ans64_db - out_db)))
    print("dc: max abs error (ans64, out)", torch.max(torch.abs(ans64_dc - out_dc)))

    print()
    print("out: max abs error (ans, out64)", torch.max(torch.abs(out64 - ans)))
    print("dd: max abs error (ans, out64)", torch.max(torch.abs(ans_dd - out64_dd)))
    print("dt: max abs error (ans, out64)", torch.max(torch.abs(ans_dt - out64_dt)))
    print("db: max abs error (ans, out64)", torch.max(torch.abs(ans_db - out64_db)))
    print("dc: max abs error (ans, out64)", torch.max(torch.abs(ans_dc - out64_dc)))

    print()
    print("out: max abs error (ans, out)", torch.max(torch.abs(out - ans64)))
    print("dd: max abs error (ans, out)", torch.max(torch.abs(ans_dd - out_dd)))
    print("dt: max abs error (ans, out)", torch.max(torch.abs(ans_dt - out_dt)))
    print("db: max abs error (ans, out)", torch.max(torch.abs(ans_db - out_db)))
    print("dc: max abs error (ans, out)", torch.max(torch.abs(ans_dc - out_dc)))