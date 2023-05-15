import torch
import torch.nn as nn
import numpy as np


class LRU(nn.Module):
    def __init__(self, H, N, reverse=False, r_min=0, r_max=1, max_phase=2 * np.pi):
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
        B = torch.rand((H, N)) / np.sqrt(2 * H) + 1j * torch.rand((H, N)) / np.sqrt(2 * H)  # H x N
        self.C = nn.Parameter(torch.rand((N, H)) / np.sqrt(N) + 1j * torch.rand((N, H)) / np.sqrt(N))  # N x H

        # Normalization factor
        diag_lambda = torch.exp(-torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log))  # N
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))  # N

        self.B = nn.Parameter(B * gamma_log)  # H x N

        self.reverse = reverse

    def forward(self, u, lenghts):
        """Forward pass of the LRU layer. Output sequence y and input_sequence u are of shape (B, L, H)."""

        Lambda = torch.exp(-torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log))  # N

        if not self.reverse:
            exp = torch.arange(-1, -u.shape[1] - 1, -1, dtype=torch.float32).view(1, -1, 1).expand(u.shape[0], -1,
                                                                                                   -1)  # B
            # x L x 1
            exp = exp + lenghts.view(-1, 1, 1)  # B x L x 1
            exp.clamp_(min=0)
        else:
            exp = torch.arange(u.shape[1], dtype=torch.float32).view(1, -1, 1).expand(u.shape[0], -1, -1)  # B x L x 1

        Lambda_exp = Lambda.pow(exp)  # B x L x N

        # Bu = torch.matmul(u.to(torch.complex32 if u.dtype==torch.float16 else torch.complex64), self.B) # B x L x N
        Bu = torch.matmul(u, self.B.real) + 1j * torch.matmul(u, self.B.imag)  # B x L x N

        prod = Lambda_exp * Bu  # B x L x N
        x = prod.cumsum(0)  # B x L x N

        # y = torch.matmul(x, self.C).real # B x L x H
        y = torch.matmul(x.real, self.C.real) - torch.matmul(x.imag, self.C.imag)  # B x L x H
        return y


if __name__ == "__main__":
    # import torch_directml
    device = "cpu"  # torch_directml.device()

    B = 4
    L = 1000
    d_model = 1024
    d_hidden = 1024
    reverse = False

    lengths = torch.randint(1, L, (B,))

    layer = LRU(d_model, d_hidden, reverse=reverse).to(device)

    seq = torch.randn(B, L, d_model, device=device)

    print("START")
    seq = layer(seq, lengths)
    print(seq.mean(), seq.std())