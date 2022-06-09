import torch
from torch import nn, einsum
# from einops import rearrange, repeat


class SinusoidalEmbeddings(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    # def forward(self, x=None, length=0, timestep=-1):
    #     """
    #     :param timestep:
    #     :param length:
    #     :param x: [time x bsz x hidden]
    #     :return:
    #     """
    #     # actually this module doesn't care about anything of x except x.size(1)
    #
    #     if x is not None:
    #         assert length == 0 and timestep == -1
    #         n = x.shape[0]  # time dimension
    #     elif length > 0:
    #         assert timestep == -1
    #         n = length
    #     elif timestep >= 0:
    #         n = timestep + 1
    #
    #     t = torch.arange(n, device=self.inv_freq.device).type_as(self.inv_freq)
    #     sinusoid_inp = torch.einsum('i , j -> i j', t, self.inv_freq)
    #     emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
    #     return emb

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[:, None, :]
            self.sin_cached = emb.sin()[:, None, :]
        return self.cos_cached, self.sin_cached

