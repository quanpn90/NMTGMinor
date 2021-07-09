import torch
import torch.nn.functional as F
from einops import rearrange, repeat


class SinusoidalEmbeddings(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x=None, length=0, timestep=-1):
        """
        :param timestep:
        :param length:
        :param x: [time x bsz x hidden]
        :return:
        """
        # actually this module doesn't care about anything of x except x.size(1)

        if x is not None:
            assert length == 0 and timestep == -1
            n = x.shape[0]  # time dimension
        elif length > 0:
            assert timestep == -1
            n = length
        elif timestep >= 0:
            n = timestep + 1

        t = torch.arange(n, device=self.inv_freq.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb
