import torch.nn as nn
import torch


#  Positional Embedding with discrete inputs
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(SinusoidalPositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, sin_first=True, bsz=None):
        """
        :param bsz: integer to repeat
        :param pos_seq: sequences of RELATIVE position indices (can be negative for future)
        :param sin_first: in Attention is all you need paper, sin is first then cosin
        """
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq.type_as(pos_seq))

        if sin_first:
            pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        else:
            pos_emb = torch.cat([sinusoid_inp.cos(), sinusoid_inp.sin()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].repeat(1, bsz, 1)
        else:
            return pos_emb[:, None, :]