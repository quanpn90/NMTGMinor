import torch
import torch.nn as nn
import torch.nn.functional as F
from onmt.models.transformer_layers import PrePostProcessing
from onmt.modules.linear import FeedForward
from onmt.modules.attention import MultiHeadAttention
from torch.autograd.function import Function
import sys
from torch.utils.checkpoint import get_device_states, set_device_states


class ReversibleTransformerEncoder(nn.Module):

    def __init__(self, opt, death_rate=0.0):

        self.variational = opt.variational_dropout
        d_model = opt.model_size
        p = opt.dropout
        self.death_rate = death_rate
        self.dropout = p
        h = opt.n_heads
        attn_p = opt.attn_dropout
        n_layers = opt.layers

        super().__init__()
        self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        self.multihead = MultiHeadAttention(h, d_model, attn_p=attn_p, share=2)

        ff_p = opt.dropout
        self.feedforward = FeedForward(opt.model_size, opt.inner_size, ff_p, variational=self.variational)