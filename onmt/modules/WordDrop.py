import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout > 0:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        mask = Variable(mask)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
            padding_idx = -1
    
    X =  F.embedding(
            words, masked_embed_weight, padding_idx, embed.max_norm,
            embed.norm_type, embed.scale_grad_by_freq, embed.sparse)

    
    
    return X
