import numpy as np
import onmt
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def embedded_dropout(embed, words, dropout=0.1, scale=None):

    if dropout > 0:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
            padding_idx = -1
    
    x = F.embedding(
            words, masked_embed_weight, padding_idx, embed.max_norm,
            embed.norm_type, embed.scale_grad_by_freq, embed.sparse)

    return x



def switchout(words,  vocab_size, tau=1.0):
    """

    :param words: torch.Tensor(b x l)
    :param vocab_size: vocabulary size
    :param tau: temperature control
    :return:
    sampled_words torch.LongTensor(b x l)
    """
    mask = torch.eq(words, onmt.Constants.BOS) | torch.eq(words, onmt.Constants.EOS) | torch.eq(words, onmt.Constants.PAD)

    # print(mask)

    lengths = (1 - mask).float().sum(dim=1)

    # print(lengths)

    batch_size, n_steps = words.size()

    # first, sample the number of words to corrupt for each sent in batch
    logits = torch.arange(n_steps).type_as(words).float() # size l

    logits = logits.mul_(-1).unsqueeze(0).expand_as(words).contiguous().masked_fill_(mask, -float("inf"))

    probs = torch.nn.functional.log_softmax(logits.mul_(tau), dim=1)

    probs = torch.exp(probs)

    num_words = torch.distributions.Categorical(probs).sample().float()

    # second, sample the corrupted positions
    corrupt_pos = num_words.div(lengths)
    corrupt_pos = corrupt_pos.unsqueeze(1).expand_as(words).contiguous()

    corrupt_pos.masked_fill_(mask, 0)
    # print(corrupt_pos) # this one currently is broken

    corrupt_pos = torch.bernoulli(corrupt_pos, out=corrupt_pos).byte()

    total_words = int(corrupt_pos.sum())

    # sample the corrupted values, which will be added to sents
    corrupt_val = torch.LongTensor(total_words).type_as(words)
    corrupt_val = corrupt_val.random_(1, vocab_size)

    # corrupts = torch.zeros(batch_size, n_steps).type_as(words).long()
    corrupts = words.clone().zero_()
    corrupts = corrupts.masked_scatter_(corrupt_pos, corrupt_val)
    # this is to ensure that
    sampled_words = words.add(corrupts).remainder_(vocab_size)

    return sampled_words