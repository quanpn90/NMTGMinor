import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import onmt


class VariationalDropout(torch.nn.Module):
    def __init__(self, p=0.5, batch_first=False):
        super().__init__()
        self.p = p
        self.batch_first = batch_first

    def forward(self, x):
        if not self.training or not self.p:
            return x

        if self.batch_first:
            m = x.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)
        else:
            m = x.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.p)

        mask = m / (1 - self.p)
        # mask = mask.expand_as(x)

        return mask * x


def variational_dropout(x, p=0.5, training=True, inplace=False, batch_first=False):

    if not training or p <= 0:
        return x

    if batch_first:
        m = x.new(x.size(0), 1, x.size(2)).bernoulli_(1 - p)
    else:
        m = x.new(1, x.size(1), x.size(2)).bernoulli_(1 - p)

    m.div_(1 - p)

    if inplace:
        x.mul_(m)
        return x
    else:
        return x * m



def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1
    # X = embed._backend.Embedding.apply(words, masked_embed_weight,
        # padding_idx, embed.max_norm, embed.norm_type,
        # embed.scale_grad_by_freq, embed.sparse
    # )
    x = F.embedding(
            words, masked_embed_weight, padding_idx, embed.max_norm,
            embed.norm_type, embed.scale_grad_by_freq, embed.sparse)

    return x


def switchout(words, vocab_size, tau=1.0, transpose=False, offset=0):
    """
    :param offset: number of initial tokens to be left "untouched"
    :param transpose: if the tensor has initial size of l x b
    :param words: torch.Tensor(b x l)
    :param vocab_size: vocabulary size
    :param tau: temperature control
    :return:
    sampled_words torch.LongTensor(b x l)
    """

    if transpose:
        words = words.t()

    if offset > 0:
        offset_words = words[:, :offset]
        words = words[:, offset:]

    mask = torch.eq(words, onmt.constants.BOS) | \
           torch.eq(words, onmt.constants.EOS) | torch.eq(words, onmt.constants.PAD)
    lengths = (1 - mask.byte()).float().sum(dim=1)
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

    corrupt_pos = torch.bernoulli(corrupt_pos, out=corrupt_pos).byte()

    total_words = int(corrupt_pos.sum())

    # sample the corrupted values, which will be added to sents
    corrupt_val = torch.LongTensor(total_words).type_as(words)
    corrupt_val = corrupt_val.random_(1, vocab_size)

    corrupts = words.clone().zero_()
    corrupts = corrupts.masked_scatter_(corrupt_pos.type_as(mask), corrupt_val)
    # to add the corruption and then take the remainder w.r.t the vocab size
    sampled_words = words.add(corrupts).remainder_(vocab_size)

    if offset > 0:
        sampled_words = torch.cat([offset_words, sampled_words], dim=1)

    if transpose:
        sampled_words = sampled_words.t()

    return sampled_words
