import torch
import onmt
# from math import is_close


def build_unigram_noise(freq, alpha=1.0):
    """
    :param alpha: scaling factor. 0.0 = uniform distribution
    :param freq: torch tensor with frequencies of each word
    :return: torch tensor - probability distribution (multinomial distribution)
    """

    probs = freq.new(*freq.size()).copy_(freq)

    # don't sample PAD or BOS
    probs[onmt.constants.PAD] = 0
    probs[onmt.constants.BOS] = 0

    probs = probs / probs.sum()
    probs = torch.pow(probs, alpha)
    probs = probs / probs.sum()

    return probs
