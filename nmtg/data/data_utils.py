import numpy as np
from typing import Mapping, Sequence

import torch
import torch.utils.data
from torch import Tensor

from nmtg.data.dictionary import Dictionary


def parse_embedding(embed_path):
    """Parse embedding text file into a dictionary of word and embedding tensors.

    The first line can have vocabulary size and dimension. The following lines
    should contain word and embedding separated by spaces.

    Example:
        2 5
        the -0.0230 -0.0264  0.0287  0.0171  0.1403
        at -0.0395 -0.1286  0.0275  0.0254 -0.0932
    """
    embed_dict = {}
    with open(embed_path) as f_embed:
        next(f_embed)  # skip header
        for line in f_embed:
            pieces = line.rstrip().split(" ")
            embed_dict[pieces[0]] = torch.tensor([float(weight) for weight in pieces[1:]])
    return embed_dict


def load_embedding(embed_dict, vocab: Dictionary, embedding):
    for idx in range(len(vocab)):
        token = vocab.symbol(idx)
        if token in embed_dict:
            embedding.weight.data[idx] = embed_dict[token]
    return embedding


def generate_length_based_batches_from_samples(samples, length_per_batch, max_examples_per_batch=128,
                                               batch_size_align=1, count_padding=True, key_fn=None,
                                               len_fn=None, filter_fn=None):
    """
    Generate batches of indices such that the total length of each batch does not exceed length_per_batch.
    :param samples: An iterable of samples
    :param length_per_batch: Maximum total length per batch
    :param max_examples_per_batch: Maximum number of samples per batch
    :param batch_size_align: Make sure batch size is a multiple of this number,
        unless it would be smaller than this number
    :param count_padding: Count padding against the total length of the batch
    :param key_fn: Optional function to get a sort key from a sample
    :param len_fn: Optional function to get the length of a sample
    :param filter_fn: Optional function that takes a sample and returns a boolean.
        Use only those examples where the function returns True
    :return: A list of batches. Each batch is a list of integer indices
    """
    if len_fn is None:
        len_fn = len
    if key_fn is None:
        key_fn = len
    if filter_fn is None:
        filter_fn = lambda x: True

    lengths = np.empty(len(samples), np.long)
    keys = []
    filtered = []
    for i, sample in enumerate(samples):
        lengths[i] = len_fn(sample)
        filtered.append(filter_fn(sample))
        keys.append(key_fn(sample))
    indices = sorted(filter(lambda x: filtered[x], range(len(lengths))), key=keys.__getitem__)

    return _generate_length_based_batches(lengths, indices, length_per_batch, max_examples_per_batch,
                                          batch_size_align, count_padding)


def generate_length_based_batches_from_lengths(lengths, length_per_batch, max_examples_per_batch=128,
                                               batch_size_align=1, count_padding=True, key_fn=None, filter_fn=None):
    """
    Generate batches of indices such that the total length of each batch does not exceed length_per_batch.
    :param lengths: A list of lengths for all samples
    :param length_per_batch: Maximum total length per batch
    :param max_examples_per_batch: Maximum number of samples per batch
    :param batch_size_align: Make sure batch size is a multiple of this number,
        unless it would be smaller than this number
    :param count_padding: Count padding against the total length of the batch
    :param key_fn: Optional function to get a sort key from a sample index
    :param filter_fn: Optional function that takes an index and returns a boolean.
        Use only indices where the function returns True
    :return: A list of batches. Each batch is a list of integer indices
    """
    if key_fn is None:
        key_fn = lengths.__getitem__
    if filter_fn is None:
        filter_fn = lambda x: True

    indices = sorted(filter(filter_fn, range(len(lengths))), key=key_fn)

    return _generate_length_based_batches(lengths, indices, length_per_batch, max_examples_per_batch,
                                          batch_size_align, count_padding)


def _generate_length_based_batches(lengths, indices, length_per_batch, max_examples_per_batch=128,
                                   batch_size_align=1, count_padding=True):
    batches = []
    cur_batch = []
    cur_batch_size = 0
    cur_batch_sizes = []
    sample_length = -1

    def batch_is_full():
        """Returns True if the batch would exceed the maximum size if we added the current example"""

        if len(cur_batch) == max_examples_per_batch:
            return True

        if count_padding:
            # because data is sorted by length, the current length is the longest one so far
            longest_length = sample_length

            if len(cur_batch_sizes) > 0:
                longest_length = max(max(cur_batch_sizes), sample_length)

            if longest_length * (len(cur_batch) + 1) > length_per_batch:
                return True
        else:
            if cur_batch_size + sample_length > length_per_batch:
                return True

        return False

    for i in indices:
        sample_length = lengths[i]

        if batch_is_full():
            current_size = len(cur_batch)

            scaled_size = current_size
            # cut down batch size to a multiple of batch_size_align
            if current_size > batch_size_align:
                scaled_size = batch_size_align * (current_size // batch_size_align)

            trunc_batch = cur_batch[:scaled_size]
            if batch_size_align > 1:
                assert (len(trunc_batch) < batch_size_align or len(trunc_batch) % batch_size_align == 0), \
                    'Batch size is not a multiple of {}, current batch_size is {}' \
                    .format(batch_size_align, len(trunc_batch))
            batches.append(trunc_batch)  # add this batch into the batch list

            cur_batch = cur_batch[scaled_size:]  # reset the current batch
            cur_batch_sizes = cur_batch_sizes[scaled_size:]
            cur_batch_size = sum(cur_batch_sizes)

        cur_batch.append(i)
        cur_batch_size += sample_length
        cur_batch_sizes.append(sample_length)

    if len(cur_batch) > 0:
        batches.append(cur_batch)

    return batches


def collate_tensor_structures(samples):
    """Collates structures of tensors of the same shape. Keeps the structure"""
    assert len(samples) > 0
    test = samples[0]

    if isinstance(test, Tensor):
        return torch.stack(samples)
    elif isinstance(test, Mapping):
        return {k: collate_tensor_structures([item[k] for item in samples]) for k in test.keys()}
    elif isinstance(test, Sequence):
        return [collate_tensor_structures([item[i] for item in samples]) for i in range(len(test))]
    else:
        raise NotImplementedError


def collate_sequences(samples, pad, align_right=False):
    """Collates 1D tensors of different length with padding."""
    lengths = [x.size(0) for x in samples]
    max_length = max(lengths)
    # initialize with batch_size * length first
    tensor = samples[0].new(len(samples), max_length).fill_(pad)

    def copy_tensor(src, dst):
        dst.copy_(src)

    for i, sample in enumerate(samples):
        copy_tensor(sample, tensor[i][max_length - lengths[i]:] if align_right else
                    tensor[i][:lengths[i]])

    return tensor, torch.tensor(lengths)
