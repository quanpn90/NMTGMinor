import logging, traceback
import os, re
import torch
import torchaudio
import math
import soundfile as sf

import torch
import torch.nn.functional as F



# this function is borrowed from Facebook
# avoid jumping into the middle of a character
def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


def pad_tensor(x, min_length=2400):

    if x.size(0) < min_length:
        x_ = x.new_zeros((min_length, x.size(1)))
        x_[:x.size(0), :] = x
        x = x_

    return x


# this function reads wav file based on the timestamp in seconds
def safe_readaudio(wav_path, start=0.0, end=0.0, sample_rate=16000):

    offset = math.floor(sample_rate * start)
    num_frames = -1 if end <= start else math.ceil(sample_rate * (end - start))

    # by default torchaudio normalizes the read tensor
    tensor, _ = torchaudio.load(wav_path, frame_offset=offset, num_frames=num_frames,
                                normalize=True, channels_first=False)
    tensor = tensor[:, 0].unsqueeze(1)

    tensor = pad_tensor(tensor)

    # tensor has size [length, num_channel] in which channel should be 1 for wav2vec
    return tensor


# this function is borrowed from fairseq
# https://github.com/pytorch/fairseq/blob/master/fairseq/utils.py
def checkpoint_paths(path, pattern=r'model_ppl_(\d+).(\d+)\_e(\d+).(\d+).pt'):
    """Retrieves all checkpoints found in `path` directory.
    Checkpoints are identified by matching filename to the specified pattern. If
    the pattern contains groups, the result will be sorted by the first group in
    descending order.
    """
    pt_regexp = re.compile(pattern)
    files = []

    # remove directories or files that don't contain "ppl"
    for fname in os.listdir(path):
        cur_path = os.path.join(path, fname)
        if os.path.isdir(cur_path):
            continue
        elif "ppl" in fname:
            files.append(fname)

    # sort py perplexity (ascending)
    files = sorted(files, key=lambda s: float(s.split("_")[2]))

    entries = []
    for i, f in enumerate(files):
        m = pt_regexp.fullmatch(f)
        if m is not None:
            idx = int(m.group(1)) if len(m.groups()) > 0 else i
            entries.append((idx, m.group(0)))
    # return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]
    return [os.path.join(path, x[1]) for x in entries]


def normalize_gradients(parameters, denom=1.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    for p in parameters:
        p.grad.detach().div_(denom)
    return


# flip a tensor on certain dimension
def flip(x, dim=0):
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
                 else x.new(torch.arange(x.size(i) - 1, -1, -1).tolist()).long()
                 for i in range(x.dim()))
    return x[inds]


# Stochastic expected length
def expected_length(length, death_rate):
    e_length = 0

    for l in range(length):
        survival_rate = 1.0 - (l + 1) / length * death_rate

        e_length += survival_rate

    return e_length


from typing import Union, Iterable

try:
    from torch._six import inf
except ModuleNotFoundError:
    from torch import inf
 

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def clip_grad_norm(
        parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False) -> torch.Tensor:
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:``parameters`` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')

    if max_norm > 0:
        clip_coef = max_norm / (total_norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for p in parameters:
            p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))
    return total_norm


class IndexFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, indices):
        # ctx.save_for_backward(indices)
        ctx.first_axis_dim = input.shape[0]
        assert input.ndim == 2
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return input[indices]

        d = input.shape[1]
        repeated_indices = indices.repeat(d, 1).view(indices.size(0), d)
        ctx.save_for_backward(repeated_indices)

        return torch.gather(input, 0, repeated_indices)

    @staticmethod
    def backward(ctx, grad_output):
        # indices, = ctx.saved_tensors
        grad_input = torch.zeros([ctx.first_axis_dim, *grad_output.shape[1:]],
                                 device=grad_output.device, dtype=grad_output.dtype)
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        repeated_indices, = ctx.saved_tensors

        grad_input.scatter_(0, repeated_indices, grad_output)
        return grad_input, None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        # ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim == 2
        output = torch.zeros(first_axis_dim, values.shape[1], device=values.device,
                             dtype=values.dtype)
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # output[indices] = values
        d = values.shape[1]
        repeated_indices = indices.repeat(d, 1).view(indices.size(0), d)
        ctx.save_for_backward(repeated_indices)
        # output.scatter_(0, repeat(indices, 'z -> z d', d=values.shape[1]), values)
        output.scatter_(0, repeated_indices, values)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        repeated_indices, = ctx.saved_tensors
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # grad_values = grad_output[indices]
        # grad_values = torch.gather(grad_output, 0, repeat(indices, 'z -> z d', d=grad_output.shape[1]))
        grad_values = torch.gather(grad_output, 0, repeated_indices)
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


def unpad_input(hidden_states, indices):
    """
    Arguments:
        hidden_states: (batch, seqlen, dim)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, dim), where total_nnz = number of tokens in selected in attention_mask.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    # seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # max_seqlen_in_batch = seqlens_in_batch.max().item()
    # cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.

    hidden_states = hidden_states.view(-1, hidden_states.size(-1))

    return index_first_axis(hidden_states, indices)


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, dim), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz)
    Return:
        hidden_states: (batch, seqlen, dim)
    """
    # dim = hidden_states.shape[-1]
    # output = torch.zeros((batch * seqlen), dim, device=hidden_states.device, dtype=hidden_states.dtype)
    # output[indices] = hidden_states
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    # return rearrange(output, '(b s) d -> b s d', b=batch)
    output = output.view(batch, seqlen, output.size(-1))

    return output

