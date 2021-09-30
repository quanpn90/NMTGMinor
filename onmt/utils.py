import logging, traceback
import os, re
import torch
import torchaudio
import math
import soundfile as sf


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


# this function reads wav file based on the timestamp in seconds
def safe_readaudio(wav_path, start=0.0, end=0.0, sample_rate=16000):

    offset = math.floor(sample_rate * start)
    num_frames = -1 if end <= start else math.ceil(sample_rate * (end - start))
    # stop = -1 if end <= start else math.ceil(sample_rate * end)
    # by default torchaudio normalizes the read tensor -> manually normalize later
    tensor, _ = torchaudio.load(wav_path, frame_offset=offset, num_frames=num_frames,
                                normalize=True, channels_first=False)
    # tensor, _ = sf.read(wav_path, start=offset, stop=stop)
    tensor = tensor[:, 0].unsqueeze(1)

    # tensor has size [length, num_channel] in which channel = 1
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
        p.grad.data.div_(denom)

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
