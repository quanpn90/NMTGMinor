import logging, traceback
import os, re
import torch


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


def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
             else x.new(torch.arange(x.size(i)-1, -1, -1).tolist()).long()
             for i in range(x.dim()))
    return x[inds]
