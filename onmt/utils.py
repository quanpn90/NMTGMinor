import logging, traceback
import os, re
import torch

def torch_persistent_save(*args, **kwargs):
    
    for i in range(3):
        try:
            return torch.save(*args, **kwargs)
        except Exception:
            if i == 2:
                logging.error(traceback.format_exc())


# Taking the mean of a tensor with size T x B x H over the time dimension
# This function takes care of the mask
def mean_with_mask(context, mask=None):

    # context dimensions: T x B x H
    # mask dimension: T x B x 1 (with unsqueeze)
    # first, we have to mask the context with zeros at the unwanted position

    if mask is not None:
        context.masked_fill_(mask, 0)

    # then take the sum over the dimension
    context_sum = torch.sum(context, dim=0, keepdim=False)

    if mask is not None:
        weights = torch.sum(1 - mask, dim=0, keepdim=False).type_as(context_sum)
        mean = context_sum.div(weights)
    else:
        mean = context_sum.div(context.size(0))

    return mean



# this function is borrowed from fairseq
# https://github.com/pytorch/fairseq/blob/master/fairseq/utils.py
def checkpoint_paths(path, pattern=r'model_ppl_(\d+).(\d+)\_e(\d+).(\d+).pt'):
    """Retrieves all checkpoints found in `path` directory.
    Checkpoints are identified by matching filename to the specified pattern. If
    the pattern contains groups, the result will be sorted by the first group in
    descending order.
    """
    pt_regexp = re.compile(pattern)
    files = os.listdir(path)
    
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




