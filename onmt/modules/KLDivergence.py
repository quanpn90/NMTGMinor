import numpy as np
import torch, math
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence_normal(p, q):

    # KL divergence between two normal distributions    

    eps = 1e-9

    var_ratio = ((p.scale + eps) / (q.scale + eps)).pow(2)
    t1 = ((p.loc - q.loc) / (q.scale + eps)).pow(2)

    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
