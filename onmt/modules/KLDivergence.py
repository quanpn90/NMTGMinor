import numpy as np
import torch, math
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence_normal(p, q, p_detach=False):

    # KL divergence between two normal distributions    

    eps = 1e-9

    p_scale = p.scale 
    p_loc = p.loc

    if p_detach:
    	p_scale = p_scale.detach()
    	p_loc = p.loc.detach()

    var_ratio = ((p_scale + eps) / (q.scale + eps)).pow(2)
    t1 = ((p_loc - q.loc) / (q.scale + eps)).pow(2)

    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())


def kl_divergence_with_prior(gaussian):

    mean = gaussian.loc
    std = gaussian.scale

    kld = -0.5 * torch.sum(1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2))

    return kld