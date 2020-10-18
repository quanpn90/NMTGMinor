import math
import torch
from collections import defaultdict
import onmt
import random


class Augmenter(object):
    """
    Implementation of the "Spec Augmentation" method
    (Only vertical and horizontal masking)
    """

    def __init__(self, F=7, mf=2, T=64, max_t=0.2, mt=2,
                 input_size=40, concat=4):

        self.F = F
        self.mf = mf
        self.T = T
        self.max_t = max_t
        self.mt = mt
        self.input_size = input_size
        self.concat = concat

    def augment(self, tensor):

        feat_size = tensor.size(1)
        original_len = tensor.size(0)
        reshape_size = feat_size / self.input_size

        tensor = tensor.float()
        # First we have to upsample the tensor (if it was downsampled during preprocessing)
        #         # Copy to a new storage because otherwise it is zeroed permanently`
        tensor_ = tensor.view(-1, self.input_size).new(*tensor.size()).copy_(tensor)

        for _ in range(self.mf):

            # frequency masking (second dimension)
            # 40 is the number of features (logmel)
            f = int(random.uniform(0.0, self.F))
            f_0 = int(random.uniform(0.0, 40 - f))

            tensor_[:, f_0:f_0 + f].zero_()

        for _ in range(self.mt):
            # time masking (first dimension)
            t = int(random.uniform(0.0, self.T))

            t = min(t, int(self.max_t * original_len))

            if original_len - t < 0:
                continue

            t_0 = int(random.uniform(0.0, original_len - t - 1))

            tensor_[t_0: t_0 + t].zero_()

        # reshaping back to downsampling
        tensor__ = tensor_.view(original_len, feat_size)

        return tensor__




