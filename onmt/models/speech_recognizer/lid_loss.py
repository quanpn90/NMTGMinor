import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

import onmt
import onmt.modules
from onmt.utils import flip


class CrossEntropyLIDLoss(_Loss):
    """
    Class for managing efficient loss computation.
    loss computations
    Users can implement their own loss computation strategy by making
    subclass of this one.
    Args:
        output_size: number of words in vocabulary()
    """

    def __init__(self, output_size, label_smoothing, fast_xentropy=False):
        super().__init__()
        self.output_size = output_size
        self.padding_idx = -1
        self.smoothing_value = label_smoothing / (output_size - 2)
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing

        # use apex fast entropy implementation
        self.fast_xentropy = fast_xentropy = False

        if self.fast_xentropy:
            try:
                from apex.contrib import xentropy as label_smoothing
                self.softmax_xentropy = label_smoothing.SoftmaxCrossEntropyLoss.apply
            except (ModuleNotFoundError, AttributeError):
                print("Fast xentropy cannot be found. Reinstalling apex with --xentropy is probably required.")
                self.softmax_xentropy = None
                self.fast_xentropy = False
        else:
            self.softmax_xentropy = None

    def forward(self, lid_logits, labels):
        """
        :param lid_logits: list of [T x B x L] logits
        :param labels: [B]
        :return:
        """
        lid_logits = torch.stack(lid_logits)
        n_layers, len_t, bsz = lid_logits.size(0), lid_logits.size(1), lid_logits.size(2)

        labels = labels.unsqueeze(0).unsqueeze(0).repeat(n_layers, len_t, 1)

        logits = lid_logits.view(-1, lid_logits.size(-1))
        gtruth = labels.view(-1)

        label_smoothing = self.label_smoothing if self.training else 0.0
        eps_i = self.smoothing_value if self.training else 0.0

        # print(logits.size(), gtruth.size())

        if not self.fast_xentropy:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)

            non_pad_mask = gtruth.ne(self.padding_idx)
            nll_loss = -lprobs.gather(1, gtruth.unsqueeze(1))[non_pad_mask]
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

            loss = (1. - label_smoothing) * nll_loss + eps_i * smooth_loss
        else:
            half_to_float = (logits.dtype == torch.half)
            loss = self.softmax_xentropy(logits, gtruth, label_smoothing, self.padding_idx, half_to_float)
            loss = loss.sum()

        return loss


