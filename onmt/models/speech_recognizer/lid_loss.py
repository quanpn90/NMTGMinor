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

    def __init__(self, output_size, label_smoothing):
        super().__init__()
        self.output_size = output_size
        self.padding_idx = -1
        self.smoothing_value = label_smoothing
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing

        # use apex fast entropy implementation
        self.fast_xentropy = fast_xentropy = False

        self.fast_xentropy = False
        try:
            import xentropy_cuda
            from onmt.modules.optimized.softmax_xentropy import SoftmaxCrossEntropyLoss
            self.softmax_xentropy = SoftmaxCrossEntropyLoss.apply
            self.fast_xentropy = True
        except (ModuleNotFoundError, AttributeError):
            self.softmax_xentropy = None
            self.fast_xentropy = False

    def forward(self, lid_logits, labels, mask):
        """
        :param lid_logits: list of [T x B x L] logits
        :param mask: [B x T]
        :return:
        """

        # here we should use logits instead of softmax/logsoftmax
        # prediction is done before the first Transformer layers

        len_t, bsz = lid_logits.size(0), lid_logits.size(1)

        # labels = labels.unsqueeze(0).unsqueeze(0).repeat(n_layers, len_t, 1)
        # labels can have three different forms:
        if labels.ndim == 1 and labels.size(0) == 1:
            labels = labels.unsqueeze(0).repeat(len_t, bsz)
        elif labels.ndim == 1 and labels.size(0) == bsz:
            labels = labels.unsqueeze(0).repeat(len_t)
        elif labels.ndim == 2:
            assert labels.size(0) == len_t, labels.size(1) == bsz
        else:
            raise NotImplementedError

        # mask should be [B x T] -> [T x B]
        mask = mask.transpose(0, 1)

        # next we need to remove padding from labels and logits
        # print(lid_logits.size(), labels.size(), mask.size())

        logits = lid_logits.view(-1, lid_logits.size(-1))
        gtruth = labels.view(-1)

        padding_mask = mask.contiguous().long()
        non_pad_indices = torch.nonzero(padding_mask.view(-1).ne(1)).squeeze(1)

        # print(logits.size(), gtruth.size(), non_pad_indices.size())
        logits = logits.index_select(0, non_pad_indices)
        gtruth = gtruth.index_select(0, non_pad_indices)

        label_smoothing = self.label_smoothing if self.training else 0.0
        eps_i = self.smoothing_value if self.training else 0.0

        # print(logits.size(), gtruth.size())

        lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)

        nll_loss = -lprobs.gather(1, gtruth.unsqueeze(1))
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()

        loss = (1. - label_smoothing) * nll_loss + eps_i * smooth_loss

        # if not self.fast_xentropy:
        #
        # else:
        #     half_to_float = (logits.dtype == torch.half)
        #     loss = self.softmax_xentropy(logits, gtruth, label_smoothing, self.padding_idx, half_to_float)
        #     loss = loss.sum()

        return loss


