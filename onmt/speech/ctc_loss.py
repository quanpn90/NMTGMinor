from distutils.version import LooseVersion
import numpy as np
import six
import torch
import torch.nn.functional as F
import onmt


class CTC(torch.nn.Module):

    def __init__(self, vocab_size, hidden_size, dropout_rate,
                 ctc_type="builtin", reduce=True,
                 padding_idx=-1, blank_idx=0):
        super().__init__()

        if padding_idx == -1:
            self.padding_idx = onmt.constants.TGT_PAD
        else:
            self.padding_idx = padding_idx

        if blank_idx == -1:
            self.blank_idx = onmt.constants.TGT_PAD
        else:
            self.blank_idx = blank_idx

        self.ctc_type = ctc_type

        # why do we need dropout at ctc ?
        self.dropout_rate = dropout_rate

        reduction_type = "sum" if reduce else "none"
        # print("CTC loss with blank id", self.padding_idx)
        self.ctc_loss = torch.nn.CTCLoss(self.padding_idx,
                                         reduction=reduction_type)

        self.ignore_id = -1
        self.reduce = reduce

    def compute_loss(self, logits, targets, ilen, olen):
        """
        :param logits:
        :param targets:
        :param ilen:
        :param olen:
        :return:
        """

        log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)

        # Use the deterministic CuDNN implementation of CTC loss to avoid
        #  [issue#17798](https://github.com/pytorch/pytorch/issues/17798)
        with torch.backends.cudnn.flags(deterministic=True):
            # print(ilen, olen)
            loss = self.ctc_loss(log_probs, targets, ilen, olen)

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            total = targets.numel()

        return loss, total

    def forward(self, model_outputs, targets, **kwargs):

        # context logits: T x B x V
        # targets: T x B
        """
        Args:
            model_outputs:
            targets:
            **kwargs:

        Returns:
            loss: the sum of log-likelihood at each position in the sequence (summed by both batch and time dimensions)
            total number of predicted elements in the targets

        """
        logits = model_outputs['encoder_logits']

        if 'wav2vec_padding_mask' in model_outputs:
            source_mask = model_outputs['wav2vec_padding_mask'].long()
        else:
            source_mask = model_outputs['src_mask'].long()

        # target mask should be T x B
        target_mask = targets.ne(self.padding_idx)
        target_lengths = target_mask.long().sum(0)

        # source mask should be B x 1 x T or B x T
        if source_mask.dim() == 3:
            input_lengths = (1 - source_mask).squeeze(1).sum(1)
        else:
            input_lengths = (1 - source_mask).sum(1)

        # target is transposed to batch first
        targets = targets.transpose(0, 1).contiguous()
        # padding_mask = targets.eq(self.padding_idx)
        # targets = targets.view(-1)  # flatten [B x T]
        #
        # # flattened the target into 1D sequence here
        # non_pad_indices = torch.nonzero(padding_mask.view(-1).ne(1)).squeeze(1)
        # targets = targets.index_select(0, non_pad_indices)

        # print(logits.size(), targets.size(), input_lengths, target_lengths)

        loss, total = self.compute_loss(logits, targets, input_lengths, target_lengths)

        # the ctc loss is the sum of loss in each element?

        target_size_ = targets.ne(self.padding_idx).long().sum()
        target_size = torch.sum(target_lengths)

        assert target_size_.item() == target_size.item()

        return loss, target_size
