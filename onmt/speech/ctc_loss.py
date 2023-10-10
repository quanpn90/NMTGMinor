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

        # self.vocab_size = vocab_size
        # self.hidden_size = hidden_size
        if padding_idx == -1:
            self.padding_idx = onmt.constants.PAD
        else:
            self.padding_idx = padding_idx

        if blank_idx == -1:
            self.blank_idx = onmt.constants.TGT_PAD
        else:
            self.blank_idx = blank_idx

        # why do we need dropout at ctc ?
        self.dropout_rate = dropout_rate

        # In case of Pytorch >= 1.7.0, CTC will be always builtin
        self.ctc_type = (
            ctc_type
            if LooseVersion(torch.__version__) < LooseVersion("1.7.0")
            else "builtin"
        )


        if self.ctc_type == "builtin":
            reduction_type = "sum" if reduce else "none"
            self.ctc_loss = torch.nn.CTCLoss(blank=onmt.constants.TGT_PAD, reduction=reduction_type, zero_infinity=True)

        else:
            raise ValueError(
                'ctc_type must be "builtin" or "warpctc": {}'.format(self.ctc_type)
            )

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

        if self.ctc_type == "builtin":

            log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)

            # Use the deterministic CuDNN implementation of CTC loss to avoid
            #  [issue#17798](https://github.com/pytorch/pytorch/issues/17798)
            with torch.backends.cudnn.flags(deterministic=True):
                loss = self.ctc_loss(log_probs, targets, ilen, olen)

            return loss

        elif self.ctc_type == "warpctc":

            return self.ctc_loss(logits, targets, ilen, olen)

        else:
            raise NotImplementedError

    def forward(self, model_outputs, targets, **kwargs):

        # context logits: T x B x V
        # targets: T x B
        logits = model_outputs['encoder_logits']

        if 'wav2vec_padding_mask' in model_outputs:
            source_mask = model_outputs['wav2vec_padding_mask'].long()
        else:
            source_mask = model_outputs['src_mask'].long()

        # target mask should be T x B
        target_mask = targets.ne(self.padding_idx)
        target_lengths = target_mask.long().sum(0)
        # print(target_lengths)

        # source mask should be B x 1 x T or B x T
        if source_mask.dim() == 3:
            input_lengths = (1 - source_mask).squeeze(1).sum(1)
        else:
            input_lengths = (1 - source_mask).sum(1)

        if self.ctc_type == 'builtin':
            # target is batch first
            targets = targets.transpose(0, 1).contiguous()
            padding_mask = targets.eq(self.padding_idx)
            targets = targets.view(-1)  # flatten [B x T]

            non_pad_indices = torch.nonzero(padding_mask.view(-1).ne(1)).squeeze(1)
            targets = targets.index_select(0, non_pad_indices)

        loss = self.compute_loss(logits, targets, input_lengths, target_lengths)

        return loss
