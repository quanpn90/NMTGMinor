"""NCE Implementation from https://github.com/Stonesjtu/Pytorch-NCE"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import onmt


class NCELoss(_Loss):

    def __init__(self, hidden_size, output_size, noise_ratio=256, logz=1, label_smoothing=0.0):
        super(NCELoss, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_ratio = noise_ratio
        self.padding_idx = onmt.constants.PAD
        self.smoothing_value = label_smoothing / (self.noise_ratio+1)
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing
        self.logz = 8

        try:
            from apex.contrib import xentropy as label_smoothing
            self.softmax_xentropy = label_smoothing.SoftmaxCrossEntropyLoss.apply
        except (ModuleNotFoundError, AttributeError):
            print("Fast xentropy cannot be found. Reinstalling apex with --xentropy is probably required.")
            exit()

    def forward(self, model_outputs, targets, **kwargs):

        if self.training:

            scores_model_target = model_outputs['scores_model_target'].float()
            scores_model_noise = model_outputs['scores_model_noise'].float()
            logprob_noise_target, logprob_noise_noise = \
                model_outputs['logprob_noise_target'].float(), \
                model_outputs['logprob_noise_noise'].float()

            # remove masking
            gtruth = targets.view(-1)
            non_pad_mask = gtruth.ne(self.padding_idx)
            non_pad_indices = torch.nonzero(non_pad_mask, as_tuple=False).squeeze(1)
            scores_model_target = scores_model_target.index_select(0, non_pad_indices)  # bsz x 1
            scores_model_noise = scores_model_noise.index_select(0, non_pad_indices)  # bsz x K
            logprob_noise_target = logprob_noise_target.index_select(0, non_pad_indices)  # bsz x 1
            logprob_noise_noise = logprob_noise_noise.index_select(0, non_pad_indices)  # bsz x K

            logit_model = torch.cat([scores_model_target, scores_model_noise], dim=1) - self.logz
            logit_noise = torch.cat([logprob_noise_target, logprob_noise_noise], dim=1)

            # prob_noise = logprob_noise.exp()
            # logtrue = logprob.exp() / (prob_noise + self.noise_ratio * prob_noise)
            # logtrue = torch.log(logtrue)  # bsz x [K + 1]
            logit_true = logit_model - logit_noise - math.log(self.noise_ratio)

            # e^-x = e(-log_model + logit_noise + math.log(noise))
            # 1 / p_model  * p_noise * K

            # e^-x + 1 = ( p_model + k p_noise ) / p_model

            label = torch.zeros_like(logit_true).add_(self.smoothing_value)
            label[:, 0].fill_(self.confidence)

            loss = F.binary_cross_entropy_with_logits(logit_true, label, None, pos_weight=None, reduction='sum')

            # loss.div_(self.noise_ratio + 1)

            loss_data = loss.data.item()

            # output_dict = {"loss": loss, "data": loss_data,
            #                "rev_loss": None, "rev_loss_data": None, "mirror_loss": None,
            #                "rec_loss": None, "rec_loss_data": None}

        else:
            # return cross entropy
            logits = model_outputs['logprobs']
            gtruth = targets.view(-1)
            half_to_float = (logits.dtype == torch.half)
            label_smoothing = self.label_smoothing if self.training else 0.0
            loss = self.softmax_xentropy(logits.view(-1, logits.size(-1)), gtruth,
                                         label_smoothing, self.padding_idx, half_to_float)
            loss = loss.sum()
            loss_data = loss.data.item()

        output_dict = {"loss": loss, "data": loss_data,
                       "rev_loss": None, "rev_loss_data": None, "mirror_loss": None,
                       "rec_loss": None, "rec_loss_data": None}

        return output_dict


