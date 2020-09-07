import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

import onmt
import onmt.modules
from onmt.utils import flip


class CrossEntropyLossBase(_Loss):
    """
    Class for managing efficient loss computation.
    loss computations
    Users can implement their own loss computation strategy by making
    subclass of this one.
    Args:
        output_size: number of words in vocabulary()
    """

    def __init__(self, output_size, label_smoothing, fast_xentropy=False):
        super(CrossEntropyLossBase, self).__init__()
        self.output_size = output_size
        self.padding_idx = onmt.constants.PAD
        self.smoothing_value = label_smoothing / (output_size - 2)
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing

        # use apex fast entropy implementation
        self.fast_xentropy = fast_xentropy

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

    def _compute_loss(self, logits, targets):
        """
        :param logits: T x B x V or B x T x V tensor (output of decoder)
        :param targets: T x B x V or B x T target tensor
        :return:
        """
        gtruth = targets.view(-1)  # B*T
        logits = logits.view(-1, logits.size(-1))  # B*T x V
        label_smoothing = self.label_smoothing if self.training else 0.0
        eps_i = self.smoothing_value if self.training else 0.0

        if not self.fast_xentropy:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)

            non_pad_mask = gtruth.ne(self.padding_idx)
            nll_loss = -lprobs.gather(1, gtruth.unsqueeze(1))[non_pad_mask]
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

            loss = (1. - label_smoothing) * nll_loss + eps_i * smooth_loss
            loss_data = loss.data.item()
        else:
            half_to_float = (logits.dtype == torch.half)
            loss = self.softmax_xentropy(logits, gtruth, label_smoothing, self.padding_idx, half_to_float)
            loss = loss.sum()
            loss_data = loss.data.item()

        return loss, loss_data

    def forward(self, model_outputs, targets, hiddens, **kwargs):
        return NotImplementedError


class NMTLossFunc(CrossEntropyLossBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, hidden_size, output_size, label_smoothing, mirror=False, fast_xentropy=False):
        """
        :param hidden_size:
        :param output_size:
        :param label_smoothing:
        :param mirror:
        :param fast_xentropy:
        """
        super(NMTLossFunc, self).__init__(output_size, label_smoothing, fast_xentropy=fast_xentropy)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.padding_idx = onmt.constants.PAD
        self.smoothing_value = label_smoothing / (output_size - 2)
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing
        self.mirror = mirror
        self.extra_modules = nn.ModuleDict()

    def add_loss_function(self, loss_function, name):
        self.extra_modules[name] = loss_function

    def get_loss_function(self, name):
        return self.extra_modules[name] if name in self.extra_modules else None

    def forward(self, model_outputs, targets, model=None, backward=False, normalizer=1, **kwargs):
        """
        Compute the loss. Subclass must define this method.
        Args:
            :param model_outputs:  a dictionary containing the predictive output from the model.
                                                      time x batch x vocab_size
                                                   or time x batch x hidden_size
            :param targets: the validate target to compare output with. time x batch
            :param model: passing the model in for necessary components
            :param backward: to control if we should perform backward pass (to free the graph) or not
            :param normalizer:the denominator of the loss before backward
        """

        outputs = model_outputs['hidden']
        # the model no longer outputs logprobs, only logits
        logits = model_outputs['logprobs']
        mirror = self.mirror

        if mirror:
            reverse_outputs = model_outputs['reverse_hidden']
            reverse_logits = model_outputs['reverse_logprobs']

            # reverse_targets = torch.flip(targets, (0, ))  # reverse the targets in time dimension
            reverse_targets = model_outputs['reverse_target']
            alpha = 1.0

        loss, loss_data = self._compute_loss(logits, targets)

        total_loss = loss

        if mirror:
            reverse_loss, rev_loss_data = self._compute_loss(reverse_logits, reverse_targets)

            # flip the reverse outputs so they have the same thing
            reverse_outputs = torch.flip(reverse_outputs, (0,))

            lengths = model_outputs['target_lengths']

            mirror_loss = 0

            # forward: 1 2 3 4 5 6 7 8
            # backward: 9 8 7 6 5 4 3 2 > 2 3 4 5 6 7 8 9
            # we want 1 == 3, 2 == 4, 5 == 7 etc because they predict the same output word

            fwd_mask = model_outputs['tgt_mask'].new(outputs.size(0), outputs.size(1)).fill_(0)
            bwd_mask = model_outputs['tgt_mask'].new(outputs.size(0), outputs.size(1)).fill_(0)

            for (b, length) in enumerate(lengths):
                L = length - 1
                fwd_mask[:L - 1, b].fill_(1)
                bwd_mask[1:L, b].fill_(1)

            fwd_mask = fwd_mask.view(-1)
            fwd_mask = torch.nonzero(fwd_mask).squeeze(1)
            fwd_hiddens = outputs.contiguous().view(-1, outputs.size(-1))
            fwd_hiddens = fwd_hiddens.index_select(0, fwd_mask)

            bwd_mask = bwd_mask.view(-1)
            bwd_mask = torch.nonzero(bwd_mask).squeeze(1)
            bwd_hiddens = reverse_outputs.contiguous().view(-1, reverse_outputs.size(-1))
            bwd_hiddens = bwd_hiddens.index_select(0, bwd_mask)

            mirror_loss_2 = F.mse_loss(fwd_hiddens, bwd_hiddens, reduction='sum')

            # print(mirror_loss_2.item(), mirror_loss.item())

            mirror_loss = mirror_loss_2.div(outputs.size(-1))

            total_loss = total_loss + reverse_loss + alpha * mirror_loss
            rev_loss = reverse_loss
        else:
            mirror_loss = None
            rev_loss = None
            rev_loss_data = None

        # if we also use reconstruction:
        if model_outputs['reconstruct']:

            rec_logits = model_outputs['rec_logprobs']
            rec_targets = model_outputs['rec_target']
            rec_loss, rec_loss_data = self._compute_loss(rec_logits, rec_targets)
            total_loss = total_loss + rec_loss
        else:
            rec_loss, rec_loss_data = None, None

        if backward:
            total_loss.div(normalizer).backward()

        output_dict = {"loss": loss, "data": loss_data,
                       "rev_loss": rev_loss, "rev_loss_data": rev_loss_data, "mirror_loss": mirror_loss,
                       "rec_loss": rec_loss, "rec_loss_data": rec_loss_data}

        # return loss, loss_data, None
        return output_dict


class CTCLossFunc(_Loss):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, output_size, label_smoothing=0.0):
        super(CTCLossFunc, self).__init__(output_size)
        self.ctc = nn.CTCLoss(output_size - 1, reduction='sum')

    def forward(self, model_outputs, targets, model=None, backward=False, normalizer=1, **kwargs):
        """
        Args:

            model_outputs: a dictionary containing the predictive output from the model.
                                                      time x batch x vocab_size
                                                   or time x batch x hidden_size
            targets: the validate target to compare output with. time x batch
            model: passing the model in for necessary components
            backward: to control if we should perform backward pass (to free the graph) or not
            normalizer: the denominator of the loss before backward
            **kwargs(optional): additional info for computing loss.
        """
        raise NotImplementedError

        # outputs = model_outputs['encoder']
        # original_outputs = outputs
        # batch_size = outputs.size(1)
        # h_size = outputs.size(-1)
        #
        # source_mask = model_outputs['src_mask']
        # target_mask = model_outputs['tgt_mask']
        #
        # target_length = target_mask.sum(0)
        # if source_mask.dim() == 3:
        #     input_length = (1-source_mask).squeeze(1).sum(1)
        # else:
        #     input_length = (1-source_mask).sum(1)
        #
        # # remove elements with more targets than input
        # comp = torch.lt(target_length,input_length)
        # target_length = target_length.index_select(0,comp.nonzero().squeeze())
        # input_length = input_length.index_select(0,comp.nonzero().squeeze())
        # outputs = outputs.index_select(1,comp.nonzero().squeeze())
        # targets = targets.index_select(1,comp.nonzero().squeeze())
        #
        # # flatten the output
        # size = outputs.size()
        # outputs = outputs.contiguous().view(-1, outputs.size(-1))
        #
        # clean_input = outputs
        #
        # # dists = generator(outputs)
        # if model is not None:
        #     # the 'second' generator is the encoder softmax one
        #     dists = model.generator[1](clean_input)
        # else:
        #     dists = clean_input
        #
        # # reshape back to 3D for CTC
        # dists = dists.view(size[0], size[1], -1)
        #
        # loss = self.ctc(dists,targets.transpose(0,1), input_length, target_length)
        #
        # loss_data = loss.data.item()
        #
        # # if not numpy.isfinite(loss_data):
        # #     print("Input:", input_length)
        # #     print("Target:", target_length)
        # #     print("Compare:", comp)
        # #     print("Selected:", comp.nonzero().squeeze().size())
        # #     loss = torch.zeros_like(loss)
        # #     loss_data = loss.data.item()
        #
        # if backward:
        #     loss.div(normalizer).backward()
        #
        # output_dict = {"loss": loss, "data": loss_data}
        # return output_dict
        # return loss,loss_data, None


class NMTAndCTCLossFunc(_Loss):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, output_size, label_smoothing=0.0, ctc_weight=0.0):
        super(NMTAndCTCLossFunc, self).__init__(output_size)
        self.ctc_weight = ctc_weight
        self.ce_loss = NMTLossFunc(output_size, label_smoothing)
        self.ctc_loss = CTCLossFunc(output_size + 1, label_smoothing)

    def forward(self, model_outputs, targets, model=None, backward=False, normalizer=1, **kwargs):
        """
        Args:

            model_outputs: a dictionary containing the predictive output from the model.
                                                      time x batch x vocab_size
                                                   or time x batch x hidden_size
            targets: the validate target to compare output with. time x batch
            model: passing the model in for necessary components
            backward: to control if we should perform backward pass (to free the graph) or not
            normalizer: the denominator of the loss before backward
            **kwargs(optional): additional info for computing loss.
        """
        ce_loss = self.ce_loss(model_outputs, targets, model, False, normalizer)
        ctc_loss = self.ctc_loss(model_outputs, targets, model, False, normalizer)

        loss = self.ctc_weight * ctc_loss['loss'] + (1 - self.ctc_weight) * ce_loss['loss']
        loss_data = self.ctc_weight * ctc_loss['data'] + (1 - self.ctc_weight) * ce_loss['data']

        if not numpy.isfinite(ctc_loss['data']):
            print("CTC_Loss:", ctc_loss['data'])
            print("NMT_Loss:", ce_loss['data'])
            print("Loss:", loss_data)
            exit()

        if backward:
            loss.div(normalizer).backward()

        output_dict = {"loss": loss, "data": loss_data}

        return output_dict

    def cuda(self):
        self.ce_loss = self.ce_loss.cuda()
        self.ctc_loss = self.ctc_loss.cuda()
        return self


class FusionLoss(CrossEntropyLossBase):

    def forward(self, model_outputs, targets, model=None, backward=False, normalizer=1, **kwargs):
        """
        Args:

            model_outputs: a dictionary containing the predictive output from the model.
                                                      time x batch x vocab_size
                                                   or time x batch x hidden_size
            targets: the validate target to compare output with. time x batch
            model: passing the model in for necessary components
            backward: to control if we should perform backward pass (to free the graph) or not
            normalizer: the denominator of the loss before backward
            **kwargs(optional): additional info for computing loss.
        """

        # in this implementation, the PRENORM algorithm is used

        tm_outputs = model_outputs['tm']['hidden']

        lm_outputs = model_outputs['lm']['hidden']

        mask = model_outputs['tgt_mask']

        # flatten the output
        tm_outputs = tm_outputs.contiguous().view(-1, tm_outputs.size(-1))
        lm_outputs = lm_outputs.contiguous().view(-1, lm_outputs.size(-1))
        targets = targets.view(-1)

        if mask is not None:
            """ We remove all positions with PAD """
            flattened_mask = mask.view(-1)

            non_pad_indices = torch.nonzero(flattened_mask).squeeze(1)

            clean_tm_input = tm_outputs.index_select(0, non_pad_indices)
            clean_lm_input = lm_outputs.index_select(0, non_pad_indices)

            clean_targets = targets.index_select(0, non_pad_indices)

        else:
            clean_tm_input = tm_outputs
            clean_lm_input = lm_outputs
            clean_targets = targets

        if model is not None:
            # the 'first' generator is the decoder softmax one

            # PRENORM algorithm from
            # https://arxiv.org/pdf/1809.00125.pdf
            # Simple Fusion: Return of the Language Model
            tm_logits = model.tm_model.generator[0](clean_tm_input, log_softmax=False)

            with torch.no_grad():
                log_lm = model.lm_model.generator[0](clean_lm_input, log_softmax=True)

            dists = F.log_softmax(tm_logits + log_lm, dim=-1)

            # # POSTNORM algorithm
            # tm_logits =  model.tm_model.generator[0](clean_tm_input, log_softmax=False)
            #
            # with torch.no_grad():
            #     lm_logits = model.lm_model.generator[0](clean_lm_input, log_softmax=False)
            #
            # dists = F.log_softmax(F.softmax(tm_logits, dim=-1) * F.softmax(lm_logits, dim=-1), dim=-1)

        else:
            raise NotImplementedError

        loss, loss_data = self._compute_loss(dists, clean_targets)

        if backward:
            loss.div(normalizer).backward()

        output_dict = {"loss": loss, "data": loss_data}

        return output_dict
