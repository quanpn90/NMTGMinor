import onmt
import onmt.modules
import torch.nn as nn
import torch, math
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from onmt.utils import flip

import numpy

class CrossEntropyLossBase(_Loss):

    """
    Class for managing efficient loss computation.
    loss computations
    Users can implement their own loss computation strategy by making
    subclass of this one.
    Args:
        output_size: number of words in vocabulary()
    """

    def __init__(self, output_size, label_smoothing):
        super(CrossEntropyLossBase, self).__init__()
        self.output_size = output_size
        self.padding_idx = onmt.constants.PAD
        self.smoothing_value = label_smoothing / (output_size - 2)
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing

    def _compute_loss(self, scores, targets):

        gtruth = targets.view(-1)  # batch * time
        scores = scores.view(-1, scores.size(-1))  # batch * time X vocab_size

        lprobs = scores
        non_pad_mask = gtruth.ne(self.padding_idx)
        nll_loss = -lprobs.gather(1, gtruth.unsqueeze(1))[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()

        eps_i = self.smoothing_value
        loss = (1. - self.label_smoothing) * nll_loss + eps_i * smooth_loss
        loss_data = nll_loss.data.item()

        return loss, loss_data

    def forward(self, model_outputs, targets, hiddens, **kwargs):

        return NotImplementedError


class NMTLossFunc(CrossEntropyLossBase):
    """
    Standard NMT Loss Computation.
    """
    def __init__(self, output_size, label_smoothing, mirror=False):
        super(NMTLossFunc, self).__init__(output_size, label_smoothing)
        self.output_size = output_size
        self.padding_idx = onmt.constants.PAD
        self.smoothing_value = label_smoothing / (output_size - 2)
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing
        self.mirror = mirror

    def forward(self, model_outputs, targets, model=None, backward=False, normalizer=1, **kwargs):
        """
        Compute the loss. Subclass must define this method.
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

        outputs = model_outputs['hidden']
        logprobs = model_outputs['logprobs']
        mirror = self.mirror

        if mirror:
            reverse_outputs = model_outputs['reverse_hidden']
            reverse_logprobs = model_outputs['reverse_logprobs']
            reverse_targets = torch.flip(targets, (0, ))

            alpha = 1.0

        loss, loss_data = self._compute_loss(logprobs, targets)

        total_loss = loss

        if mirror:
            reverse_loss, _ = self._compute_loss(reverse_logprobs, reverse_targets)

            mask = model_outputs['tgt_mask']
            flattened_mask = mask.view(-1)
            non_pad_indices = torch.nonzero(flattened_mask).squeeze(1)
            # remove the pads
            outputs = outputs.contiguous().view(-1, outputs.size(-1))
            outputs = outputs.index_select(0, non_pad_indices)

            # reverse_mask = torch.flip(mask.long(), (0, )).bool
            # Here we have to flip the reverse outputs again so the states have the same order with the original seq
            reverse_outputs = torch.flip(reverse_outputs, (0, ))
            reverse_mask = mask
            flattened_reverse_mask = reverse_mask.view(-1)
            reverse_non_pad_indices = torch.nonzero(flattened_reverse_mask).squeeze(1)
            reverse_outputs = reverse_outputs.contiguous().view(-1, reverse_outputs.size(-1))
            reverse_outputs = reverse_outputs.index_select(0, reverse_non_pad_indices)

            # mirror_loss = (outputs - reverse_outputs).float() ** 2
            mirror_outputs = reverse_outputs.detach()
            mirror_loss = F.mse_loss(outputs, mirror_outputs, reduction='sum')
            total_loss = total_loss + reverse_loss + alpha * mirror_loss

        if backward:
            total_loss.div(normalizer).backward()

        output_dict = {"loss": loss, "data": loss_data}

        # return loss, loss_data, None
        return output_dict

        #

        # targets = targets.view(-1)
        # if mask is not None and logprobs.dim() < 3:
        #     """ We remove all positions with PAD """
        #     flattened_mask = mask.view(-1)
        #
        #     non_pad_indices = torch.nonzero(flattened_mask).squeeze(1)
        #
        #     clean_targets = targets.index_select(0, non_pad_indices)
        #
        # else:
        #     clean_targets = targets


class CTCLossFunc(_Loss):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, output_size, label_smoothing=0.0):
        super(CTCLossFunc, self).__init__(output_size)
        self.ctc = nn.CTCLoss(output_size-1, reduction='sum')

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

    def __init__(self, output_size, label_smoothing=0.0, ctc_weight = 0.0):
        super(NMTAndCTCLossFunc, self).__init__(output_size)
        self.ctc_weight = ctc_weight
        self.ce_loss = NMTLossFunc(output_size,label_smoothing)
        self.ctc_loss = CTCLossFunc(output_size+1,label_smoothing)

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
        ce_loss = self.ce_loss(model_outputs, targets, model,False, normalizer)
        ctc_loss = self.ctc_loss(model_outputs, targets, model, False, normalizer)

        loss = self.ctc_weight * ctc_loss['loss'] + (1-self.ctc_weight) * ce_loss['loss']
        loss_data = self.ctc_weight * ctc_loss['data'] + (1-self.ctc_weight) * ce_loss['data']

        if not numpy.isfinite(ctc_loss['data']):
            print("CTC_Loss:",ctc_loss['data'])
            print("NMT_Loss:",ce_loss['data'])
            print("Loss:",loss_data)
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
