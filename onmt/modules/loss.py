import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

import onmt
import onmt.modules
from onmt.utils import flip


def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


class CrossEntropyLossBase(_Loss):
    """
    Class for managing  efficient loss computation.
    loss computations
    Users can implement their own loss computation strategy by making
    subclass of this one.
    Args:
        output_size: number of words in vocabulary()
    """

    def __init__(self, output_size, label_smoothing, padding_idx=0, **kwargs):
        super(CrossEntropyLossBase, self).__init__()
        self.output_size = output_size
        self.padding_idx = padding_idx
        self.smoothing_value = label_smoothing / (output_size - 2)
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing

        # use apex fast entropy implementation
        self.fast_xentropy = False
        try:
            import xentropy_cuda
            from onmt.modules.optimized.softmax_xentropy import SoftmaxCrossEntropyLoss
            self.softmax_xentropy = SoftmaxCrossEntropyLoss.apply
            self.fast_xentropy = True
        except (ModuleNotFoundError, AttributeError):
            print("[INFO] Fast xentropy cannot be found. Using PyTorch/Python based cross entropy loss.")
            self.softmax_xentropy = None
            self.fast_xentropy = False

    def _compute_loss(self, logits, targets, vocab_mask=None, softmaxed=False):
        """
        :param logits: T x B x V or B x T x V tensor (output of decoder)
        :param targets: T x B x V or B x T target tensor
        :param vocab_mask V: bool tensor or None
        :return:
        """
        label_smoothing = self.label_smoothing if self.training else 0.0

        gtruth = targets.view(-1)  # B*T
        logits = logits.view(-1, logits.size(-1))  # B*T x V

        eps_i = self.smoothing_value if self.training else 0.0
        fast_entropy = self.fast_xentropy and not softmaxed

        go_to_slow_code = False
        if not softmaxed:
            # Try the fastest softmax + loglikelihood implementation first
            if fast_entropy:
                half_to_float = (logits.dtype == torch.half)
                loss = self.softmax_xentropy(logits, gtruth, label_smoothing, self.padding_idx, half_to_float)

                # We need to return the loss data without masking bad positions
                # Otherwise the values from "low" validation perplexities cannot be trusted
                with torch.no_grad():
                    loss_data = loss.sum().data.item()

                bad_loss = torch.logical_or(torch.isinf(loss), torch.isnan(loss))
                if bad_loss.any():
                    loss.masked_fill_(bad_loss, 0)

                loss = loss.sum()
            else:
                try:
                    # Otherwise backoff to Pytorch (1.10+)
                    loss = F.cross_entropy(logits.float(), gtruth, weight=None,
                                           ignore_index=self.padding_idx, reduction='none',
                                           label_smoothing=label_smoothing)

                    with torch.no_grad():
                        loss_data = loss.sum().data.item()

                    bad_loss = torch.logical_or(torch.isinf(loss), torch.isnan(loss))
                    if bad_loss.any():
                        loss.masked_fill_(bad_loss, 0)

                    loss = loss.sum()

                except AttributeError:
                    go_to_slow_code = True
        else:
            go_to_slow_code = True

        # Then backoff to manual python code
        if go_to_slow_code:
            if not softmaxed:
                lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            else:
                lprobs = logits

            non_pad_mask = gtruth.ne(self.padding_idx)
            nll_loss = -lprobs.gather(1, gtruth.unsqueeze(1))[non_pad_mask]
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask] if eps_i > 0 else None
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum() if eps_i > 0 else None
            loss = (1. - label_smoothing) * nll_loss + eps_i * smooth_loss if eps_i > 0 else nll_loss
            loss_data = loss.data.item()

        return loss, loss_data

    def forward(self, model_outputs, targets, hiddens, **kwargs):
        return NotImplementedError


class NMTLossFunc(CrossEntropyLossBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, hidden_size, output_size, label_smoothing, mirror=False, padding_idx=0):
        """
        :param hidden_size:
        :param output_size:
        :param label_smoothing:
        :param mirror:
        :param padding_idx:
        """
        super(NMTLossFunc, self).__init__(output_size, label_smoothing, padding_idx=padding_idx)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.padding_idx = padding_idx
        self.smoothing_value = label_smoothing / output_size
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing
        self.mirror = mirror
        self.extra_modules = nn.ModuleDict()

    def set_label_smoothing(self, new_value):

        self.label_smoothing = new_value
        self.confidence = 1.0 - self.label_smoothing
        self.smoothing_value = self.label_smoothing / self.output_size

    def add_loss_function(self, loss_function, name):
        self.extra_modules[name] = loss_function

    def get_loss_function(self, name):
        return self.extra_modules[name] if name in self.extra_modules else None

    def forward(self, model_outputs, targets, model=None, vocab_mask=None, **kwargs):
        """
        Compute the loss. Subclass must define this method.
        Args:
            :param vocab_mask:
            :param model_outputs:  a dictionary containing the predictive output from the model.
                                                      time x batch x vocab_size
                                                   or time x batch x hidden_size
            :param targets: the validate target to compare output with. time x batch
            :param model: passing the model in for necessary components
        """

        softmaxed = model_outputs['softmaxed']
        outputs = model_outputs['hidden']
        # the model no longer outputs logprobs, only logits
        logits = model_outputs['logprobs']
        mirror = self.mirror

        targets_ = targets.view(-1)
        non_pad_mask = torch.nonzero(targets_.ne(self.padding_idx)).squeeze(1)
        labels = targets_.index_select(0, non_pad_mask)
        logits = logits.view(-1, logits.size(-1)).index_select(0, non_pad_mask)

        with torch.no_grad():
            if not softmaxed:
                logprobs = F.log_softmax(logits, dim=1)
            else:
                logprobs = logits
            preds = torch.argmax(logprobs, dim=1)
            correct = (preds == labels).sum().item()
            total = labels.numel()

        if mirror:
            reverse_outputs = model_outputs['reverse_hidden']
            reverse_logits = model_outputs['reverse_logprobs']

            # reverse_targets = torch.flip(targets, (0, ))  # reverse the targets in time dimension
            reverse_targets = model_outputs['reverse_target']
            alpha = 1.0

        loss, loss_data = self._compute_loss(logits, labels, vocab_mask=vocab_mask, softmaxed=softmaxed)

        total_loss = loss

        if mirror:
            reverse_loss, rev_loss_data = self._compute_loss(reverse_logits, reverse_targets, softmaxed=softmaxed)

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
            rec_loss, rec_loss_data = self._compute_loss(rec_logits, rec_targets, softmaxed=softmaxed)
            total_loss = total_loss + rec_loss
        else:
            rec_loss, rec_loss_data = None, None

        output_dict = {"loss": loss, "data": loss_data,
                       "rev_loss": rev_loss, "rev_loss_data": rev_loss_data, "mirror_loss": mirror_loss,
                       "rec_loss": rec_loss, "rec_loss_data": rec_loss_data,
                       "correct": correct, "total": total}

        # return loss, loss_data, None
        return output_dict


class MPCLoss(_Loss):

    def forward(self, model_outputs, **kwargs):

        mpc_rec = model_outputs['mpc']  # T x B x F
        original_source = model_outputs['original_source']  # T x B x F
        masked_positions = model_outputs['masked_positions']  # T x B

        mask = model_outputs['src_mask']
        mask = mask.squeeze(1).transpose(0, 1)

        # because mask is input.eq(pad) which means the pad positions are 1
        # reverse the mask so that the correct positions are 1
        flattened_mask = ~mask.view(-1)

        # get the non-zero positions and index select
        non_pad_indices = torch.nonzero(flattened_mask).squeeze(1)

        clean_rec = mpc_rec.view(-1, mpc_rec.size(-1)).index_select(0, non_pad_indices)
        clean_source = original_source.view(-1, original_source.size(-1)).index_select(0, non_pad_indices)
        clean_masked_positions = masked_positions.view(-1).index_select(0, non_pad_indices)

        clean_masked_positions = torch.nonzero(clean_masked_positions).squeeze(1)
        # print(clean_masked_positions)
        #
        # # next, choose the masked positions
        mpc_rec = clean_rec.index_select(0, clean_masked_positions)
        source = clean_source.index_select(0, clean_masked_positions)

        loss = F.l1_loss(mpc_rec.float(), source.float(), reduction='sum')
        loss_data = loss.item()

        output_dict = {"loss": loss, "data": loss_data, "numel": mpc_rec.size(0)}

        return output_dict


class ClassifierLoss(CrossEntropyLossBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, hidden_size, output_size, fast_xentropy=False):
        """
        :param hidden_size:
        :param output_size:
        :param label_smoothing:
        :param mirror:
        :param fast_xentropy:
        """
        super(ClassifierLoss, self).__init__(output_size, 0.0, fast_xentropy=fast_xentropy)
        self.hidden_size = hidden_size
        self.output_size = output_size
        # no label smoothing?
        self.smoothing_value = 0.0
        self.confidence = 1.0
        self.label_smoothing = 0.0
        self.padding_idx = -9999999999  # don't pad

    def forward(self, model_outputs, targets, model=None,
                granularity="average", **kwargs):
        """
        Compute the loss. Subclass must define this method.
        Args:
            :param granularity:
            :param model_outputs:  a dictionary containing the predictive output from the model.
                                                      time x batch x vocab_size
                                                   or time x batch x hidden_size
            :param targets: the validate target to compare output with. time x batch
            :param model: passing the model in for necessary components
        """

        softmaxed = model_outputs['softmaxed']
        # the model no longer outputs logprobs, only logits
        logits = model_outputs['logprobs']

        # assert targets.size(0) == 1
        # targets should be [1 x batch]
        t = logits.size(0)
        # targets = targets.repeat(t, 1)  # --> time x batch

        # print(logits.size())

        mask = model_outputs['src_mask']

        mask = mask.squeeze(1).transpose(0, 1)
        # because mask is input.eq(pad) which means the pad positions are 1
        # reverse the mask so that the correct positions are 1
        # flattened_mask = ~mask.view(-1)

        # get the non-zero positions and index select
        # non_pad_indices = torch.nonzero(flattened_mask).squeeze(1)

        if granularity == 'average':
            mask = mask.unsqueeze(-1)
            logits.masked_fill_(mask, 0)
            lengths = (1 - mask.long()).squeeze(-1).sum(dim=0, keepdim=False)
            clean_logits = logits.sum(dim=0, keepdim=False).div(lengths.unsqueeze(-1))
            clean_targets = targets.squeeze(0)  # --> batch
        else:
            raise NotImplementedError
        # clean_targets = targets.view(-1).index_select(0, non_pad_indices)
        # clean_logits = logits.view(-1, logits.size(-1)).index_select(0, non_pad_indices)
        #
        # # loss, loss_data = self._compute_loss(clean_logits, clean_targets, softmaxed=softmaxed)
        loss = F.cross_entropy(clean_logits.float(), clean_targets, weight=None,
                                ignore_index=-100, reduction='sum')
        loss_data = loss.item()
        #
        predictions = F.log_softmax(clean_logits.float()).topk(1, dim=1)[1].squeeze(1)
        #
        n_correct = (clean_targets.eq(predictions.long())).sum()

        output_dict = {"loss": loss, "data": loss_data, "numel": clean_targets.numel(),
                       "n_correct": n_correct}

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

