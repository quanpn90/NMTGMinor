import onmt
import onmt.modules
import torch.nn as nn
import torch, math

from torch.nn.modules.loss import _Loss
from onmt.modules.Loss import NMTLossFunc
from collections import defaultdict


class NMTL2Loss(NMTLossFunc):
    """
    Standard NMT Loss Computation.
    """

    def forward(self, output_dict, targets, model=None, backward=False,
                tgt_mask=None, normalizer=1, params=None, **kwargs):
        """
        Compute the loss. Subclass must define this method.
        Args:
            output_dict: the predictive output from the model. time x batch x vocab_size
                                                   or time x batch x hidden_size
            targets: the validate target to compare output with. time x batch
            backward
            tgt_mask: for masking the target (saving memory)
            generator: in case we want to save memory and
            normalizer: the denomination term of the loss
            params: a dictionary of additional parameters required for computing loss function
            **kwargs(optional): additional info for computing loss.
        """

        outputs = output_dict['hiddens']
        tgt_outputs = output_dict['tgt_hiddens']
        mask = tgt_mask
        # flatten the output
        outputs = outputs.contiguous().view(-1, outputs.size(-1))
        tgt_outputs = tgt_outputs.contiguous().view(-1, tgt_outputs.size(-1))
        targets = targets.view(-1)

        if params is None:
            params = defaultdict(lambda: 0.0)

        if mask is not None:
            """ We remove all positions with PAD 
                to save memory on unwanted positions
            """
            flattened_mask = mask.view(-1)

            non_pad_indices = torch.nonzero(flattened_mask).squeeze(1)

            clean_output_from_src = outputs.index_select(0, non_pad_indices)

            clean_targets = targets.index_select(0, non_pad_indices)

            clean_output_from_tgt = tgt_outputs.index_select(0, non_pad_indices)

        else:
            clean_output_from_src = outputs
            clean_targets = targets
            clean_output_from_tgt = tgt_outputs

        log_probs_from_src = model.generator(clean_output_from_src)

        dists_from_src = torch.distributions.categorical.Categorical(logits=log_probs_from_src)

        log_probs_from_tgt = model.tgt_generator(clean_output_from_tgt)

        dists_from_tgt = torch.distributions.categorical.Categorical(logits=log_probs_from_tgt)

        loss, loss_data = self._compute_loss(log_probs_from_src, clean_targets)

        # l2_loss = (clean_output_from_src.float() - clean_output_from_tgt.float()) ** 2
        # l2_loss =

        l2_loss = 0.5 * torch.distributions.kl.kl_divergence(dists_from_src, dists_from_tgt).sum() + \
                  0.5 * torch.distributions.kl.kl_divergence(dists_from_tgt, dists_from_src).sum()
        # l2_loss = (log_probs_from_src - log_probs_from_tgt) ** 2
        l2_loss = l2_loss.sum()
        loss = loss + params['l2'] * l2_loss

        if backward:
            loss.div(normalizer).backward()

        output = defaultdict(lambda: None)
        output['loss'] = loss
        output['nll'] = loss_data
        output['l2_target'] = l2_loss.item()

        return output
