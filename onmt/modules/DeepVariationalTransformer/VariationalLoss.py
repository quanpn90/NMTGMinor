import onmt
import onmt.modules
import torch.nn as nn
import torch, math

from torch.nn.modules.loss import _Loss
from collections import defaultdict



class LossFuncBase(_Loss):

    """
    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.
    Args:
        output_size: number of words in vocabulary()
    """
    
    def __init__(self, output_size):
        super(LossFuncBase, self).__init__()
        self.output_size = output_size
        self.padding_idx = onmt.Constants.PAD
    
    def _compute_loss(self, scores, targets):
        return NotImplementedError
    
    def forward(self, dists, targets, hiddens, **kwargs):
        """
        Compute the loss. Subclass must define this method.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError
        
        

class VariationalLoss(LossFuncBase):
    
    
    """
    Standard NMT Loss Computation.
    """
    def __init__(self, output_size, opt):
        super().__init__(output_size)
        label_smoothing = opt.label_smoothing
        if label_smoothing > 0:
            # When label smoothing is turned on,
            # KL-divergence between q_{smoothed ground truth prob.}(w)
            # and p_{prob. computed by model}(w) is minimized.
            # If label smoothing value is set to zero, the loss
            # is equivalent to NLLLoss or CrossEntropyLoss.
            # All non-true labels are uniformly set to low-confidence.
            self.smoothing_value = label_smoothing / (output_size - 1)
            
            

            
        else:
            weight = torch.ones(output_size)
            weight[self.padding_idx] = 0     
            self.func = nn.NLLLoss(weight, reduction='sum')
        
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing
        self.kl_lambda = opt.var_kl_lambda

        
    def _compute_loss(self, scores, targets, smooth=True):
        
        gtruth = targets.view(-1) # batch * time
        scores = scores.view(-1, scores.size(-1)) # batch * time X vocab_size
        
        tdata = gtruth
            
        # squeeze is a trick to know if mask has dimension or not
        # ~ mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()
        # ~ likelihood = torch.gather(scores, 1, tdata.unsqueeze(1))
        # ~ tmp_ = self.one_hot.repeat(gtruth.size(0), 1)
        # ~ tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
        # ~ if mask.numel() > 0:
            # ~ likelihood.index_fill_(0, mask, 0)
            # ~ tmp_.index_fill_(0, mask, 0)
       
        # ~ gtruth = torch.autograd.Variable(tmp_, requires_grad=False)
        # ~ loss = self.func(scores, gtruth)
        # ~ loss_data = - likelihood.sum(0).item()
        
        lprobs = scores
        non_pad_mask = gtruth.ne(self.padding_idx)
        nll_loss = -lprobs.gather(1, gtruth.unsqueeze(1))[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()

        n_targets = non_pad_mask.sum()
         
        eps_i = self.smoothing_value
        loss = (1. - self.label_smoothing)   * nll_loss + eps_i * smooth_loss
        loss_data = nll_loss.item()


        nll = loss_data
        smoothed_nll = loss

        return (smoothed_nll, nll, n_targets)
        
   
    def forward(self, output_dict, targets, generator=None, backward=False, tgt_mask=None, normalizer=1, kl_lambda=0.001):
        """
        Compute the loss. Subclass must define this method.
        Args:
             
            outputs: the predictive output from the model. time x batch x vocab_size
                                                    or time x batch x hidden_size 
            target: the validate target to compare output with. time x batch
            generator: in case we want to save memory and 
            **kwargs(optional): additional info for computing loss.
        """
        
        outputs = output_dict['hiddens']
        p_z = output_dict['p_z']
        q_z = output_dict['q_z']
        original_outputs = outputs
        batch_size = outputs.size(1)
        h_size = outputs.size(-1)
        mask = tgt_mask
        # flatten the output
        outputs = outputs.contiguous().view(-1, outputs.size(-1))
        targets = targets.view(-1)
        
        p_entropy = sum([p_z_.entropy().sum().item() for p_z_ in p_z])
        q_entropy = sum([q_z_.entropy().sum().item() for q_z_ in q_z])
        
        if mask is not None:
            """ We remove all positions with PAD 
                to save memory on unwanted positions
            """
            flattened_mask = mask.view(-1)
            
            non_pad_indices = torch.nonzero(flattened_mask).squeeze(1)
            
            clean_input = outputs.index_select(0, non_pad_indices)
            
            clean_targets = targets.index_select(0, non_pad_indices)
        
        else:
            clean_input = outputs
            clean_targets = targets
        
        dists = generator(clean_input)
        
        smoothed_nll, nll, n_targets = self._compute_loss(dists, clean_targets)
        kl = output_dict['kl'].sum().div_(len(p_z))
        kl_prior = output_dict['kl_prior'].sum().div_(len(p_z))

        lambda_ = kl_lambda
        # lambda_ = self.kl_lambda # to avoid exploding (maybe)
        loss = smoothed_nll + (kl + kl_prior) * lambda_

        
        if backward:
            loss.div(normalizer).backward()
            
        output = defaultdict(lambda: None)
        output['loss'] = loss
        output['kl'] = kl.item()
        output['nll'] = nll
        output['p_entropy'] = p_entropy
        output['q_entropy'] = q_entropy
        
        return output
