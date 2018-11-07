import onmt
import onmt.modules
import torch.nn as nn
import torch, math

from torch.nn.modules.loss import _Loss
from onmt.modules.Loss import LossFuncBase

class LossFuncBase(_Loss):

    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations
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
    def __init__(self, output_size, label_smoothing=0.0, shard_size=1):
        super().__init__(output_size)
        self.shard_split = shard_size
        
        if label_smoothing > 0:
            # When label smoothing is turned on,
            # KL-divergence between q_{smoothed ground truth prob.}(w)
            # and p_{prob. computed by model}(w) is minimized.
            # If label smoothing value is set to zero, the loss
            # is equivalent to NLLLoss or CrossEntropyLoss.
            # All non-true labels are uniformly set to low-confidence.
            self.smoothing_value = label_smoothing / (output_size - 1)
            # ~ self.func = nn.KLDivLoss(size_average=False)
            # ~ one_hot = torch.randn(1, output_size)
            # ~ one_hot.fill_(self.smoothing_value)
            # ~ one_hot[0][self.padding_idx] = 0
            # ~ self.register_buffer('one_hot', one_hot)
            

            
        else:
            weight = torch.ones(output_size)
            weight[self.padding_idx] = 0     
            self.func = nn.NLLLoss(weight, reduction='sum')
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing

        
    def cross_entropy(self, scores, targets, smooth=True):
        
        batch_size, n_steps = scores.size(0), scores.size(1)
        gtruth = targets.view(-1) # batch * time
        scores = scores.view(-1, scores.size(-1)) # batch * time X vocab_size
        
        tdata = gtruth
        non_pad_mask = gtruth.ne(self.padding_idx).float().unsqueeze(1)
        # print(scores.size())
        # print(gtruth.size())
        # print(non_pad_mask.size())

        lprobs = scores 
        nll_loss = -lprobs.gather(1, gtruth.unsqueeze(1)) * non_pad_mask
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True) * non_pad_mask
        # nll_loss = nll_loss.sum()
        # smooth_loss = smooth_loss.sum()
         
        eps_i = self.smoothing_value
        loss = (1. - self.label_smoothing) * nll_loss + eps_i * smooth_loss

        loss = loss.view(batch_size, n_steps, 1) 
        # non-reduced loss
        return nll_loss, loss

    def score_function_estimator_loss(self, decoder_inference, nll, mask):

        kl = None
        loss = None
        mask_float = mask.float()
        log_probs_p = nll.detach() * (-1.0) 
        for layer in decoder_inference:

            # each layer has an inference loss
            inference_output = decoder_inference[layer]
            log_probs_q = inference_output['log_probs'].transpose(0, 1)
            kl_div = inference_output['kl'].transpose(0, 1) * mask_float

            reinforce_loss = - log_probs_q * log_probs_p * mask_float

            if loss is None:
                loss = reinforce_loss
            else:
                loss = loss + reinforce_loss

            if kl is None:
                kl = kl_div
            else:
                kl = kl + kl_div
            # print(log_probs_q.size(), log_probs_p.size())

        loss = loss / len(decoder_inference)
        kl = kl / len(decoder_inference)

        return loss, kl
        
   
    def forward(self, outputs, targets, generator=None, backward=False, src_mask=None, tgt_mask=None, normalizer=1):
        """
        Compute the loss. Subclass must define this method.
        Args:
             
            outputs: the predictive output from the model. time x batch x vocab_size
                                                   or time x batch x hidden_size 
            target: the validate target to compare output with. time x batch
            generator: in case we want to save memory and 
            **kwargs(optional): additional info for computing loss.
        """
        output = dict()
        hidden = outputs['hidden']
        # encoder_inference = outputs['encoder_inference']
        decoder_inference = outputs['decoder_inference']
        
        original_outputs = hidden
        batch_size =  hidden.size(1)
        h_size =  hidden.size(-1)
        
        # flatten the output
        # hidden =  hidden.contiguous().view(-1,  hidden.size(-1))
        # targets = targets.view(-1)
        
        
        # if mask is not None:
        #     """ We remove all positions with PAD 
        #         to save memory on unwanted positions
        #     """
        #     flattened_mask = mask.view(-1)
            
        #     non_pad_indices = torch.nonzero(flattened_mask).squeeze(1)
            
        #     clean_input =  hidden.index_select(0, non_pad_indices)
            
        #     clean_targets = targets.index_select(0, non_pad_indices)
        
        # else:
        clean_input =  hidden
        clean_targets = targets
        
        dists = generator(clean_input)
        
        nll, cross_entropy = self.cross_entropy(dists, clean_targets)
        loss = cross_entropy.sum()
        output['nll'] = loss.item()
        # print(cross_entropy.size())
        # loss, loss_data = self._compute_loss(dists, clean_targets)
        if backward:
            scfe_loss, kl_loss = self.score_function_estimator_loss(decoder_inference, cross_entropy, tgt_mask)
            loss = loss + scfe_loss.sum() 
            loss.div(normalizer).backward()
                    
            
        output['loss'] = loss
        
        
        return output
