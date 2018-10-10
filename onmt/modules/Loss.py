import onmt
import onmt.modules
import torch.nn as nn
import torch, math

from torch.nn.modules.loss import _Loss

#~ class LabelSmoothedCrossEntropyCriterion(_Loss):
#~ 
    #~ def __init__(self, n_targets, eps):
        #~ super().__init__()
        #~ self.eps = eps
        #~ self.n_targets = n_targets
#~ 
    #~ @staticmethod
    #~ def add_args(parser):
        #~ """Add criterion-specific arguments to the parser."""
        #~ parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            #~ help='epsilon for label smoothing, 0 means no label smoothing')
#~ 
    #~ def forward(self, model, sample, reduce=True):
        #~ """Compute the loss for the given sample.
        #~ Returns a tuple with three elements:
        #~ 1) the loss, as a Variable
        #~ 2) the sample size, which is used as the denominator for the gradient
        #~ 3) logging outputs to display while training
        #~ """
        #~ net_output = model(**sample['net_input'])
        #~ lprobs = model.get_normalized_probs(net_output, log_probs=True)
        #~ target = sample['target'].unsqueeze(-1)
        #~ non_pad_mask = target.ne(self.padding_idx)
        #~ nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        #~ smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        #~ if reduce:
            #~ nll_loss = nll_loss.sum()
            #~ smooth_loss = smooth_loss.sum()
        #~ eps_i = self.eps / lprobs.size(-1)
        #~ loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
#~ 
        #~ sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        #~ logging_output = {
            #~ 'loss': utils.item(loss.data) if reduce else loss.data,
            #~ 'nll_loss': utils.item(nll_loss.data) if reduce else loss.data,
            #~ 'ntokens': sample['ntokens'],
            #~ 'sample_size': sample_size,
        #~ }
        #~ return loss, sample_size, logging_output


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
        
        

class NMTLossFunc(LossFuncBase):
    
    
    """
    Standard NMT Loss Computation.
    """
    def __init__(self, output_size, label_smoothing=0.0, shard_size=1):
        super(NMTLossFunc, self).__init__(output_size)
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

        
    def _compute_loss(self, scores, targets, smooth=True):
        
        gtruth = targets.view(-1) # batch * time
        scores = scores.view(-1, scores.size(-1)) # batch * time X vocab_size
        
        if self.confidence < 1: # label smoothing
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
             
            eps_i = self.smoothing_value
            loss = (1. - self.label_smoothing)   * nll_loss + eps_i * smooth_loss
            loss_data = nll_loss.item()
            
        else:
            loss = self.func(scores.float(), gtruth)
            loss_data = loss.data.item()

        return (loss, loss_data)
        
   
    def forward(self, outputs, targets, generator=None, backward=False, mask=None, normalizer=1):
        """
        Compute the loss. Subclass must define this method.
        Args:
             
            outputs: the predictive output from the model. time x batch x vocab_size
                                                   or time x batch x hidden_size 
            target: the validate target to compare output with. time x batch
            generator: in case we want to save memory and 
            **kwargs(optional): additional info for computing loss.
        """
        
        original_outputs = outputs
        batch_size = outputs.size(1)
        h_size = outputs.size(-1)
        
        # flatten the output
        outputs = outputs.contiguous().view(-1, outputs.size(-1))
        targets = targets.view(-1)
        
        
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
        
        loss, loss_data = self._compute_loss(dists, clean_targets)
        
        if backward:
            loss.div(normalizer).backward()
            
        output = dict()
        output['loss'] = loss
        output['nll'] = loss_data
        
        return output
