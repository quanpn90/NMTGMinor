import onmt
import onmt.modules
import torch.nn as nn
import torch



class LossFuncBase(nn.Module):

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
            self.func = nn.KLDivLoss(size_average=False)
            one_hot = torch.randn(1, output_size)
            one_hot.fill_(label_smoothing / (output_size - 2))
            one_hot[0][self.padding_idx] = 0
            self.register_buffer('one_hot', one_hot)
        else:
            weight = torch.ones(output_size)
            weight[self.padding_idx] = 0
            self.func = nn.NLLLoss(weight, size_average=False)
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing

        
    def _compute_loss(self, scores, targets):
        
        gtruth = targets.view(-1) # batch * time
        scores = scores.view(-1, scores.size(-1)) # batch * time X vocab_size
        
        if self.confidence < 1: # label smoothing
            tdata = gtruth.data
            
            # squeeze is a trick to know if mask has dimension or not
            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze() 
            likelihood = torch.gather(scores.data, 1, tdata.unsqueeze(1))
            tmp_ = self.one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.dim() > 0:
                likelihood.index_fill_(0, mask, 0)
                tmp_.index_fill_(0, mask, 0)
           
            gtruth = torch.autograd.Variable(tmp_, requires_grad=False)

        loss = self.func(scores, gtruth)
        if self.confidence < 1:
            loss_data = - likelihood.sum(0)
        else:
            loss_data = loss.data[0]
        

        return (loss, loss_data)
        
   
    def forward(self, outputs, targets, generator=None, backward=False):
        """
        Compute the loss. Subclass must define this method.
        Args:
             
            outputs: the predictive output from the model. time x batch x vocab_size
                                                   or time x batch x hidden_size 
            target: the validate target to compare output with. time x batch
            generator: in case we want to save memory and 
            **kwargs(optional): additional info for computing loss.
        """
        
        if generator is not None:
            outputs = torch.autograd.Variable(outputs.data, requires_grad=(backward))
        
        outputs_split = torch.split(outputs, self.shard_split)
        targets_split = torch.split(targets, self.shard_split)
        
        loss_data = 0
        for i, (outputs_t, target_t) in enumerate(zip(outputs_split, targets_split)):
            
            # if the generator is provided then transform
            if generator is not None:
                dist_t = generator(outputs_t)
            else:
                dist_t = outputs_t 
            
            # actual loss function between the predictive distribution and target
            loss_t, loss_data_t = self._compute_loss(dist_t, target_t)

            loss_data += loss_data_t[0]
            
            # backward from loss
            # note: we only compute the gradients w.r.t the outputs 
            if backward:
                loss_t.backward()
            
        grad_outputs = None if outputs.grad is None else outputs.grad.data
        
        return loss_data, grad_outputs
