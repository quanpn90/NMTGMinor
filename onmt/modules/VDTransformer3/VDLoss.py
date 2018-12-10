import onmt
import onmt.modules
import torch.nn as nn
import torch, math

from torch.nn.modules.loss import _Loss


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
        
        

class VDLoss(LossFuncBase):
    
    
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
   
        else:
            weight = torch.ones(output_size)
            weight[self.padding_idx] = 0     
            self.func = nn.NLLLoss(weight, reduction='sum')
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing

        
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
        # print(lprobs.size(), gtruth.size())
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
        
   
    def forward(self, output_dict, targets, generator=None, backward=False, tgt_mask=None, normalizer=1, kl_lambda=0.0):
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
        original_outputs = outputs
        output = dict()

        nsteps = outputs.size(0)
        batch_size = outputs.size(1)
        h_size = outputs.size(-1)

        mask = tgt_mask
       
        
        clean_input = outputs
        clean_targets = targets

        # T x B x V
        dists = generator(clean_input)
        # print(dists.size(), clean_targets.size())

        smoothed_nll, nll, n_targets = self._compute_loss(dists, clean_targets)

        goldScores = outputs.new(batch_size).zero_()

        for t in range(nsteps):
            gen_t = dists[t]
            tgt_t = targets[t].unsqueeze(1)
            # print(gen_t.size(), tgt_t.size())
            scores = gen_t.data.gather(1, tgt_t)
            smooth_scores = scores.sum(dim=-1, keepdim=True)

            mask = tgt_t.eq(onmt.Constants.PAD)
            scores.masked_fill_(mask, 0)
            smooth_scores.masked_fill_(mask, 0)

            eps_i = self.smoothing_value
            smooth_scores = (1. - self.label_smoothing)   * scores + eps_i * smooth_scores

        
            goldScores += smooth_scores.squeeze(1).type_as(goldScores)
        
        
        # processing the baseline
        with torch.no_grad():
            b = output_dict['baseline']

            b_dists = generator(b)   

            b_scores = outputs.new(batch_size).zero_()

            for t in range(nsteps):
                b_gen_t = b_dists[t]
                tgt_t = targets[t].unsqueeze(1)
                # print(gen_t.size(), tgt_t.size())
                scores = b_gen_t.data.gather(1, tgt_t)
                smooth_loss = scores.sum(dim=-1, keepdim=True)

                mask = tgt_t.eq(onmt.Constants.PAD)
                scores.masked_fill_(mask, 0)
                smooth_scores.masked_fill_(mask, 0)

                eps_i = self.smoothing_value
                smooth_scores = (1. - self.label_smoothing)  * scores + eps_i * smooth_scores
            
                b_scores += smooth_scores.squeeze(1).type_as(b_scores)
        
        
        log_q_z = output_dict['log_q_z']
        n_dists = output_dict['p_z'].probs.size(1)

        # compute R for the REINFORCE gradient
        # R = goldScores.unsqueeze(1).type_as(log_q_z)
        R_ = goldScores - b_scores
        b = b_scores
        R = R_.detach()

        # b = output_dict['baseline'].type_as(log_q_z)
        output['ce'] = goldScores.sum().item()


        # baseline for the reinforce gradient
        # b_coeff_ = 0.01
        # baseline_loss = (b - R)**2
        # baseline_loss = baseline_loss.sum() * b_coeff_ / n_dists

        # inference loss is - logQ * R
        inference_loss = - log_q_z  * R.unsqueeze(1).type_as(log_q_z) * 0.01
        inference_loss = inference_loss.sum()

        
        # KL divergence
        kl = output_dict['kl'].sum()

        # also try to increase entropy to avoid latent collapse
        q_entropy = output_dict['q_z'].entropy() # posterior entropy 
        p_entropy = output_dict['p_z'].entropy() # prior entropy 

        q_ent_loss = torch.abs(q_entropy - 1.0).sum()
        p_ent_loss = torch.abs(p_entropy - 1.0).sum()


        lambda_ = 0.1
        ent_coeff = 1.0
        # lambda_ = kl_lambda
        loss = smoothed_nll + ( kl * lambda_   / n_dists )  \
                            + inference_loss + ( (q_ent_loss + p_ent_loss) * ent_coeff / n_dists )

        
        if backward:
            loss.div(normalizer).backward()

        
            
        
        output['loss'] = loss # this is actually dangerous to keep
        output['kl'] = kl.item()
        output['nll'] = nll
        output['baseline'] = b.sum().item()
        output['R'] = R.data.sum().item()
        output['q_entropy'] = q_entropy.sum().item()
        output['p_entropy'] = p_entropy.sum().item()
        return output
