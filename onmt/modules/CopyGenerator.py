import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda

import onmt


class CopyGenerator(nn.Module):
    """Generator module that additionally considers copying
    words directly from the source.
    The main idea is that we have an extended "dynamic dictionary".
    It contains `|tgt_dict|` words plus an arbitrary number of
    additional words introduced by the source sentence.
    For each source sentence we have a `src_map` that maps
    each source word to an index in `tgt_dict` if it known, or
    else to an extra word.
    The copy generator is an extended version of the standard
    generator that computse three values.
    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of instead copying a
      word from the source, computed using a bernoulli
    * :math:`p_{copy}` the probility of copying a word instead.
      taken from the attention distribution directly.
    The model returns a distribution over the extend dictionary,
    computed as
    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`
    .. mermaid::
       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O
    Args:
       input_size (int): size of input representation
       tgt_dict (Vocab): output target dictionary
    """
        
    def __init__(self, hidden size, output_size):
        
        super(CopyGenerator, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size

        # gate for linear
        self.linear_gate = nn.Linear(hidden_size, 1)
        
        # for i in dicts:
        #     vocabSize = dicts[i].size()
        #     self.outputSizes.append(vocabSize)
        #     self.inputSizes.append(inputSize)
        #
        # self.linear = onmt.modules.MultiLinear(self.inputSizes, self.outputSizes)
        # self.linear_copy = onmt.modules.MultiLinear(self.inputSizes, 1)
        #
        # self.dicts = dicts
        
    def switchID(self, tgtID):
        
        self.linear.switchID(tgtID)
        self.linear_copy.switchID(tgtID)
                            
    def forward(self, input, attn, src_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by compying
        source words.
        Args:
           hidden (`FloatTensor`): hidden outputs `[batch*tlen, input_size]`
           attn (`FloatTensor`): attn for each `[batch*tlen, input_size]`
           src_map (`FloatTensor`):
             A sparse indicator matrix mapping each source word to
             its index in the tgt dictionary.
             `[src_len, batch, dict_size]`
        We assume that the src and target share the dictionary to use this feature
        So the src index is the same as the index in the target dict 
        """
        
        batch_by_tlen, _ = input.size()
        batch_by_tlen_, src_len = attn.size()
        src_len_, batch, vocab_size = src_map.size()

        """ Probability of copying p(z=1) batch. """
        copy_prob = F.sigmoid(self.linear_copy(input))
        
        """ probabilities from the model output """
        logits = self.linear(input)
        prob = F.softmax(logits)
        p_g = torch.mul(prob,  1 - copy_prob.expand_as(prob))
        
        """ probabilities from copy """
        mul_attn = torch.mul(attn, copy_prob.expand_as(attn)).view(-1, batch, slen) # tlen_, batch, src_len
        p_c = torch.bmm(mul_attn.transpose(0, 1),
                              src_map.transpose(0, 1)).transpose(0, 1) # tlen, batch, vocab_size
        
        # added 1e-20 for numerical stability
        output = torch.log(p_g + p_c + 1e-20)
        
        # from this log probability we can use normal loss function ?
        return output
    

    #~ def __init__(self, input_size, tgt_dict):
        #~ super(CopyGenerator, self).__init__()
        #~ self.linear = nn.Linear(input_size, len(tgt_dict))
        #~ self.linear_copy = nn.Linear(input_size, 1)
        #~ self.tgt_dict = tgt_dict
#~ 
    #~ def forward(self, hidden, attn, src_map):
        
        #~ Compute a distribution over the target dictionary
        #~ extended by the dynamic dictionary implied by compying
        #~ source words.
        #~ Args:
           #~ hidden (`FloatTensor`): hidden outputs `[batch*tlen, input_size]`
           #~ attn (`FloatTensor`): attn for each `[batch*tlen, input_size]`
           #~ src_map (`FloatTensor`):
             #~ A sparse indicator matrix mapping each source word to
             #~ its index in the "extended" vocab containing.
             #~ `[src_len, batch, extra_words]`
        #~ """
        #~ # CHECKS
        #~ batch_by_tlen, _ = hidden.size()
        #~ batch_by_tlen_, slen = attn.size()
        #~ slen_, batch, cvocab = src_map.size()
        #~ aeq(batch_by_tlen, batch_by_tlen_)
        #~ aeq(slen, slen_)
#~ 
        #~ # Original probabilities.
        #~ logits = self.linear(hidden)
        #~ logits[:, self.tgt_dict.stoi[onmt.io.PAD_WORD]] = -float('inf')
        #~ prob = F.softmax(logits)
#~ 
        #~ # Probability of copying p(z=1) batch.
        #~ copy = F.sigmoid(self.linear_copy(hidden))
#~ 
        #~ # Probibility of not copying: p_{word}(w) * (1 - p(z))
        #~ out_prob = torch.mul(prob,  1 - copy.expand_as(prob))
        #~ mul_attn = torch.mul(attn, copy.expand_as(attn))
        #~ copy_prob = torch.bmm(mul_attn.view(-1, batch, slen)
                              #~ .transpose(0, 1),
                              #~ src_map.transpose(0, 1)).transpose(0, 1)
        #~ copy_prob = copy_prob.contiguous().view(-1, cvocab)
        #~ return torch.cat([out_prob, copy_prob], 1)

