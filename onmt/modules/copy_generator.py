import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda
from onmt.modules.linear import XavierLinear
import math
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

    def __init__(self, hidden_size, output_size, fix_norm=False):

        super(CopyGenerator, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size

        # this constant is used to inverse the softmax function
        self.c = 0.1712209

        # gate for linear
        self.linear_copy = XavierLinear(hidden_size, 1)

        # we need a linear projector for attention
        # self.linear_attn = XavierLinear(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

        stdv = 1. / math.sqrt(self.linear.weight.size(1))

        torch.nn.init.uniform_(self.linear.weight, -stdv, stdv)

        self.fix_norm = fix_norm

    def forward(self, model_outputs):
        """

        """
        input = model_outputs['hidden']
        context = model_outputs['context']
        src = model_outputs['src']

        fix_norm = self.fix_norm
        tlen, bsz, hsize = input.size()

        #
        # batch_by_tlen, _ = input.size()
        # batch_by_tlen_, src_len = attn.size()
        # src_len_, batch, vocab_size = src_map.size()

        """ Probability of copying p(z=1) batch. """
        copy_prob = torch.sigmoid(self.linear_copy(input))  # T x B x 1

        """ probabilities from the model output """

        if not fix_norm:
            logits = self.linear(input)
        else:
            normalized_weights = F.normalize(self.linear.weight, dim=-1)
            normalized_bias = self.linear.bias
            logits = F.linear(input, normalized_weights, normalized_bias)

        prob = F.softmax(logits.float(), dim=-1, dtype=torch.float32)
        p_g = torch.mul(prob, 1 - copy_prob)  # tlen x B x V

        """ probabilities from copy """
        query = input.transpose(0, 1)
        keys = context.transpose(0, 1)  # B x slen x H

        attn_score = torch.bmm(query, keys.transpose(1, 2))  # B x tlen x slen
        src_mask = src.eq(onmt.constants.PAD).unsqueeze(1)  # B x s_len

        attn_score = attn_score.float().masked_fill_(src_mask, -float('inf')).type_as(attn_score)
        attns = F.softmax(attn_score.float(), dim=-1)  # B x tlen x slen

        p_c = torch.mul(attns.transpose(0, 1), copy_prob)  # tlen x B x slen

        src_indices = src.unsqueeze(0).expand_as(p_c)
        # add the probabilities into the positions directly
        p_g.scatter_add_(2, src_indices, p_c)
        # p_c = torch.bmm(mul_attn, src)

        # mul_attn = torch.mul(attn, copy_prob.expand_as(attn)).view(-1, batch, slen) # tlen_, batch, src_len
        # p_c = torch.bmm(mul_attn.transpose(0, 1),
        #                       src_map.transpose(0, 1)).transpose(0, 1) # tlen, batch, vocab_size

        # revert the softmax function to get logits
        output = torch.log(p_g) + self.c

        # the logits is then used in the normal loss function
        return output
