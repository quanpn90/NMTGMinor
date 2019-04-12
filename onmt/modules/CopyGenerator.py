import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda

import onmt


class CopyGenerator(nn.Module):

    def __init__(self, hidden_size, output_size):

        super(Generator, self).__init__()
        # self.hidden_size = hidden_size
        # self.output_size = output_size
        # # ~ self.linear = onmt.modules.Transformer.Layers.XavierLinear(hidden_size, output_size)
        # self.linear = nn.Linear(hidden_size, output_size)
        #
        # stdv = 1. / math.sqrt(self.linear.weight.size(1))
        #
        # torch.nn.init.uniform_(self.linear.weight, -stdv, stdv)
        #
        # self.linear.bias.data.zero_()

        # input that we need:

        # hidden layers from decoder
        # encoder hidden layers and source/or mask

        self.linear = nn.Linear(hidden_size, output_size)

        # self.attention = ...

        # self.gate = nn.Linear(hidden_size, 1)

    def forward(self, input, log_softmax=True):

        # added float to the end
        # print(input.size())
        logits = self.linear(input).float()

        if log_softmax:
            output = F.log_softmax(logits, dim=-1)
        else:
            output = logits
        return output

# class CopyGenerator(nn.Module):
#     """Generator module that additionally considers copying
#     words directly from the source.
#     The main idea is that we have an extended "dynamic dictionary".
#     It contains `|tgt_dict|` words plus an arbitrary number of
#     additional words introduced by the source sentence.
#     For each source sentence we have a `src_map` that maps
#     each source word to an index in `tgt_dict` if it known, or
#     else to an extra word.
#     The copy generator is an extended version of the standard
#     generator that computse three values.
#     * :math:`p_{softmax}` the standard softmax over `tgt_dict`
#     * :math:`p(z)` the probability of instead copying a
#       word from the source, computed using a bernoulli
#     * :math:`p_{copy}` the probility of copying a word instead.
#       taken from the attention distribution directly.
#     The model returns a distribution over the extend dictionary,
#     computed as
#     :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`
#     Args:
#        input_size (int): size of input representation
#        tgt_dict (Vocab): output target dictionary
#     """
#
#     def __init__(self, opt, dicts):
#
#         super(CopyGenerator, self).__init__()
#
#         # inputSize = opt.rnn_size
#         # self.inputSizes = []
#         # self.outputSizes = []
#         #
#         # for i in dicts:
#         #     vocabSize = dicts[i].size()
#         #     self.outputSizes.append(vocabSize)
#         #     self.inputSizes.append(inputSize)
#         #
#         # self.linear = onmt.modules.MultiLinear(self.inputSizes, self.outputSizes)
#         # self.linear_copy = onmt.modules.MultiLinear(self.inputSizes, 1)
#         #
#         # self.dicts = dicts
#
#
#     def forward(self, input, attn, src_map):
#         """
#         Compute a distribution over the target dictionary
#         extended by the dynamic dictionary implied by compying
#         source words.
#         Args:
#            hidden (`FloatTensor`): hidden outputs `[batch*tlen, input_size]`
#            attn (`FloatTensor`): attn for each `[batch*tlen, input_size]`
#            src_map (`FloatTensor`):
#              A sparse indicator matrix mapping each source word to
#              its index in the tgt dictionary.
#              `[src_len, batch, dict_size]`
#         We assume that the src and target share the dictionary to use this feature
#         So the src index is the same as the index in the target dict
#         """
#
#         batch_by_tlen, _ = input.size()
#         batch_by_tlen_, src_len = attn.size()
#         src_len_, batch, vocab_size = src_map.size()
#
#         """ Probability of copying p(z=1) batch. """
#         copy_prob = F.sigmoid(self.linear_copy(input))
#
#         """ probabilities from the model output """
#         logits = self.linear(input)
#         prob = F.softmax(logits)
#         p_g = torch.mul(prob,  1 - copy_prob.expand_as(prob))
#
#         """ probabilities from copy """
#         mul_attn = torch.mul(attn, copy_prob.expand_as(attn)).view(-1, batch, slen) # tlen_, batch, src_len
#         p_c = torch.bmm(mul_attn.transpose(0, 1),
#                               src_map.transpose(0, 1)).transpose(0, 1) # tlen, batch, vocab_size
#
#         # added 1e-20 for numerical stability
#         output = torch.log(p_g + p_c + 1e-20)
#
#         # from this log probability we can use normal loss function ?
#         return output

    def forward(self, input, attn, src, return_log=True):
        #
        # """ First, we want to flatten the input """
        # input = input.view(-1, input.size(-1))
        # attn = attn.view(-1, attn.size(-1))
        # batch_by_tlen_, slen = attn.size()
        # batch_size = src.size(1)
        # tlen = batch_by_tlen_ / batch_size
        #
        # # Compute the normal distribution by logits
        # logits = self.linear(input)
        #
        # p_g = F.softmax(logits)  # tlen * batch x vocab_size
        #
        # # Decide mixture coefficients
        # copy = F.sigmoid(self.linear_copy(input))
        #
        # # Probibility of word coming from the generator distribution
        # # ~ p_g = torch.mul(prob,  1 - copy.expand_as(prob)) # tlen * batch x 1
        # p_g = p_g.mul(1 - copy.expand_as(p_g))
        #
        # # Probibility of word coming from the copy pointer distribution
        # p_c = torch.mul(attn, copy.expand_as(attn))  # tlen * batch x slen
        #
        #
        # # Idea: the ids of the source words are the same as the ids of the target words
        # # So all we need to do is the scatter_add the corresponding probabilities to the output distribution
        # # and avoid large matrices multiplication
        #
        # # In_place seems to work here, but if we modify p_g then error will appear
        # p_g.scatter_add_(1, src.t().repeat(tlen, 1), p_c)
        #
        # # matrix multiplication:
        # # b x tlen x slen  *  b x slen x vocabsize
        # # transpose into tlen x b x slen
        # # ~ p_c = torch.bmm(mul_attn.view(-1 , batch_size, slen).transpose(0, 1),
        # # ~ src_map.transpose(0, 1)).transpose(0, 1)
        # # ~
        # # ~ p_c = p_c.contiguous().view(-1, p_c.size(-1))
        #
        # # ~ output = p_g + p_c
        #
        # # log probabilities
        #
        # output = p_g.clamp(min=1e-8)
        #
        # if return_log:
        #     output = torch.log(output)

        return output