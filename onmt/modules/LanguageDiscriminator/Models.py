import numpy as np
import torch, math
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class LanguageDiscriminator(nn.Module):
    """A language discriminator model
       The main component is an LSTM running on top of embeddings
       (can be the output of transformer encoder)
       Then we have to take the final state

    Args:
        opt: list of options ( see train.py )
        dicts : dictionary (for source language)
    """

    def __init__(self, opt, embedding, n_language):

        super(LanguageDiscriminator, self).__init__()

        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout
        self.output_size = n_language

        self.word_lut = embedding

        self.linear_softmax = nn.Linear(self.model_size, self.output_size)

        self.fwd_lstm = nn.LSTM(self.model_size, self.model_size, 1)

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, input, input_length):
        """
        :param input: Torch.LongTensor with size T x B
        :param input_length: Torch.LongTensor with size B
        :return:
        the probability of the language that the input is in
        output size: B x n_language
        """

        # first, the input is being sorted by length (descendant order)
        sorted_length, sorted_indices = torch.sort(input_length, descending=True)

        sorted_input = torch.index_select(input, 1, sorted_indices)

        emb = self.word_lut(sorted_input)

        emb = emb.transpoes(0, 1)

        output = emb

        ltsm_outputs, (hn, cn) = self.fwd_lstm(output)

        # the first dimension is 1 (1 layer * 1 direction)
        output = hn.squeeze(0)

        # unsort the output from the result to get the output from the original indices
        unsorted_output = output.new(*output.size())
        unsorted_output.scatter_(0, sorted_indices, output)

        output = self.dropout(output)

        # project to n language output and apply linear softmax
        output = self.linear_softmax(output)
        output = nn.functional.log_softmax(output)

        return output
