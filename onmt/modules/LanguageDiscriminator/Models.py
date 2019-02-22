import numpy as np
import torch, math
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from onmt.modules.WordDrop import embedded_dropout

# from onmt.modules.Utilities import mean_with_mask


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

        self.fwd_lstm = nn.LSTM(self.model_size, self.model_size, 2, dropout=self.dropout, bidirectional=False)

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

        # emb = self.word_lut(sorted_input)
        emb = embedded_dropout(self.word_lut, sorted_input, dropout=self.word_dropout if self.training else 0)

        emb = self.dropout(emb)

        # pack into LSTM input format
        lstm_input = pack(emb, sorted_length)

        lstm_outputs, (hn, cn) = self.fwd_lstm(lstm_input)

        # the first dimension is 1 (1 layer * 1 direction)
        # output = hn.squeeze(0)
        lstm_outputs, _ = unpack(lstm_outputs)
        sum_ = lstm_outputs.sum(dim=0, keepdim=False)
        mean_ = sum_ / sorted_length.unsqueeze(1).type_as(sum_)

        output = mean_

        # unsort the output from the result to get the output from the original indices
        unsorted_output = output.new(*output.size())
        unsorted_output[sorted_indices] = output
        # unsorted_output.scatter_(0, sorted_indices.unsqueeze(1), output)

        output = self.dropout(unsorted_output)

        # project to n language output and apply linear softmax
        output = self.linear_softmax(output.float())
        output = nn.functional.log_softmax(output, dim=-1)

        return output
