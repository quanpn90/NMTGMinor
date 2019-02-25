import numpy as np
import torch, math
import torch.nn as nn
import onmt
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from onmt.modules.WordDrop import embedded_dropout
from onmt.modules.Transformer.Layers import EncoderLayer, PrePostProcessing
from onmt.modules.Utilities import mean_with_mask


class LanguageDiscriminator(nn.Module):
    """A language discriminator model
       The main component is an LSTM running on top of embeddings
       (can be the context of transformer encoder)
       Then we have to take the final state

    Args:
        opt: list of options ( see train.py )
        dicts : dictionary (for source language)
    """

    def __init__(self, opt, embedding, position_encoder, n_language):

        super(LanguageDiscriminator, self).__init__()

        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        self.layers = 2
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout
        self.context_size = n_language
        self.residual_dropout = opt.residual_dropout

        # self.word_lut = embedding

        # self.position_encoder = position_encoder

        self.layer_modules = nn.ModuleList([EncoderLayer(self.n_heads, self.model_size, self.dropout,
                                                         self.inner_size, self.attn_dropout, self.residual_dropout)
                                            for _ in range(self.layers)])

        self.linear_softmax = nn.Linear(self.model_size, self.context_size)

        self.modules = nn.ModuleList()

        # self.fwd_lstm = nn.LSTM(self.model_size, self.model_size, 2, dropout=self.dropout, bidirectional=False)

        self.dropout = nn.Dropout(self.dropout)

        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')

    def forward(self, input, context):
        """
        :param input: Torch.LongTensor with size B x T
        :param context: Torch.Tensor with size B x T x H
        :return:
        the probability of the language that the input is in
        context size: B x n_language
        """

        # first, the input is being sorted by length (descendant order)
        # sorted_length, sorted_indices = torch.sort(input_length, descending=True)
        #
        # sorted_input = torch.index_select(input, 1, sorted_indices)
        #
        # # emb = self.word_lut(sorted_input)
        # emb = embedded_dropout(self.word_lut, sorted_input, dropout=self.word_dropout if self.training else 0)
        #
        # emb = self.dropout(emb)
        #
        # # pack into LSTM input format
        # lstm_input = pack(emb, sorted_length)
        #
        # lstm_contexts, (hn, cn) = self.fwd_lstm(lstm_input)
        #
        # # the first dimension is 1 (1 layer * 1 direction)
        # # context = hn.squeeze(0)
        # lstm_contexts, _ = unpack(lstm_contexts)
        # sum_ = lstm_contexts.sum(dim=0, keepdim=False)
        # mean_ = sum_ / sorted_length.unsqueeze(1).type_as(sum_)
        #
        # context = mean_
        #
        # # unsort the context from the result to get the context from the original indices
        # unsorted_context = context.new(*context.size())
        # unsorted_context[sorted_indices] = context
        # # unsorted_context.scatter_(0, sorted_indices.unsqueeze(1), context)
        #
        # context = self.dropout(unsorted_context)
        #
        # # project to n language context and apply linear softmax
        # context = self.linear_softmax(context.float())
        # context = nn.functional.log_softmax(context, dim=-1)
        # input = input.transpose(0, 1)
        # emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)

        context = self.dropout(context)

        # context = self.position_encoder(context)

        # batch_size x 1 x len_src for broadcasting
        mask_src = input.eq(onmt.Constants.PAD).unsqueeze(1)

        for i, layer in enumerate(self.layer_modules):

            context = layer(context, mask_src)  # len_src x batch_size x d_model

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the context, since the context can grow very large, being the sum of
        # a whole stack of unnormalized layer contexts.
        context = self.postprocess_layer(context)

        mask = input.eq(onmt.Constants.PAD).transpose(0, 1).unsqueeze(2)

        output = mean_with_mask(context, mask)

        output = self.dropout(output)
        #
        # # project to n language context and apply linear softmax
        output = self.linear_softmax(output.float())
        output = nn.functional.log_softmax(output, dim=-1)

        return output
