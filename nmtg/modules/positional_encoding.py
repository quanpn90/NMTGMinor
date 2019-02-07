import math

import torch
from torch import nn
from torch.nn import Parameter

from nmtg.sequence_generator import IncrementalState


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, batch_first=True):
        super().__init__()
        self.model_dim = model_dim
        self.batch_first = batch_first

    def forward(self, inputs, input_mask=None, incremental_state: IncrementalState = None):
        raise NotImplementedError

    def mask_outputs(self, outputs, input_mask):
        if self.batch_first:
            outputs.masked_fill_(input_mask.eq(0).unsqueeze(2), 0)
        else:
            outputs.masked_fill_(input_mask.eq(0).transpose(0, 1).unsqueeze(2), 0)


class SinusoidalPositionalEncoding(PositionalEncoding):
    """
    Adds positional embeddings to standard word embeddings
    This matches the original TensorFlow implementation at https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py.

    Args:
        model_dim:   dimension of model
        batch_first: Whether the input and output have the batch or the time dimension first

    Inputs Shapes:
        inputs: batch_size x len_seq x model_dim  or  len_seq x batch_size x model_dim
        input_mask: batch_size x len_seq regardless of batch_first

    Outputs Shapes:
        out:   batch_size x len_seq x model_dim  or  len_seq x batch_size x model_dim

    """

    def __init__(self, model_dim, batch_first=True, initial_length=512):
        super().__init__(model_dim, batch_first)
        self.register_buffer('pos_emb', None)
        self.current_length = None
        self.generate(initial_length)

    def generate(self, new_max_len):
        position = torch.arange(new_max_len, dtype=torch.float)

        num_timescales = self.model_dim // 2
        log_timescale_increment = math.log(10000) / (num_timescales - 1)
        inv_timescales = torch.exp(
            torch.arange(0, num_timescales, dtype=torch.float) * -log_timescale_increment)
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        pos_emb = torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), 1)
        self.pos_emb = pos_emb
        self.current_length = new_max_len

    def forward(self, inputs, input_mask=None, incremental_state: IncrementalState = None):
        seq_len = inputs.size(1 if self.batch_first else 0)
        needed_length = seq_len

        if incremental_state is not None:
            timestep = incremental_state.get(self, 'timestep', 0)
            needed_length = timestep + seq_len
            incremental_state.set(self, 'timestep', needed_length)

        if needed_length > self.current_length:
            self.generate(self.current_length)

        if incremental_state is None:
            emb = self.pos_emb[:needed_length, :]
        else:
            emb = self.pos_emb[needed_length - seq_len:needed_length, :]

        if not self.batch_first:
            emb = emb.unsqueeze(1)

        out = inputs + emb

        if input_mask is not None:
            self.mask_outputs(out, input_mask)

        return out


class LearnedPositionalEncoding(PositionalEncoding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, model_dim, max_length, batch_first=True):
        super().__init__(model_dim, batch_first)
        self.pos_emb = Parameter(torch.Tensor(max_length, model_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embeddings)

    def forward(self, inputs, input_mask=None, incremental_state: IncrementalState = None):
        seq_len = inputs.size(1 if self.batch_first else 0)
        needed_length = seq_len

        if incremental_state is not None:
            timestep = incremental_state.get(self, 'timestep', 0)
            needed_length = timestep + seq_len
            incremental_state.set(self, 'timestep', needed_length)

        if incremental_state is None:
            emb = self.pos_emb[:needed_length, :]
        else:
            emb = self.pos_emb[needed_length - seq_len:needed_length, :]

        if not self.batch_first:
            emb = emb.unsqueeze(1)

        out = inputs + emb

        if input_mask is not None:
            self.mask_outputs(out, input_mask)

        return out


class RNNPositionalEncoding(PositionalEncoding):
    def __init__(self, rnn):
        model_dim = rnn.hidden_size
        if rnn.bidirectional:
            model_dim *= 2
        super().__init__(model_dim, rnn.batch_first)
        self.rnn = rnn

    def forward(self, inputs, input_mask=None, incremental_state: IncrementalState = None):
        if incremental_state is not None:
            initial_state = incremental_state.get(self, 'initial_state', None)
            out, state = self.rnn(inputs, initial_state)
            incremental_state.set(self, 'initial_state', state)
        else:
            out, _ = self.rnn(inputs)

        if input_mask is not None:
            self.mask_outputs(out, input_mask)

        return out
