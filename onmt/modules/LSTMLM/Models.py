import numpy as np
import torch, math
import torch.nn as nn
from onmt.modules.Transformer.Models import TransformerDecodingState
from onmt.modules.BaseModel import NMTModel, Reconstructor, DecoderState
import onmt
from onmt.modules.WordDrop import embedded_dropout
#~ from onmt.modules.Checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint
from collections import defaultdict
from onmt.modules.Transformer.Layers import PositionalEncoding, PrePostProcessing
from onmt.modules.TransformerLM.Layers import LMDecoderLayer


def custom_layer(module):
    def custom_forward(*args):
        output = module(*args)
        return output
    return custom_forward

class LSTMLMDecoder(nn.Module):
    """Encoder in 'Attention is all you need'

    Args:
        opt
        dicts
    """

    def __init__(self, opt, dicts):

        super().__init__()

        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout
        self.time = opt.time
        self.encoder_type = opt.encoder_type

        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d', static=False)

        self.word_lut = nn.Embedding(dicts.size(),
                                     self.model_size,
                                     padding_idx=onmt.Constants.PAD)


        self.rnn = nn.LSTM(self.model_size, self.model_size, num_layers=3, dropout=self.dropout)

        self.postprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d', static=False)

        self.h = None
        self.c = None

    def renew_buffer(self, new_len):

        return

    def forward(self, input,  **kwargs):
        """
        Inputs Shapes:
            input: (Variable)  len_tgt x batch_size
        Outputs Shapes:
            out: len_tgt x batch_size x  d_model
        """

        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)

        emb = self.preprocess_layer(emb)

        if self.h is None:
            lstm_mem = None
        else:
            lstm_mem = (self.h.detach(), self.c.detach())

        output, (h, c) = self.rnn(emb, lstm_mem)

        output = self.postprocess_layer(output)

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = output
        output_dict['lstm_mem'] = (h, c)

        self.h = h
        self.c = c

        return output_dict

    def step(self, input, decoder_state):
        """
        Inputs Shapes:
            input: (Variable) batch_size x len_tgt (wanna tranpose)
            context: (Variable) batch_size x len_src x d_model
            mask_src (Tensor) batch_size x len_src
            buffer (List of tensors) List of batch_size * len_tgt-1 * d_model for self-attention recomputing
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src

        """
        buffers = decoder_state.attention_buffers

        if decoder_state.input_seq is None:
            decoder_state.input_seq = input
        else:
            # concatenate the last input to the previous input sequence
            decoder_state.input_seq = torch.cat([decoder_state.input_seq, input], 0)
        input = decoder_state.input_seq.transpose(0, 1)
        input_ = input[:,-1].unsqueeze(1)

        # output_buffer = list()

        # batch_size = input_.size(0)

        """ Embedding: batch_size x 1 x d_model """
        emb = self.word_lut(input_)

        if isinstance(emb, tuple):
            emb = emb[0]

        # Preprocess layer: adding dropout
        emb = self.preprocess_layer(emb)

        emb = emb.transpose(0, 1)

        # batch_size x 1 x len_src


        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.Constants.PAD).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)
        mask_tgt = mask_tgt[:, -1, :].unsqueeze(1)
        # print(mask_tgt)

        output = emb.contiguous()

        for i, layer in enumerate(self.layer_modules):

            buffer = buffers[i] if i in buffers else None
            assert(output.size(0) == 1)

            output, coverage, buffer = layer.step(output, mask_tgt,buffer=buffer)

            decoder_state.update_attention_buffer(buffer, i)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        output = self.postprocess_layer(output)

        return output, coverage


class LSTMLM(NMTModel):
    """Main model in 'Attention is all you need' """

    def __init__(self, encoder, decoder, generator=None):
        super().__init__( encoder, decoder, generator)
        self.model_size = self.decoder.model_size

    def forward(self, batch):
        """
        Inputs Shapes:
            src: len_src x batch_size
            tgt: len_tgt x batch_size

        Outputs Shapes:
            out:      batch_size*len_tgt x model_size


        """
        # we only need target for language model
        tgt = batch.get('target_input') # T x B
        tgt_out = batch.get('target_output') # T x B

        decoder_output = self.decoder(tgt)

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = decoder_output['hidden']

        return output_dict

    def reset_states(self):

        self.decoder.h = None
        self.decoder.c = None

    def step(self, input_t, decoder_state):
        """
        Decoding function:
        generate new decoder output based on the current input and current decoder state
        the decoder state is updated in the process
        :param input_t: the input word index at time t
        :param decoder_state: object DecoderState containing the buffers required for decoding
        :return: a dictionary containing: log-prob output and the attention coverage
        """

        hidden, coverage = self.decoder.step(input_t, decoder_state)

        log_prob = self.generator[0](hidden.squeeze(0))

        output_dict = defaultdict(lambda: None)

        output_dict['log_prob'] = log_prob

        return output_dict

    # print a sample
    def sample(self):

        pass


    def create_decoder_state(self, batch, beam_size=1):

        return LSTMDecodingState(None, None, beam_size=beam_size, model_size=self.model_size)


class LSTMDecodingState(TransformerDecodingState):

    def __init__(self, src, context, beam_size=1, model_size=512):

        # if audio only take one dimension since only used for mask

        self.beam_size = beam_size

        self.input_seq = None
        self.h = None
        self.c = None
        self.model_size = model_size


    def update_beam(self, beam, b, remaining_sents, idx):

        for tensor in [self.src, self.input_seq]  :

            if tensor is None:
                continue

            t_, br = tensor.size()
            sent_states = tensor.view(t_, self.beam_size, remaining_sents)[:, :, idx]

            sent_states.copy_(sent_states.index_select(
                1, beam[b].getCurrentOrigin()))

        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]

            if buffer_ is None:
                continue

            for k in buffer_:
                t_, br_, d_ = buffer_[k].size()
                sent_states = buffer_[k].view(t_, self.beam_size, remaining_sents, d_)[:, :, idx, :]

                sent_states.data.copy_(sent_states.data.index_select(
                            1, beam[b].getCurrentOrigin()))

    # in this section, the sentences that are still active are
    # compacted so that the decoder is not run on completed sentences
    def prune_complete_beam(self, active_idx, remaining_sents):

        model_size = self.model_size

        def update_active(t):
            if t is None:
                return t
            # select only the remaining active sentences
            view = t.data.view(-1, remaining_sents, model_size)
            new_size = list(t.size())
            new_size[-2] = new_size[-2] * len(active_idx) // remaining_sents
            return view.index_select(1, active_idx).view(*new_size)

        def update_active_2d(t):
            if t is None:
                return t
            view = t.view(-1, remaining_sents)
            new_size = list(t.size())
            new_size[-1] = new_size[-1] * len(active_idx) // remaining_sents
            new_t = view.index_select(1, active_idx).view(*new_size)
            return new_t

        self.context = update_active(self.context)

        self.input_seq = update_active_2d(self.input_seq)

        self.src = update_active_2d(self.src)

        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]

            for k in buffer_:
                buffer_[k] = update_active(buffer_[k])
