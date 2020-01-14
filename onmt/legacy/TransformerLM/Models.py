import numpy as np
import torch, math
import torch.nn as nn
from onmt.models.transformers import TransformerDecodingState
from onmt.modules.base_seq2seq import NMTModel, Reconstructor, DecoderState
import onmt
from onmt.modules.dropout import embedded_dropout
#~ from onmt.modules.Checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint
from collections import defaultdict
from onmt.models.transformer_layers import PositionalEncoding, PrePostProcessing
from onmt.legacy.TransformerLM.Layers import LMDecoderLayer


def custom_layer(module):
    def custom_forward(*args):
        output = module(*args)
        return output
    return custom_forward

class TransformerLMDecoder(nn.Module):
    """Encoder in 'Attention is all you need'

    Args:
        opt
        dicts


    """

    def __init__(self, opt, dicts, positional_encoder):

        super(TransformerLMDecoder, self).__init__()

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

        if opt.time == 'positional_encoding':
            self.time_transformer = positional_encoder
        else:
            raise NotImplementedError

        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d', static=False)

        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')

        self.word_lut = nn.Embedding(dicts.size(),
                                     self.model_size,
                                     padding_idx=onmt.constants.PAD)

        self.positional_encoder = positional_encoder

        len_max = self.positional_encoder.len_max
        mask = torch.ByteTensor(np.triu(np.ones((len_max,len_max)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)

        self.build_modules()

    def build_modules(self):
        self.layer_modules = nn.ModuleList([LMDecoderLayer(self.n_heads, self.model_size,
                                                         self.dropout, self.inner_size,
                                                         self.attn_dropout,
                                                         ) for _ in range(self.layers)])

    def renew_buffer(self, new_len):

        print(new_len)
        self.positional_encoder.renew(new_len)
        mask = torch.ByteTensor(np.triu(np.ones((new_len,new_len)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)

    def forward(self, input,  **kwargs):
        """
        Inputs Shapes:
            input: (Variable) batch_size x len_tgt (wanna tranpose)
            context: (Variable) batch_size x len_src x d_model
            mask_src (Tensor) batch_size x len_src
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src

        """

        """ Embedding: batch_size x len_tgt x d_model """


        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        if self.time == 'positional_encoding':
            emb = emb * math.sqrt(self.model_size)
        """ Adding positional encoding """
        emb = self.time_transformer(emb)
        if isinstance(emb, tuple):
            emb = emb[0]
        emb = self.preprocess_layer(emb)

        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.constants.PAD).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)

        output = emb.transpose(0, 1).contiguous()

        for i, layer in enumerate(self.layer_modules):
            output, coverage = layer(output, mask_tgt)  # batch_size x len_src x d_model

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        output = self.postprocess_layer(output)

        output_dict = { 'hidden': output, 'coverage': coverage }

        # return output, None
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

        """ Adding positional encoding """
        if self.time == 'positional_encoding':
            emb = emb * math.sqrt(self.model_size)
            emb = self.time_transformer(emb, t=input.size(1))
        else:
            # prev_h = buffer[0] if buffer is None else None
            # emb = self.time_transformer(emb, prev_h)
            # buffer[0] = emb[1]
            raise NotImplementedError

        if isinstance(emb, tuple):
            emb = emb[0]
        # emb should be batch_size x 1 x dim

        # Preprocess layer: adding dropout
        emb = self.preprocess_layer(emb)

        emb = emb.transpose(0, 1)

        # batch_size x 1 x len_src


        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.constants.PAD).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
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


class TransformerLM(NMTModel):
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
        tgt = batch.get('target_input')
        tgt_out = batch.get('target_output')

        tgt = tgt.transpose(0, 1)
        decoder_output = self.decoder(tgt)

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = decoder_output['hidden']

        return output_dict

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

        return TransformerDecodingState(None, None, beam_size=beam_size, model_size=self.model_size)

