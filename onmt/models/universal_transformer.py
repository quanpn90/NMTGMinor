import torch
import torch.nn as nn
from onmt.models.transformer_layers import PositionalEncoding, PrePostProcessing
from onmt.models.transformer_layers import EncoderLayer, DecoderLayer
from onmt.models.transformers import TransformerEncoder, TransformerDecoder, Transformer, TransformerDecodingState
import onmt
from onmt.modules.base_seq2seq import NMTModel, Reconstructor, DecoderState
from onmt.modules.dropout import embedded_dropout
from onmt.models.transformer_layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing
from onmt.models.universal_transformer_layers import UniversalEncoderLayer, UniversalDecoderLayer
from onmt.utils import flip, expected_length
from collections import defaultdict
import math
import sys

torch.set_printoptions(threshold=500000)


class UniversalTransformerEncoder(TransformerEncoder):

    def __init__(self, opt, dicts, positional_encoder, encoder_type='text', language_embeddings=None):
        self.death_rate = opt.death_rate
        self.double_position = opt.double_position
        self.learnable_position_encoding = opt.learnable_position_encoding
        self.layer_modules = list()
        self.asynchronous = opt.asynchronous
        self.max_memory_size = opt.max_memory_size
        self.extra_context_size = opt.extra_context_size
        self.max_pos_length = opt.max_pos_length
        self.universal_layer = None
        self.max_layers = opt.layers

        # build_modules will be called from the inherited constructor
        super(UniversalTransformerEncoder, self).__init__(opt, dicts, positional_encoder, encoder_type,
                                                          language_embeddings)

        self.positional_encoder = positional_encoder

        # learnable embeddings for each layer
        self.layer_embedding = nn.Embedding(opt.layers, opt.model_size)

        self.d_head = self.model_size // self.n_heads

    def build_modules(self):

        e_length = expected_length(self.layers, self.death_rate)
        print("* Universal Transformer Encoder with Absolute Attention with %.2f expected layers" % e_length)
        self.universal_layer = UniversalEncoderLayer(self.opt, death_rate=self.death_rate)

    def forward(self, input, input_pos=None, input_lang=None, streaming=False, **kwargs):
        """
        Inputs Shapes:
            input: batch_size x src_len (wanna tranpose)
        Outputs Shapes:
            out: batch_size x src_len x d_model
            mask_src
        """

        """ Embedding: batch_size x src_len x d_model """
        if self.input_type == "text":
            mask_src = input.eq(onmt.constants.PAD).unsqueeze(1)  # batch_size x 1 x len_src for broadcasting

            # apply switchout
            # if self.switchout > 0 and self.training:
            #     vocab_size = self.word_lut.weight.size(0)
            #     input = switchout(input, vocab_size, self.switchout)

            emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        else:
            if not self.cnn_downsampling:
                mask_src = input.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD).unsqueeze(1)
                input = input.narrow(2, 1, input.size(2) - 1)
                emb = self.audio_trans(input.contiguous().view(-1, input.size(2))).view(input.size(0),
                                                                                        input.size(1), -1)
                emb = emb.type_as(input)
            else:
                long_mask = input.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD)
                input = input.narrow(2, 1, input.size(2) - 1)

                # first resizing to fit the CNN format
                input = input.view(input.size(0), input.size(1), -1, self.channels)
                input = input.permute(0, 3, 1, 2)

                input = self.audio_trans(input)
                input = input.permute(0, 2, 1, 3).contiguous()
                input = input.view(input.size(0), input.size(1), -1)
                # print(input.size())
                input = self.linear_trans(input)

                mask_src = long_mask[:, 0:input.size(1) * 4:4].unsqueeze(1)
                # the size seems to be B x T ?
                emb = input

        mask_src = mask_src.bool()

        """ Scale the emb by sqrt(d_model) """
        emb = emb * math.sqrt(self.model_size)

        """ Adding language embeddings """
        if self.use_language_embedding:
            assert self.language_embedding is not None

            if self.language_embedding_type in ['sum', 'all_sum']:
                lang_emb = self.language_embedding(input_lang)
                emb = emb + lang_emb.unsqueeze(1)

        time_encoding = self.positional_encoder.get_positional_embeddings(emb)

        # B x T x H -> T x B x H
        context = self.preprocess_layer(emb.transpose(0, 1))

        for i in range(self.max_layers):
            layer_vector = torch.LongTensor([i]).to(emb.device)
            layer_vector = self.layer_embedding(layer_vector).unsqueeze(0)  # 1 x 1 x model_size

            context = self.universal_layer(context, time_encoding, layer_vector, mask_src)

        # last layer norm
        context = self.postprocess_layer(context)

        output_dict = defaultdict(lambda: None, {'context': context, 'src_mask': mask_src, 'src': input})

        if streaming:
            streaming_state.prev_src_mem_size += sum(input_length.tolist())
            streaming_state.prune_source_memory(self.max_memory_size)
            # streaming_state.update_src_mems(hids, qlen)
            output_dict['streaming_state'] = streaming_state

        return output_dict


class UniversalTransformerDecoder(TransformerDecoder):

    def __init__(self, opt, dicts, positional_encoder, language_embeddings=None, ignore_source=False):

        self.death_rate = opt.death_rate
        self.max_memory_size = opt.max_memory_size
        self.stream_context = opt.stream_context
        self.extra_context_size = opt.extra_context_size
        self.universal_layer = None
        opt.ignore_source = ignore_source
        self.max_layers = opt.layers

        # build_modules will be called from the inherited constructor
        super(UniversalTransformerDecoder, self).__init__(opt, dicts,
                                                          positional_encoder,
                                                          language_embeddings,
                                                          ignore_source)

        self.positional_encoder = positional_encoder
        # Parameters for the position biases
        self.layer_embeddings = nn.Embedding(opt.layers, opt.model_size)

    def renew_buffer(self, new_len):
        return

    def build_modules(self):

        e_length = expected_length(self.layers, self.death_rate)

        print("* Universal Transformer Decoder with Absolute Attention with %.2f expected layers" % e_length)

        self.universal_layer = UniversalDecoderLayer(self.opt, death_rate=self.death_rate)

    # TODO: merging forward_stream and forward
    # TODO: write a step function for encoder

    def forward(self, input, context, src, input_pos=None, input_lang=None, streaming=False, **kwargs):
        """
                Inputs Shapes:
                    input: (Variable) batch_size x len_tgt (wanna tranpose)
                    context: (Variable) batch_size x src_len x d_model
                    mask_src (Tensor) batch_size x src_len
                Outputs Shapes:
                    out: batch_size x len_tgt x d_model
                    coverage: batch_size x len_tgt x src_len

                """

        """ Embedding: batch_size x len_tgt x d_model """
        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        if self.time == 'positional_encoding':
            emb = emb * math.sqrt(self.model_size)

        if self.use_language_embedding:
            lang_emb = self.language_embeddings(input_lang)  # B x H or 1 x H
            if self.language_embedding_type == 'sum':
                emb = emb + lang_emb
            elif self.language_embedding_type == 'concat':
                # replace the bos embedding with the language
                bos_emb = lang_emb.expand_as(emb[:, 0, :])
                emb[:, 0, :] = bos_emb

                lang_emb = lang_emb.unsqueeze(1).expand_as(emb)
                concat_emb = torch.cat([emb, lang_emb], dim=-1)
                emb = torch.relu(self.projector(concat_emb))
            else:
                raise NotImplementedError

        if context is not None:
            if self.encoder_type == "audio":
                if not self.encoder_cnn_downsampling:
                    mask_src = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD).unsqueeze(1)
                else:
                    long_mask = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD)
                    mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
            else:

                mask_src = src.data.eq(onmt.constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        len_tgt = input.size(1)
        mask_tgt = torch.triu(
            emb.new_ones(len_tgt, len_tgt), diagonal=1).byte().unsqueeze(0)

        mask_tgt = mask_tgt.bool()

        time_embedding = self.positional_encoder.get_positional_embeddings(emb)

        output = self.preprocess_layer(emb.transpose(0, 1).contiguous())



        for i in range(self.max_layers):
            layer_tensor = torch.LongTensor([i]).to(output.device)
            layer_embedding = self.layer_embeddings(layer_tensor)

            output, coverage, _ = self.universal_layer(output, time_embedding, layer_embedding, context,
                                                       mask_tgt, mask_src)

        # last layer norm
        output = self.postprocess_layer(output)

        output_dict = {'hidden': output, 'coverage': coverage, 'context': context}
        output_dict = defaultdict(lambda: None, output_dict)

        return output_dict

    def step(self, input, decoder_state, streaming=False):
        """
        Inputs Shapes:
            input: (Variable) batch_size x len_tgt (wanna tranpose)
            context: (Variable) batch_size x src_len x d_model
            mask_src (Tensor) batch_size x src_len
            buffer (List of tensors) List of batch_size * len_tgt-1 * d_model for self-attention recomputing
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x src_len

        """

        if streaming:
            return self.step_streaming(input, decoder_state)

        context = decoder_state.context
        buffers = decoder_state.attention_buffers
        lang = decoder_state.tgt_lang
        mask_src = decoder_state.src_mask

        if decoder_state.concat_input_seq:
            if decoder_state.input_seq is None:
                decoder_state.input_seq = input
            else:
                # concatenate the last input to the previous input sequence
                decoder_state.input_seq = torch.cat([decoder_state.input_seq, input], 0)
            input = decoder_state.input_seq.transpose(0, 1)  # B x T

        src = decoder_state.src.transpose(0, 1) if decoder_state.src is not None else None

        # use the last value of input to continue decoding
        if input.size(1) > 1:
            input_ = input[:, -1].unsqueeze(1).transpose(0, 1)
        else:
            input_ = input.transpose(0, 1)

        """ Embedding: batch_size x 1 x d_model """
        emb = self.word_lut(input_) * math.sqrt(self.model_size)
        input = input.transpose(0, 1)
        klen = input.size(0)
        # emb = self.word_lut(input) * math.sqrt(self.model_size)

        if self.double_position:
            input_pos = torch.arange(input.size(0), dtype=emb.dtype, device=emb.device)
            input_pos = input_pos.unsqueeze(1).repeat(1, input.size(1))
            tgt_len, bsz = input_pos.size(0), input_pos.size(1)
            input_pos_ = input_pos.view(-1).type_as(emb)
            abs_pos = self.positional_encoder(input_pos_).squeeze(1).view(tgt_len, bsz, -1)
            emb = emb + abs_pos[-1:, :, :]

        if self.use_language_embedding:
            lang_emb = self.language_embeddings(lang)  # B x H

            if self.language_embedding_type in ['sum', 'all_sum']:
                emb = emb + lang_emb
            elif self.language_embedding_type == 'concat':
                if input.size(0) == 1:
                    emb[0] = lang_emb

                lang_emb = lang_emb.unsqueeze(0).expand_as(emb)
                concat_emb = torch.cat([emb, lang_emb], dim=-1)
                emb = torch.relu(self.projector(concat_emb))
            else:
                raise NotImplementedError

        # prepare position encoding
        qlen = emb.size(0)
        mlen = klen - qlen

        dec_attn_mask = torch.triu(
            emb.new_ones(qlen, klen), diagonal=1 + mlen).byte()[:, :, None]

        pad_mask = input.eq(onmt.constants.PAD).byte()  # L x B

        dec_attn_mask = dec_attn_mask + pad_mask.unsqueeze(0)
        dec_attn_mask = dec_attn_mask.gt(0)

        if onmt.constants.torch_version >= 1.2:
            dec_attn_mask = dec_attn_mask.bool()

        if context is not None:
            if self.encoder_type == "audio":
                if not self.encoder_cnn_downsampling:
                    mask_src = src.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD).unsqueeze(1)
                else:
                    long_mask = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD)
                    mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
            else:

                mask_src = src.eq(onmt.constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        output = emb.contiguous()

        for i, layer in enumerate(self.layer_modules):
            buffer = buffers[i] if i in buffers else None
            # assert (output.size(0) == 1)

            # output, coverage, buffer = layer.step(output, context, pos_emb,
            #                                       dec_attn_mask, mask_src, buffer=buffer)
            output, coverage, buffer = layer(output, context, dec_attn_mask, mask_src,
                                             incremental=True, incremental_cache=buffer)

            decoder_state.update_attention_buffer(buffer, i)

        output = self.postprocess_layer(output)
        output = output[-1].unsqueeze(0)

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = output
        output_dict['coverage'] = coverage
        output_dict['context'] = context

        return output_dict
