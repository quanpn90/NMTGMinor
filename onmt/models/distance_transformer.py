import torch
import torch.nn as nn
from onmt.models.transformer_layers import PositionalEncoding, PrePostProcessing
from onmt.models.transformer_layers import EncoderLayer, DecoderLayer
from onmt.models.transformers import TransformerEncoder, TransformerDecoder, Transformer, TransformerDecodingState
import onmt
from onmt.modules.base_seq2seq import NMTModel, Reconstructor, DecoderState
from onmt.modules.dropout import embedded_dropout
from onmt.models.transformer_layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing
from onmt.models.distance_transformer_layers import DistanceTransformerEncoderLayer, DistanceTransformerDecoderLayer
from onmt.utils import flip, expected_length
from collections import defaultdict
import math
import sys

torch.set_printoptions(threshold=500000)


class DistanceTransformerEncoder(TransformerEncoder):

    def __init__(self, opt, dicts, positional_encoder, encoder_type='text', language_embeddings=None):
        self.death_rate = opt.death_rate
        self.double_position = opt.double_position
        self.learnable_position_encoding = opt.learnable_position_encoding
        self.layer_modules = list()
        self.asynchronous = opt.asynchronous
        self.max_memory_size = opt.max_memory_size
        self.extra_context_size = opt.extra_context_size
        self.max_pos_length = opt.max_pos_length

        # build_modules will be called from the inherited constructor
        super(DistanceTransformerEncoder, self).__init__(opt, dicts, positional_encoder, encoder_type,
                                                         language_embeddings)

        # learnable position encoding
        self.positional_encoder = None

        self.d_head = self.model_size // self.n_heads

    def build_modules(self):

        e_length = expected_length(self.layers, self.death_rate)
        print("* Transformer Encoder with Distance Attention with %.2f expected layers" % e_length)

        self.layer_modules = nn.ModuleList()

        for _l in range(self.layers):
            # linearly decay the death rate
            death_r = (_l + 1.0) / self.layers * self.death_rate

            block = DistanceTransformerEncoderLayer(self.n_heads, self.model_size,
                                                    self.dropout, self.inner_size, self.attn_dropout,
                                                    variational=self.varitional_dropout, death_rate=death_r)

            self.layer_modules.append(block)

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
            bsz_first_input = input
            input = input.transpose(0, 1)
            # mask_src = input.eq(onmt.constants.PAD).unsqueeze(0)  # batch_size x src_len x 1 for broadcasting

            dec_attn_mask = bsz_first_input.eq(onmt.constants.PAD).unsqueeze(1)

            if streaming:
                raise NotImplementedError
                streaming_state = kwargs.get('streaming_state', None)
                mems = streaming_state.src_mems
                # mem_len = streaming_state.src_mems[0].size(0)
                mem_len = streaming_state.prev_src_mem_size
                input_length = kwargs.get('src_lengths', None)
                streaming_state = kwargs.get('streaming_state', None)
                mask_src = self.create_stream_mask(input, input_length, mem_len)
                mask_src = mask_src.unsqueeze(2)
            else:
                mem_len = 0
                mask_src = input.eq(onmt.constants.PAD).unsqueeze(0)  # batch_size x src_len x 1 for broadcasting
                mems = None

            emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)

            if self.double_position:
                assert input_pos is not None
                # flatten
                src_len, bsz = input_pos.size(0), input_pos.size(1)
                input_pos_ = input_pos.contiguous().view(-1).type_as(emb)
                abs_pos = self.positional_encoder(input_pos_)
                abs_pos = abs_pos.squeeze(1).view(src_len, bsz, -1)

            else:
                abs_pos = None

            """ Adding language embeddings """
            if self.use_language_embedding:
                assert self.language_embedding is not None
                # There is no "unsqueeze" here because the input is T x B x H and lang_emb is B x H
                if self.language_embedding_type in ['sum', 'all_sum']:
                    lang_emb = self.language_embedding(input_lang)
                    emb = emb + lang_emb.unsqueeze(1)

        else:
            if streaming:
                raise NotImplementedError

            if not self.cnn_downsampling:
                mask_src = input.narrow(2, 0, 1).squeeze(2).transpose(0, 1).eq(onmt.constants.PAD).unsqueeze(0)
                dec_attn_mask = input.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD).unsqueeze(1)
                input = input.narrow(2, 1, input.size(2) - 1)
                emb = self.audio_trans(input.contiguous().view(-1, input.size(2))).view(input.size(0),
                                                                                        input.size(1), -1)
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

                mask_src = long_mask[:, 0:input.size(1) * 4:4].transpose().unsqueeze(0)
                dec_attn_mask = long_mask[:, 0:input.size(1) * 4:4].unsqueeze(1)
                # the size seems to be B x T ?
                emb = input

            emb = emb.transpose(0, 1)
            input = input.transpose(0, 1)
            abs_pos = None
            mem_len = 0

        if onmt.constants.torch_version >= 1.2:
            mask_src = mask_src.bool()

        """ Scale the emb by sqrt(d_model) """
        emb = emb * math.sqrt(self.model_size)

        if self.double_position and abs_pos is not None:
            # adding position encoding
            emb = emb + abs_pos

        """ Adding positional encoding """
        qlen = input.size(0)
        klen = qlen + mem_len

        # Asynchronous positions: 2K+1 positions instead of K+1

        # because the batch dimension is lacking

        # B x T x H -> T x B x H
        context = emb

        # Apply dropout to both context and pos_emb
        context = self.preprocess_layer(context)

        for i, layer in enumerate(self.layer_modules):
            # src_len x batch_size x d_model

            if streaming:
                buffer = streaming_state.src_buffer[i]
                context, buffer = layer(context, mask_src, incremental=True, incremental_cache=buffer)
                streaming_state.src_buffer[i] = buffer
            else:
                context = layer(context, mask_src)

        # last layer norm
        context = self.postprocess_layer(context)

        output_dict = defaultdict(lambda: None, {'context': context, 'src_mask': dec_attn_mask, 'src': input})

        if streaming:
            streaming_state.prev_src_mem_size += sum(input_length.tolist())
            streaming_state.prune_source_memory(self.max_memory_size)
            # streaming_state.update_src_mems(hids, qlen)
            output_dict['streaming_state'] = streaming_state

        return output_dict


class DistanceTransformerDecoder(TransformerDecoder):

    def __init__(self, opt, dicts, positional_encoder, language_embeddings=None, ignore_source=False):

        self.death_rate = opt.death_rate
        self.double_position = opt.double_position
        self.max_memory_size = opt.max_memory_size
        self.stream_context = opt.stream_context
        self.extra_context_size = opt.extra_context_size

        # build_modules will be called from the inherited constructor
        super(DistanceTransformerDecoder, self).__init__(opt, dicts,
                                                         positional_encoder,
                                                         language_embeddings,
                                                         ignore_source,
                                                         allocate_positions=False)
        self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)
        self.d_head = self.model_size // self.n_heads
        # Parameters for the position biases
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))

    def renew_buffer(self, new_len):
        return

    def build_modules(self):

        e_length = expected_length(self.layers, self.death_rate)

        print("* Transformer Decoder with Distance Attention with %.2f expected layers" % e_length)

        self.layer_modules = nn.ModuleList()

        for l in range(self.layers):
            # linearly decay the death rate
            death_r = (l + 1.0) / self.layers * self.death_rate

            block = DistanceTransformerDecoderLayer(self.n_heads, self.model_size,
                                                    self.dropout, self.inner_size, self.attn_dropout,
                                                    variational=self.variational_dropout, death_rate=death_r)

            self.layer_modules.append(block)

    def process_embedding(self, input, input_lang=None):

        return input

    def create_context_mask(self, input, src, src_lengths, tgt_lengths, extra_context_length=0):
        """
        Generate the mask so that part of the target attends to a part of the source
        :param extra_context_length:
        :param input:
        :param src:
        :param src_lengths:
        :param tgt_lengths:
        :return:
        """

        mask = None

        if self.stream_context == 'global':
            # Global context: one target attends to everything in the source
            for (src_length, tgt_length) in zip(src_lengths, tgt_lengths):

                if mask is None:
                    prev_src_length = 0
                    prev_tgt_length = 0
                else:
                    prev_src_length, prev_tgt_length = mask.size(1), mask.size(0)

                # current sent attend to current src sent and all src in the past
                current_mask = input.new_zeros(tgt_length, src_length + prev_src_length)

                # the previous target cannot attend to the current source
                if prev_tgt_length > 0:
                    prev_mask = input.new_ones(prev_tgt_length, src_length)
                    prev_mask = torch.cat([mask, prev_mask], dim=-1)
                else:
                    prev_mask = None

                # the output mask has two parts: the prev and the current
                if prev_mask is not None:
                    mask = torch.cat([prev_mask, current_mask], dim=0)
                else:
                    mask = current_mask

        elif self.stream_context in ['local', 'limited']:
            # Local context: only attends to the aligned context
            for (src_length, tgt_length) in zip(src_lengths, tgt_lengths):

                if mask is None:
                    prev_src_length = 0
                    prev_tgt_length = 0
                else:
                    prev_src_length, prev_tgt_length = mask.size(1), mask.size(0)

                # current tgt sent attend to only current src sent
                if prev_src_length > 0:
                    current_mask = torch.cat([input.new_ones(tgt_length, prev_src_length - extra_context_length),
                                              input.new_zeros(tgt_length, src_length + extra_context_length)], dim=-1)
                else:
                    current_mask = input.new_zeros(tgt_length, src_length + extra_context_length)

                # the previous target cannot attend to the current source
                if prev_tgt_length > 0:
                    prev_mask = input.new_ones(prev_tgt_length, src_length)
                    prev_mask = torch.cat([mask, prev_mask], dim=-1)
                else:
                    prev_mask = None

                # the output mask has two parts: the prev and the current
                if prev_mask is not None:
                    mask = torch.cat([prev_mask, current_mask], dim=0)
                else:
                    mask = current_mask

        mask = mask.bool()
        return mask

    def create_self_attn_mask(self, input, tgt_lengths, prev_tgt_mem_size):
        """
        Create a mask for the target words attending to the past
        :param input:
        :param tgt_lengths:
        :param prev_tgt_mem_size:
        :return:
        """

        if self.stream_context in ['local', 'global']:
            qlen = sum(tgt_lengths.tolist())
            mlen = prev_tgt_mem_size
            klen = qlen + mlen
            mask = torch.triu(input.new_ones(qlen, klen), diagonal=1 + mlen).bool()[:, :, None]
        elif self.stream_context in ['limited']:

            # past_length = prev_tgt_mem_size
            mask = None
            # assert prev_tgt_mem_size == 0, "This model is limited and doesn't accept memory"

            for length in tgt_lengths:

                past_length = mask.size(0) if mask is not None else 0

                if past_length > 0:
                    # don't look at the past
                    past_mask = input.new_ones(length, past_length)
                else:
                    past_mask = None

                # pay attention to the past words in the current sentence
                current_mask = torch.triu(input.new_ones(length, length), diagonal=1)

                if past_mask is not None:
                    current_mask = torch.cat([past_mask, current_mask], dim=1)

                if mask is None:
                    mask = current_mask
                else:
                    no_future_mask = input.new_ones(past_length, length)
                    mask = torch.cat([mask, no_future_mask], dim=1)
                    mask = torch.cat([mask, current_mask], dim=0)

            mask = mask.bool().unsqueeze(-1)

        return mask

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
        input = input.transpose(0, 1)  # T x B
        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        emb = emb * math.sqrt(self.model_size)

        if streaming:
            src_lengths = kwargs.get("src_lengths", None)
            tgt_lengths = kwargs.get("tgt_lengths", None)
            streaming_state = kwargs.get("streaming_state")
            # mems = streaming_state.tgt_mems
            mem_len = streaming_state.prev_tgt_mem_size
            extra_context = streaming_state.extra_context
            extra_context_length = extra_context.size(0) if extra_context is not None else 0
            # mem_len = mems[0].size(0) if mems is not None else 0
        else:
            mem_len = 0
            mems = None
            extra_context = None

        if self.double_position:
            assert input_pos is not None
            tgt_len, bsz = input_pos.size(0), input_pos.size(1)
            input_pos_ = input_pos.view(-1).type_as(emb)
            abs_pos = self.positional_encoder(input_pos_).squeeze(1).view(tgt_len, bsz, -1)

            emb = emb + abs_pos

        if self.use_language_embedding:
            lang_emb = self.language_embeddings(input_lang)  # B x H or 1 x H
            if self.language_embedding_type == 'sum':
                emb = emb + lang_emb
            elif self.language_embedding_type == 'concat':
                # replace the bos embedding with the language
                bos_emb = lang_emb.expand_as(emb[0])
                emb[0] = bos_emb

                lang_emb = lang_emb.unsqueeze(0).expand_as(emb)
                concat_emb = torch.cat([emb, lang_emb], dim=-1)
                emb = torch.relu(self.projector(concat_emb))
            else:
                raise NotImplementedError

        if context is not None:
            if self.encoder_type == "audio":
                if not self.encoder_cnn_downsampling:
                    mask_src = src.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD).unsqueeze(1)
                else:
                    long_mask = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD)
                    mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
            else:
                if streaming:
                    context_attn_mask = self.create_context_mask(input, src,
                                                                 src_lengths, tgt_lengths,
                                                                 extra_context_length)
                    mask_src = context_attn_mask.unsqueeze(0)
                else:
                    mask_src = src.eq(onmt.constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        qlen = input.size(0)
        klen = qlen + mem_len
        # preparing self-attention mask. The input is either left or right aligned

        if streaming:
            dec_attn_mask = self.create_self_attn_mask(input, tgt_lengths, mem_len)
        else:
            dec_attn_mask = torch.triu(
                emb.new_ones(qlen, klen), diagonal=1 + mem_len).byte()[:, :, None]
            pad_mask = input.eq(onmt.constants.PAD).byte()  # L x B

            dec_attn_mask = dec_attn_mask + pad_mask.unsqueeze(0)
            dec_attn_mask = dec_attn_mask.gt(0)
            if onmt.constants.torch_version >= 1.2:
                dec_attn_mask = dec_attn_mask.bool()

        pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)

        output = self.preprocess_layer(emb.contiguous())

        if streaming:
            hids = [output]
            if extra_context is not None:
                context = torch.cat([extra_context, context], dim=0)
                # print(context.size(), context_attn_mask.size())

        for i, layer in enumerate(self.layer_modules):
            # batch_size x src_len x d_model output, coverage = layer(output, context, pos_emb, self.r_w_bias,
            # self.r_r_bias, dec_attn_mask, mask_src)
            # mems_i = mems[i] if mems is not None and streaming and
            # self.stream_context in ['local', 'global'] else None
            if streaming:
                buffer = streaming_state.tgt_buffer[i]
                output, coverage, buffer = layer(output, context, dec_attn_mask, context_attn_mask,
                                                 incremental=True, incremental_cache=buffer, reuse_source=False)
                streaming_state.tgt_buffer[i] = buffer
            else:
                output, coverage, _ = layer(output, context, dec_attn_mask, mask_src)
                # if streaming:
                #     hids.append(output)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        output = self.postprocess_layer(output)

        output_dict = {'hidden': output, 'coverage': coverage, 'context': context}
        output_dict = defaultdict(lambda: None, output_dict)

        if streaming:
            streaming_state.prev_tgt_mem_size += sum(tgt_lengths.tolist())
            streaming_state.prune_target_memory(self.max_memory_size)

            # if we use the extra context: keep the last context
            if self.extra_context_size > 0:
                extra_context = context[-self.extra_context_size:].detach()
                streaming_state.extra_context = extra_context

            # if self.stream_context in ['local', 'global']:
            #     streaming_state.update_tgt_mems(hids, qlen)
            output_dict['streaming_state'] = streaming_state

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

    def step_streaming(self, input, decoder_state):
        """Step function in streaming case"""

        raise NotImplementedError

        # context = decoder_state.context
        # lang = decoder_state.tgt_lang
        # streaming_state = decoder_state.streaming_state
        #
        # # for global model: push the context in
        #
        # if decoder_state.concat_input_seq:
        #     if decoder_state.input_seq is None:
        #         decoder_state.input_seq = input
        #     else:
        #         # concatenate the last input to the previous input sequence
        #         decoder_state.input_seq = torch.cat([decoder_state.input_seq, input], 0)
        #     input = decoder_state.input_seq.transpose(0, 1)  # B x T
        #
        # src = decoder_state.src.transpose(0, 1) if decoder_state.src is not None else None
        #
        # # use the last value of input to continue decoding
        # if input.size(1) > 1:
        #     input_ = input[:, -1].unsqueeze(1).transpose(0, 1)
        # else:
        #     input_ = input.transpose(0, 1)
        #
        # emb = self.word_lut(input_) * math.sqrt(self.model_size)
        # input = input.transpose(0, 1)  # B x T to T x B
        # klen = input.size(0)
        #
        # # If we start a new sentence to decode: reset the context memory
        # if klen == 1:
        #     streaming_state.reset_context_memory()
        #     if self.stream_context == 'limited':
        #         streaming_state.reset_target_memory()
        #
        # if self.use_language_embedding:
        #     lang_emb = self.language_embeddings(lang)  # B x H or 1 x H
        #     if self.language_embedding_type == 'sum':
        #         emb = emb + lang_emb
        #     elif self.language_embedding_type == 'concat':
        #         # replace the bos embedding with the language
        #         bos_emb = lang_emb.expand_as(emb[0])
        #         emb[0] = bos_emb
        #
        #         lang_emb = lang_emb.unsqueeze(0).expand_as(emb)
        #         concat_emb = torch.cat([emb, lang_emb], dim=-1)
        #         emb = torch.relu(self.projector(concat_emb))
        #     else:
        #         raise NotImplementedError
        #
        # # need to manually definte src_lengths and tgt_lengths here
        # src_lengths = torch.LongTensor([context.size(0)])
        # tgt_lengths = torch.LongTensor([1])
        #
        # if context is not None:
        #     context_attn_mask = self.create_context_mask(input, src, src_lengths, tgt_lengths)
        #     context_attn_mask = context_attn_mask.unsqueeze(0)
        # else:
        #     context_attn_mask = None
        #
        # dec_attn_mask = self.create_self_attn_mask(input, tgt_lengths, streaming_state.prev_tgt_mem_size)
        #
        # dec_attn_mask = dec_attn_mask[:, -1:, :]
        #
        # klen = 1 + streaming_state.prev_tgt_mem_size
        #
        # output = emb
        #
        # for i, layer in enumerate(self.layer_modules):
        #     # T x B x d_model
        #     buffer = streaming_state.tgt_buffer[i]
        #     # output, coverage = layer(output, context, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask, mask_src)
        #     # reuse_source = True if input.size(1) == 1 else False
        #     reuse_source = True
        #
        #     # reuse source is True in this case because we can reuse the context ...
        #     output, coverage, buffer = layer(output, context, dec_attn_mask, context_attn_mask,
        #                                      incremental=True, incremental_cache=buffer, reuse_source=reuse_source)
        #     streaming_state.tgt_buffer[i] = buffer
        #
        # output = self.postprocess_layer(output)
        #
        # streaming_state.prev_tgt_mem_size += 1
        # streaming_state.prune_target_memory(self.max_memory_size + input.size(0))
        #
        # extra_context = context[-self.extra_context_size:].detach()
        #
        # output_dict = defaultdict(lambda: None, {'hidden': output, 'coverage': coverage, 'context': context})
        # output_dict['streaming_state'] = streaming_state
        #
        # return output_dict