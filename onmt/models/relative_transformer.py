import torch
import torch.nn as nn
from onmt.models.transformer_layers import PositionalEncoding, PrePostProcessing
from onmt.models.transformer_layers import EncoderLayer, DecoderLayer
from onmt.models.transformers import TransformerEncoder, TransformerDecoder
import onmt
from onmt.modules.dropout import embedded_dropout
from onmt.models.transformer_layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing
from onmt.models.relative_transformer_layers import RelativeTransformerEncoderLayer, RelativeTransformerDecoderLayer
from onmt.utils import flip, expected_length
from collections import defaultdict
import math

torch.set_printoptions(threshold=500000)


#  Positional Embedding with discrete inputs
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(SinusoidalPositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, sin_first=True, bsz=None):
        """
        :param bsz:
        :param pos_seq: sequences of RELATIVE position indices (can be negative for future)
        :param sin_first: in Attention is all you need paper, sin is first then cosin
        """
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)

        if sin_first:
            pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        else:
            pos_emb = torch.cat([sinusoid_inp.cos(), sinusoid_inp.sin()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class LearnablePostionEmbedding(nn.Module):

    def __init__(self, max_pos, demb):
        super(LearnablePostionEmbedding, self).__init__()
        self.max_pos = max(max_pos, 5000)
        # self.embedding = nn.Embedding(2 * max_pos + 1, demb)
        self.embedding = nn.Embedding(self.max_pos, demb)

    def forward(self, input):
        # pos = torch.clamp(input, 0, self.max_pos)
        # k = min((pos.size(0) - 1) // 2, self.max_pos)
        return self.embedding(input)


class RelativeTransformerEncoder(TransformerEncoder):

    def __init__(self, opt, dicts, positional_encoder, encoder_type='text', language_embeddings=None):
        self.death_rate = opt.death_rate
        self.double_position = opt.double_position
        self.learnable_position_encoding = opt.learnable_position_encoding
        self.layer_modules = list()
        self.asynchronous = opt.asynchronous
        self.max_memory_size = opt.max_memory_size

        # build_modules will be called from the inherited constructor
        super(RelativeTransformerEncoder, self).__init__(opt, dicts, positional_encoder, encoder_type,
                                                         language_embeddings)

        # learnable position encoding
        if self.learnable_position_encoding:
            self.max_pos_length = opt.max_pos_length
            # pos_emb = self.model_size // self.n_heads
            pos_emb = self.model_size
            self.positional_encoder = LearnablePostionEmbedding(self.max_pos_length, pos_emb)
            print("* Learnable position encoding with max %d positions" % self.max_pos_length)
        else:
            # or using pre-set sinusoidal
            self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)

        self.d_head = self.model_size // self.n_heads

    def build_modules(self):

        e_length = expected_length(self.layers, self.death_rate)
        print("* Transformer Encoder with Relative Attention with %.2f expected layers" % e_length)

        self.layer_modules = nn.ModuleList()

        for _l in range(self.layers):
            # linearly decay the death rate
            death_r = (_l + 1.0) / self.layers * self.death_rate

            block = RelativeTransformerEncoderLayer(self.n_heads, self.model_size,
                                                    self.dropout, self.inner_size, self.attn_dropout,
                                                    variational=self.varitional_dropout, death_rate=death_r)

            self.layer_modules.append(block)

    def create_stream_mask(self, input, input_length, prev_mem_size):

        lengths = input_length.tolist()
        mask = None

        for length in lengths:

            # the current mask should be either None or

            if mask is None:
                prev_length = 0
            else:
                prev_length = mask.size(1)

            # n current queries attend to n + p keys
            current_mask = input.new_zeros(length, length + prev_length)

            if prev_length > 0:
                prev_mask = input.new_ones(prev_length, length)
                prev_mask = torch.cat([mask, prev_mask], dim=-1)
            else:
                prev_mask = None

            if prev_mask is not None:
                mask = torch.cat([prev_mask, current_mask], dim=0)
            else:
                mask = current_mask

        if prev_mem_size > 0:
            # all current elements attend to all buffer elements
            buffer_mask = mask.new_zeros(mask.size(0), prev_mem_size)
            mask = torch.cat([buffer_mask, mask], dim=-1)

        mask = mask.bool()

        return mask
        # raise NotImplementedError

    def forward_stream(self, input, input_pos, input_lang, **kwargs):
        input_length = kwargs.get('src_lengths', None)
        streaming_state = kwargs.get('streaming_state', None)
        input = input.transpose(0, 1)

        mask_src = self.create_stream_mask(input, input_length, streaming_state.prev_src_mem_size)
        mask_src = mask_src.unsqueeze(2)

        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)  #

        """ Adding language embeddings """
        if self.use_language_embedding:
            assert self.language_embedding is not None
            # There is no "unsqueeze" here because the input is T x B x H and lang_emb is B x H
            if self.language_embedding_type in ['sum', 'all_sum']:
                lang_emb = self.language_embedding(input_lang)
                emb = emb + lang_emb.unsqueeze(1)

        """ Scale the emb by sqrt(d_model) """
        emb = emb * math.sqrt(self.model_size)

        klen = input.size(0) + streaming_state.prev_src_mem_size

        pos = torch.arange(klen - 1, -klen, -1.0, device=emb.device, dtype=emb.dtype)

        pos_emb = self.positional_encoder(pos)

        context = emb

        # Apply dropout to both context and pos_emb
        context = self.preprocess_layer(context)

        pos_emb = self.preprocess_layer(pos_emb)

        for i, layer in enumerate(self.layer_modules):
            # src_len x batch_size x d_model
            buffer = streaming_state.src_buffer[i]
            context, buffer = layer(context, pos_emb, mask_src, incremental=True, incremental_cache=buffer)
            streaming_state.src_buffer[i] = buffer

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        context = self.postprocess_layer(context)

        output_dict = defaultdict(lambda: None, {'context': context, 'src_mask': mask_src, 'src': input})
        output_dict['streaming_state'] = streaming_state
        streaming_state.prev_src_mem_size += sum(input_length.tolist())

        streaming_state.prune_source_memory(self.max_memory_size)

        # streaming_state.src_lengths = streaming_state.src_lengths[1000:]

        return output_dict

    def forward(self, input, input_pos=None, input_lang=None, streaming=False, **kwargs):
        """
        Inputs Shapes:
            input: batch_size x src_len (wanna tranpose)
        Outputs Shapes:
            out: batch_size x src_len x d_model
            mask_src
        """
        if streaming:
            return self.forward_stream(input, input_pos, input_lang, **kwargs)

        """ Embedding: batch_size x src_len x d_model """
        if self.input_type == "text":
            bsz_first_input = input
            input = input.transpose(0, 1)
            # mask_src = input.eq(onmt.constants.PAD).unsqueeze(1)  # batch_size x src_len x 1 for broadcasting
            mask_src = input.eq(onmt.constants.PAD).unsqueeze(0)
            dec_attn_mask = bsz_first_input.eq(onmt.constants.PAD).unsqueeze(1)

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

        if onmt.constants.torch_version >= 1.2:
            mask_src = mask_src.bool()

        """ Scale the emb by sqrt(d_model) """
        emb = emb * math.sqrt(self.model_size)

        if self.double_position and abs_pos is not None:
            # adding position encoding
            emb = emb + abs_pos

        """ Adding positional encoding """
        klen = input.size(0)

        # Asynchronous positions: 2K+1 positions instead of K+1
        if self.asynchronous:
            pos = torch.arange(klen - 1, -klen, -1.0, device=emb.device, dtype=emb.dtype)
        else:
            # Everything should be asynchronous now
            pos = torch.arange(klen - 1, -klen, -1.0, device=emb.device, dtype=emb.dtype)
            # pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)

        pos_emb = self.positional_encoder(pos)

        if self.learnable_position_encoding:
            raise NotImplementedError

        # if not self.learnable_position_encoding:
        #     # L x 1 x H
        #     past_pos = pos[:klen+1]
        #     future_pos = pos[-klen:]
        #     past_pos_emb = self.positional_encoder(past_pos, sin_first=True)
        #     future_pos_emb = self.positional_encoder(future_pos, sin_first=False)
        #     pos_emb = torch.cat([past_pos_emb, future_pos_emb], dim=0)
        # else:
        #     raise NotImplementedError
        #     pos = pos.long()
        #     # pos = torch.arange(klen - 1, -klen, -1.0, device=emb.device, dtype=input.dtype)
        #     # clamp the positions (all postions from afar are treated equally, maybe?)
        #     # (2L-1) x 1 x H
        #     pos_emb = self.positional_encoder(pos.unsqueeze(1))
        #     # print(pos_emb.size())

        # B x T x H -> T x B x H
        context = emb

        # Apply dropout to both context and pos_emb
        context = self.preprocess_layer(context)

        pos_emb = self.preprocess_layer(pos_emb)

        # print(context.size(), mask_src.size())

        for i, layer in enumerate(self.layer_modules):
            # src_len x batch_size x d_model
            context = layer(context, pos_emb, mask_src)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        context = self.postprocess_layer(context)

        output_dict = defaultdict(lambda: None, {'context': context, 'src_mask': dec_attn_mask, 'src': input})

        return output_dict


class RelativeTransformerDecoder(TransformerDecoder):

    def __init__(self, opt, dicts, positional_encoder, language_embeddings=None, ignore_source=False):

        self.death_rate = opt.death_rate
        self.double_position = opt.double_position
        self.max_memory_size = opt.max_memory_size

        # build_modules will be called from the inherited constructor
        super(RelativeTransformerDecoder, self).__init__(opt, dicts,
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

        print("* Transformer Decoder with Relative Attention with %.2f expected layers" % e_length)

        self.layer_modules = nn.ModuleList()

        for l in range(self.layers):
            # linearly decay the death rate
            death_r = (l + 1.0) / self.layers * self.death_rate

            block = RelativeTransformerDecoderLayer(self.n_heads, self.model_size,
                                                    self.dropout, self.inner_size, self.attn_dropout,
                                                    variational=self.variational_dropout, death_rate=death_r)

            self.layer_modules.append(block)

    def process_embedding(self, input, input_lang=None):

        return input

    def create_self_attn_mask(self):

        raise NotImplementedError

    def create_context_mask(self, input, src, src_lengths, tgt_lengths):

        mask = None

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

        mask = mask.bool()

        return mask

    def create_self_attn_mask(self, input, tgt_lengths, prev_tgt_mem_size):

        qlen = sum(tgt_lengths.tolist())
        mlen = prev_tgt_mem_size
        klen = qlen + mlen

        mask = torch.triu(input.new_ones(qlen, klen), diagonal=1 + mlen).bool()[:, :, None]

        return mask

    def forward_stream(self, input, context, src, input_pos, input_lang, **kwargs):

        input = input.transpose(0, 1)
        src_lengths = kwargs.get("src_lengths", None)
        tgt_lengths = kwargs.get("tgt_lengths", None)
        streaming_state = kwargs.get("streaming_state")

        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        emb = emb * math.sqrt(self.model_size)

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
            context_attn_mask = self.create_context_mask(input, src, src_lengths, tgt_lengths)
            context_attn_mask = context_attn_mask.unsqueeze(0)
        else:
            context_attn_mask = None

        dec_attn_mask = self.create_self_attn_mask(input, tgt_lengths, streaming_state.prev_tgt_mem_size)

        klen = input.size(0) + streaming_state.prev_tgt_mem_size
        pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)

        pos_emb = self.positional_encoder(pos)

        output = self.preprocess_layer(emb)

        for i, layer in enumerate(self.layer_modules):
            # batch_size x src_len x d_model
            buffer = streaming_state.tgt_buffer[i]
            # output, coverage = layer(output, context, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask, mask_src)
            output, coverage, buffer = layer(output, context, pos_emb, dec_attn_mask, context_attn_mask,
                                             incremental=True, incremental_cache=buffer, reuse_source=False)
            streaming_state.tgt_buffer[i] = buffer

        output = self.postprocess_layer(output)

        streaming_state.prev_tgt_mem_size += sum(tgt_lengths.tolist())
        streaming_state.prune_target_memory(self.max_memory_size)

        output_dict = defaultdict(lambda: None, {'hidden': output, 'coverage': coverage, 'context': context})
        output_dict['streaming_state'] = streaming_state

        return output_dict

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
        if streaming:
            return self.forward_stream(input, context, src, input_pos, input_lang, **kwargs)

        """ Embedding: batch_size x len_tgt x d_model """
        input = input.transpose(0, 1)  # T x B
        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        emb = emb * math.sqrt(self.model_size)

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
                mask_src = src.eq(onmt.constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        qlen = input.size(0)
        klen = input.size(0)
        mlen = klen - qlen  # extra memory if expanded
        # preparing self-attention mask. The input is either left or right aligned
        dec_attn_mask = torch.triu(
            emb.new_ones(qlen, klen), diagonal=1 + mlen).byte()[:, :, None]
        pad_mask = input.eq(onmt.constants.PAD).byte()  # L x B

        dec_attn_mask = dec_attn_mask + pad_mask.unsqueeze(0)
        dec_attn_mask = dec_attn_mask.gt(0)
        if onmt.constants.torch_version >= 1.2:
            dec_attn_mask = dec_attn_mask.bool()

        pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)

        pos_emb = self.positional_encoder(pos)

        output = self.preprocess_layer(emb.contiguous())

        pos_emb = self.preprocess_layer(pos_emb)

        for i, layer in enumerate(self.layer_modules):
            # batch_size x src_len x d_model
            # output, coverage = layer(output, context, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask, mask_src)
            output, coverage, _ = layer(output, context, pos_emb, dec_attn_mask, mask_src)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        output = self.postprocess_layer(output)

        output_dict = {'hidden': output, 'coverage': coverage, 'context': context}

        return output_dict

    def step(self, input, decoder_state):
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

        pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)

        pos_emb = self.positional_encoder(pos)

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
            output, coverage, buffer = layer(output, context, pos_emb, dec_attn_mask, mask_src,
                                             incremental=True, incremental_cache=buffer)

            decoder_state.update_attention_buffer(buffer, i)

        output = self.postprocess_layer(output)
        # print(output.size())
        output = output[-1].unsqueeze(0)

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = output
        output_dict['coverage'] = coverage
        output_dict['context'] = context

        return output_dict

        self.src_memory_length = 0
        self.tgt_memory_length = 0
