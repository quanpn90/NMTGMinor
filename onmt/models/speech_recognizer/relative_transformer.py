import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.models.transformer_layers import PositionalEncoding
from onmt.modules.pre_post_processing import PrePostProcessing
from onmt.models.transformers import TransformerEncoder, TransformerDecoder, Transformer, TransformerDecodingState
from onmt.modules.sinusoidal_positional_encoding import SinusoidalPositionalEmbedding
import onmt
from onmt.modules.base_seq2seq import NMTModel, Reconstructor, DecoderState
from onmt.modules.dropout import embedded_dropout
from .relative_transformer_layers import RelativeTransformerEncoderLayer, RelativeTransformerDecoderLayer
from onmt.utils import flip, expected_length
from collections import defaultdict
import math
import sys
from onmt.modules.checkpoint import checkpoint
# from torch.utils.checkpoint import checkpoint
from onmt.modules.identity import Identity

torch.set_printoptions(threshold=500000)


def create_forward_function(module):
    def forward_pass(*inputs):
        return module(*inputs)

    return forward_pass


class SpeechTransformerEncoder(TransformerEncoder):

    def __init__(self, opt, dicts, positional_encoder, encoder_type='text', language_embeddings=None):
        self.death_rate = opt.death_rate
        self.learnable_position_encoding = opt.learnable_position_encoding
        self.layer_modules = list()
        self.asynchronous = opt.asynchronous
        self.max_memory_size = opt.max_memory_size
        self.extra_context_size = opt.extra_context_size
        self.experimental = opt.experimental
        self.unidirectional = opt.unidirectional
        self.reversible = opt.src_reversible
        self.n_heads = opt.n_heads
        self.fast_self_attn = opt.fast_self_attention
        self.checkpointing = opt.checkpointing
        self.mpw = opt.multilingual_partitioned_weights
        self.multilingual_linear_projection = opt.multilingual_linear_projection
        self.mln = opt.multilingual_layer_norm
        self.no_input_scale = opt.no_input_scale
        self.learnable_position_encoding = opt.learnable_position_encoding
        self.rotary_position_encoding = opt.rotary_position_encoding
        self.max_pos_length = opt.max_pos_length

        # TODO: multilingually linear transformation

        # build_modules will be called from the inherited constructor
        super().__init__(opt, dicts, positional_encoder, encoder_type, language_embeddings)

        # learnable position encoding
        if self.learnable_position_encoding:
            assert not self.rotary_position_encoding
            self.positional_encoder = None
        elif self.rotary_position_encoding:
            from onmt.modules.rotary_postional_encodings import SinusoidalEmbeddings
            self.positional_encoder = SinusoidalEmbeddings(opt.model_size // opt.n_heads)
        else:
            # or using pre-set sinusoidal
            self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)

        self.d_head = self.model_size // self.n_heads

        if self.multilingual_linear_projection:
            self.linear_proj = nn.Parameter(torch.Tensor(opt.n_languages, self.model_size, self.model_size))

            std_ = math.sqrt(2.0 / (self.model_size + self.model_size))
            torch.nn.init.normal_(self.linear_proj, 0.0, std_)

        self.mln = opt.multilingual_layer_norm

        if not opt.rezero:
            self.postprocess_layer = PrePostProcessing(opt.model_size, opt.dropout, sequence='n', multilingual=self.mln,
                                                       n_languages=opt.n_languages)
        else:
            self.postprocess_layer = Identity()

    def build_modules(self):

        e_length = expected_length(self.layers, self.death_rate)
        print("* Transformer Encoder with Relative Attention with %.2f expected layers" % e_length)
        if self.unidirectional:
            print("* Running a unidirectional Encoder.")

        self.layer_modules = nn.ModuleList()

        for _l in range(self.layers):
            # linearly decay the death rate
            death_r = (_l + 1.0) / self.layers * self.death_rate

            block = RelativeTransformerEncoderLayer(self.opt, death_rate=death_r)
            self.layer_modules.append(block)

    def forward(self, input, input_pos=None, input_lang=None, streaming=False, factorize=True,
                return_states=False, pretrained_layer_states=None, **kwargs):
        """
        :param pretrained_layer_states:
        :param return_states: also return the (unnormalized) outputs of each state
        :param factorize:
        :param input: [B x T x Input_Size]
        :param input_pos: [B x T] positions
        :param input_lang: [B] language ids of each sample
        :param streaming: connect different segments in transformer-xl style
        :param kwargs:
        :return:
        """
        with torch.no_grad():
            nan_mask = torch.isnan(input)
            if nan_mask.any():
                input.masked_fill_(nan_mask, 0)

        if not self.cnn_downsampling:
            mask_src = input.narrow(2, 0, 1).squeeze(2).transpose(0, 1).eq(onmt.constants.PAD).unsqueeze(0)
            dec_attn_mask = input.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD).unsqueeze(1)
            input = input.narrow(2, 1, input.size(2) - 1)
            emb = self.audio_trans(input.contiguous().view(-1, input.size(2))).view(input.size(0),
                                                                                    input.size(1), -1)
            emb = emb.type_as(input)
        else:
            long_mask = input.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD)
            input = input.narrow(2, 1, input.size(2) - 1)

            # first resizing to fit the CNN format
            # note that this is actually conv2d so channel=1, f=40
            input = input.view(input.size(0), input.size(1), -1, self.channels)
            input = input.permute(0, 3, 1, 2)  # [bsz, channels, time, f]

            # apply CNN
            input = self.audio_trans(input)
            input = input.permute(0, 2, 1, 3).contiguous()
            input = input.view(input.size(0), input.size(1), -1)
            input = self.linear_trans(input)
            dec_attn_mask = long_mask[:, 0:input.size(1) * 4:4].unsqueeze(1)
            mask_src = long_mask[:, 0:input.size(1) * 4:4].transpose(0, 1).unsqueeze(0)
            # the size seems to be B x T ?
            emb = input

        emb = emb.transpose(0, 1)
        input = input.transpose(0, 1)
        abs_pos = None
        mem_len = 0
        mems = None

        if self.unidirectional:
            qlen = input.size(0)
            klen = qlen + mem_len
            attn_mask_src = torch.triu(
                emb.new_ones(qlen, klen), diagonal=1 + mem_len).byte()[:, :, None]

            pad_mask = mask_src

            mask_src = pad_mask + attn_mask_src
            # dec_attn_mask = dec_attn_mask + pad_mask.unsqueeze(0)
            mask_src = mask_src.gt(0)

        mask_src = mask_src.bool()

        """ Scale the emb by sqrt(d_model) """
        if not self.no_input_scale:
            emb = emb * math.sqrt(self.model_size)

        """ Adding positional encoding """
        qlen = input.size(0)
        klen = qlen + mem_len

        if not self.learnable_position_encoding:
            pos = torch.arange(klen - 1, -klen, -1.0, device=emb.device, dtype=emb.dtype)
            # pos_emb has size 2T+1 x 1 x H
            pos_emb = self.positional_encoder(pos, bsz=input.size(1))
            pos_emb = self.preprocess_layer(pos_emb)
        else:
            range_vec = torch.arange(klen, device=emb.device)
            range_mat = range_vec.unsqueeze(-1).expand(-1, klen).transpose(0, 1)
            distance_mat = range_vec - range_mat.transpose(0, 1)
            distance_mat.clamp_(-self.max_pos_length, self.max_pos_length).add_(self.max_pos_length)
            pos_emb = distance_mat

        # B x T x H -> T x B x H
        context = emb

        if streaming:
            hids = [context]

        # Apply dropout to both context and pos_emb
        context = self.preprocess_layer(context)

        # maybe try multiplicative ...
        # adding the speech represetation into the first input
        if pretrained_layer_states is not None:
            context = context + pretrained_layer_states

        layer_states = dict()

        if self.mpw:
            input_lang = self.factor_embeddings(input_lang).squeeze(0)
            assert input_lang.ndim == 1

        if self.reversible:
            context = torch.cat([context, context], dim=-1)

            assert streaming is not True, "Streaming and Reversible is not usable yet."
            context = ReversibleEncoderFunction.apply(context, pos_emb, self.layer_modules, mask_src)
        else:
            for i, layer in enumerate(self.layer_modules):
                # src_len x batch_size x d_model

                mems_i = mems[i] if mems is not None and streaming and self.max_memory_size > 0 else None

                context = layer(context, pos_emb, mask_src, mems=mems_i, src_lang=input_lang, factorize=factorize)

                # Summing the context
                # if pretrained_layer_states is not None and i == (self.layers - 1):
                #     context = context + pretrained_layer_states[i]

                if streaming:
                    hids.append(context)

        if return_states:
            layer_states = context

        # final layer norm
        context = self.postprocess_layer(context, factor=input_lang)

        if self.multilingual_linear_projection:
            language_linear_weight_ = torch.index_select(self.linear_proj, 0, input_lang).squeeze(0)
            # context = F.linear(context, language_linear_weight_)
            t, b = context.size(0), context.size(1)
            context = torch.mm(context.view(-1, context.size(-1)), language_linear_weight_)
            context = context.view(t, b, context.size(-1))

        output_dict = defaultdict(lambda: None, {'context': context, 'src_mask': dec_attn_mask,
                                                 'src': input, 'pos_emb': pos_emb})

        if return_states:
            output_dict['layer_states'] = layer_states

        if streaming:
            # streaming_state.prev_src_mem_size += sum(input_length.tolist())
            # streaming_state.prune_source_memory(self.max_memory_size)
            streaming_state.update_src_mems(hids, qlen)
            output_dict['streaming_state'] = streaming_state

        return output_dict


class SpeechTransformerDecoder(TransformerDecoder):

    def __init__(self, opt, dicts, positional_encoder,
                 language_embeddings=None, ignore_source=False,
                 ):

        self.death_rate = opt.death_rate
        self.max_memory_size = opt.max_memory_size
        self.stream_context = opt.stream_context
        self.extra_context_size = opt.extra_context_size
        self.n_heads = opt.n_heads
        self.fast_self_attn = opt.fast_self_attention
        self.mpw = opt.multilingual_partitioned_weights
        self.learnable_position_encoding = opt.learnable_position_encoding
        self.max_pos_length = opt.max_pos_length

        # build_modules will be called from the inherited constructor
        super().__init__(opt, dicts, positional_encoder, language_embeddings,
                         ignore_source,
                         allocate_positions=False)
        if self.learnable_position_encoding:
            self.positional_encoder = None
        else:
            self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)
        self.d_head = self.model_size // self.n_heads
        # Parameters for the position biases - deprecated. kept for backward compatibility
        # self.r_w_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))
        # self.r_r_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))

        self.mln = opt.multilingual_layer_norm

        if not opt.rezero:
            self.postprocess_layer = PrePostProcessing(opt.model_size, opt.dropout, sequence='n', multilingual=self.mln,
                                                       n_languages=opt.n_languages)
        else:
            self.postprocess_layer = Identity()

    def renew_buffer(self, new_len):
        return

    def build_modules(self):

        self.death_rate = 0.0
        e_length = expected_length(self.layers, self.death_rate)
        self.opt.ignore_source = self.ignore_source
        opt = self.opt
        print("* Speech Transformer Decoder with Relative Attention with %.2f layers" % e_length)

        self.layer_modules = nn.ModuleList()

        for _l in range(self.layers):
            # linearly decay the death rate
            death_r = (_l + 1.0) / self.layers * self.death_rate

            block = RelativeTransformerDecoderLayer(self.opt, death_rate=death_r)

            self.layer_modules.append(block)

    def process_embedding(self, input, input_lang=None):

        return input

    # TODO: merging forward_stream and forward
    # TODO: write a step function for encoder

    def forward(self, input, context, src, input_pos=None,
                src_lang=None, tgt_lang=None, streaming=False, factorize=True, **kwargs):
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

        mem_len = 0
        mems = None
        extra_context = None

        if self.use_language_embedding:
            lang_emb = self.language_embeddings(tgt_lang)  # B x H or 1 x H
            if self.language_embedding_type == 'sum':
                emb = emb + lang_emb
            elif self.language_embedding_type == 'concat':
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
            elif self.encoder_type in ["wav2vec2", "wav2vec2_scp"]:
                mask_src = src
            else:
                mask_src = src.eq(onmt.constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        qlen = input.size(0)
        klen = qlen + mem_len
        # preparing self-attention mask. The input must be left-aligned

        dec_attn_mask = torch.triu(
            emb.new_ones(qlen, klen), diagonal=1 + mem_len).byte()[:, :, None]

        dec_attn_mask = dec_attn_mask.bool()

        if not self.learnable_position_encoding:
            pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)
            pos_emb = self.positional_encoder(pos, bsz=input.size(1))
            pos_emb = self.preprocess_layer(pos_emb)
        else:
            range_vec = torch.arange(klen, device=emb.device)
            range_mat = range_vec.unsqueeze(-1).expand(-1, klen).transpose(0, 1)
            distance_mat = range_vec - range_mat.transpose(0, 1)
            distance_mat.clamp_(-self.max_pos_length, self.max_pos_length).add_(self.max_pos_length)
            pos_emb = distance_mat

        # pos_emb = self.positional_encoder(pos, bsz=input.size(1))
        output = self.preprocess_layer(emb.contiguous())
        # pos_emb = self.preprocess_layer(pos_emb)

        if self.mpw:
            src_lang = self.factor_embeddings(src_lang).squeeze(0)
            tgt_lang = self.factor_embeddings(tgt_lang).squeeze(0)
            assert src_lang.ndim == 1 and tgt_lang.ndim == 1

        for i, layer in enumerate(self.layer_modules):
            output, coverage, _ = layer(output, context, pos_emb, dec_attn_mask, mask_src,
                                        src_lang=src_lang, tgt_lang=tgt_lang, factorize=factorize)

        output = self.postprocess_layer(output, factor=tgt_lang)

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

        context = decoder_state.context
        buffers = decoder_state.attention_buffers
        lang = decoder_state.tgt_lang
        src_lang = decoder_state.src_lang
        buffering = decoder_state.buffering

        if decoder_state.concat_input_seq:
            if decoder_state.input_seq is None:
                decoder_state.input_seq = input
            else:
                # concatenate the last input to the previous input sequence
                decoder_state.input_seq = torch.cat([decoder_state.input_seq, input], 0)
            input = decoder_state.input_seq.transpose(0, 1)  # B x T

        src = decoder_state.src.transpose(0, 1) if decoder_state.src is not None else None
        src_mask = decoder_state.src_mask if decoder_state.src_mask is not None else None

        if buffering:
            # use the last value of input to continue decoding
            if input.size(1) > 1:
                input_ = input[:, -1].unsqueeze(1).transpose(0, 1)
            else:
                input_ = input.transpose(0, 1)
        else:
            input_ = input.transpose(0, 1)  # from B x T to T x B

        """ Embedding: batch_size x 1 x d_model """
        emb = self.word_lut(input_) * math.sqrt(self.model_size)
        input = input.transpose(0, 1)
        klen = input.size(0)
        # emb = self.word_lut(input) * math.sqrt(self.model_size)

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

        # pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)
        # pos_emb = self.positional_encoder(pos)
        if self.learnable_position_encoding:
            if buffering:
                distance_mat = torch.arange(-klen + 1, 1, 1, device=emb.device).unsqueeze(0)
            else:
                range_vec = torch.arange(klen, device=emb.device)
                range_mat = range_vec.unsqueeze(-1).expand(-1, klen).transpose(0, 1)
                distance_mat = range_vec - range_mat.transpose(0, 1)
            distance_mat.clamp_(-self.max_pos_length, self.max_pos_length).add_(self.max_pos_length)
            pos_emb = distance_mat
        else:
            pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)
            pos_emb = self.positional_encoder(pos)

        dec_attn_mask = torch.triu(
            emb.new_ones(klen, klen), diagonal=1 + mlen).byte()[:, :, None]  # [:, :, None]

        if buffering:
            dec_attn_mask = dec_attn_mask[-1].unsqueeze(0)
        dec_attn_mask = dec_attn_mask.bool()

        if context is not None:
            if self.encoder_type == "audio":
                # The "slow" version of translator only keeps the source mask of audio as src
                # Thats why we need to check if the src has already been narrowed before
                if src.dim() == 3:
                    if not self.encoder_cnn_downsampling:
                        mask_src = src.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD).unsqueeze(1)
                    else:
                        long_mask = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD)
                        mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
                elif self.encoder_cnn_downsampling:
                    long_mask = src.eq(onmt.constants.PAD)
                    mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
                else:
                    mask_src = src.eq(onmt.constants.PAD).unsqueeze(1)
            elif self.encoder_type == "wav2vec2":
                # mask_src = src
                mask_src = src_mask
            else:

                mask_src = src.eq(onmt.constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        output = emb.contiguous()

        for i, layer in enumerate(self.layer_modules):
            buffer = buffers[i] if i in buffers else None

            if buffering:
                output, coverage, buffer = layer(output, context, pos_emb, dec_attn_mask, mask_src,
                                                 tgt_lang=lang, src_lang=src_lang,
                                                 incremental=True, incremental_cache=buffer)
                decoder_state.update_attention_buffer(buffer, i)
            else:
                output, coverage, _ = layer(output, context, pos_emb, dec_attn_mask, mask_src,
                                            tgt_lang=lang, src_lang=src_lang)

        # normalize and take the last time step
        output = self.postprocess_layer(output, factor=lang)
        output = output[-1].unsqueeze(0)

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = output
        output_dict['coverage'] = coverage
        output_dict['context'] = context

        return output_dict


class RelativeTransformer(Transformer):

    def create_decoder_state(self, batch, beam_size=1, type=1, streaming=False, previous_decoding_state=None,
                             factorize=True,
                             pretrained_layer_states=None, **kwargs):
        """
        Generate a new decoder state based on the batch input
        :param factorize:
        :param pretrained_layer_states:
        :param previous_decoding_state:
        :param streaming:
        :param type:
        :param batch: Batch object (may not contain target during decoding)
        :param beam_size: Size of beam used in beam search
        :return:
        """

        # in this case batch size should be 1
        src = batch.get('source')
        src_pos = batch.get('source_pos')
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')
        src_lengths = batch.src_lengths

        src_transposed = src.transpose(0, 1)

        if previous_decoding_state is None:

            # if the previous stream is None (the first segment in the stream)
            # then proceed normally like normal translation
            # init a new stream state
            streaming_state = self.init_stream()

            encoder_output = self.encoder(src_transposed, input_pos=src_pos,
                                          input_lang=src_lang, src_lengths=src_lengths,
                                          streaming=streaming, streaming_state=streaming_state,
                                          factorize=factorize, pretrained_layer_states=pretrained_layer_states)

            if streaming:
                decoder_state = StreamDecodingState(src, tgt_lang, encoder_output['context'],
                                                    encoder_output['src_mask'],
                                                    beam_size=beam_size, model_size=self.model_size, type=type,
                                                    cloning=True, streaming_state=streaming_state)
            else:
                decoder_state = TransformerDecodingState(src, tgt_lang, encoder_output['context'],
                                                         encoder_output['src_mask'],
                                                         beam_size=beam_size, model_size=self.model_size, type=type)
        else:
            streaming_state = previous_decoding_state.streaming_state

            # to have the same batch/beam size with the previous memory ..
            src_transposed = src_transposed.repeat(beam_size, 1)
            src = src.repeat(1, beam_size)

            encoder_output = self.encoder(src_transposed, input_pos=src_pos,
                                          input_lang=src_lang, src_lengths=src_lengths,
                                          streaming=True, streaming_state=streaming_state)

            context = encoder_output['context']

            if self.decoder.extra_context_size > 0:
                # print("Using extra context with extra %d states" % self.decoder.extra_context_size)
                # print("")
                prev_context = previous_decoding_state.context
                extra_context = prev_context[-self.decoder.extra_context_size:].detach()
                context = torch.cat([extra_context, context], dim=0)

                prev_src = previous_decoding_state.src[-self.decoder.extra_context_size:].detach()
                src = torch.cat([prev_src, src], dim=0)

            decoder_state = StreamDecodingState(src, tgt_lang, context,
                                                encoder_output['src_mask'],
                                                beam_size=beam_size, model_size=self.model_size, type=type,
                                                cloning=False, streaming_state=streaming_state)

        return decoder_state

    def init_stream(self):

        param = next(self.parameters())
        layers = self.decoder.layers
        streaming_state = StreamState(layers, self.decoder.max_memory_size, param.device, param.dtype)
        return streaming_state

    def step(self, input_t, decoder_state, streaming=False):
        """
        Decoding function:
        generate new decoder output based on the current input and current decoder state
        the decoder state is updated in the process
        :param streaming:
        :param input_t: the input word index at time t
        :param decoder_state: object DecoderState containing the buffers required for decoding
        :return: a dictionary containing: log-prob output and the attention coverage
        """

        output_dict = self.decoder.step(input_t, decoder_state, streaming=streaming)
        output_dict['src'] = decoder_state.src.transpose(0, 1)

        log_prob = self.generator[0](output_dict).squeeze(0)
        log_prob = F.log_softmax(log_prob, dim=-1, dtype=torch.float32)

        coverage = output_dict['coverage']
        last_coverage = coverage[:, -1, :].squeeze(1)

        output_dict['log_prob'] = log_prob
        output_dict['coverage'] = last_coverage

        return output_dict

    def set_memory_size(self, src_memory_size, tgt_memory_size):

        self.encoder.max_memory_size = src_memory_size
        self.decoder.max_memory_size = tgt_memory_size