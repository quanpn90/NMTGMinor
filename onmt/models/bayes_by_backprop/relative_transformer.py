import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.models.transformer_layers import PositionalEncoding, PrePostProcessing
from onmt.models.transformer_layers import EncoderLayer, DecoderLayer
from onmt.models.transformers import TransformerEncoder, TransformerDecoder, Transformer, TransformerDecodingState
import onmt
from onmt.modules.base_seq2seq import NMTModel, Reconstructor, DecoderState
from onmt.modules.dropout import embedded_dropout
from onmt.models.transformer_layers import PrePostProcessing
from onmt.utils import flip, expected_length
from collections import defaultdict
import math
import sys
from onmt.models.bayes_by_backprop.relative_transformer_layers import \
    TransformerEncoderLayer, TransformerDecoderLayer


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
            return pos_emb[:, None, :].repeat(1, bsz, 1)
        else:
            return pos_emb[:, None, :]


class RelativeTransformerEncoder(TransformerEncoder):

    def __init__(self, opt, dicts, positional_encoder, encoder_type='text', language_embeddings=None):
        self.death_rate = opt.death_rate
        self.layer_modules = list()
        self.asynchronous = opt.asynchronous
        self.max_memory_size = opt.max_memory_size
        self.extra_context_size = opt.extra_context_size
        self.experimental = opt.experimental
        self.unidirectional = opt.unidirectional
        self.reversible = opt.src_reversible
        self.n_heads = opt.n_heads
        self.fast_self_attn = opt.fast_self_attention

        # build_modules will be called from the inherited constructor
        super(RelativeTransformerEncoder, self).__init__(opt, dicts, positional_encoder, encoder_type,
                                                         language_embeddings)

        # learnable position encoding

        self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)

        self.d_head = self.model_size // self.n_heads

    def build_modules(self):
        assert self.opt.src_reversible == False
        e_length = expected_length(self.layers, self.death_rate)
        print("* Bayes-By-Backprop Relative Transformer Encoder with %.2f expected layers" % e_length)
        if self.unidirectional:
            print("* Running a unidirectional Encoder.")

        self.layer_modules = nn.ModuleList()

        for _l in range(self.layers):
            # linearly decay the death rate
            death_r = (_l + 1.0) / self.layers * self.death_rate
            block = TransformerEncoderLayer(self.opt, death_rate=death_r)
            self.layer_modules.append(block)

    def forward(self, input, input_pos=None, input_lang=None, **kwargs):
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
            # mask_src = input.eq(onmt.constants.PAD).unsqueeze(1)  # batch_size x src_len x 1 for broadcasting
            dec_attn_mask = bsz_first_input.eq(onmt.constants.PAD).unsqueeze(1)

            mem_len = 0
            mask_src = input.eq(onmt.constants.PAD).unsqueeze(0)  # batch_size x src_len x 1 for broadcasting
            mems = None

            emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)

            """ Adding language embeddings """
            if self.use_language_embedding:
                assert self.language_embedding is not None
                # There is no "unsqueeze" here because the input is T x B x H and lang_emb is B x H
                if self.language_embedding_type in ['sum', 'all_sum']:
                    lang_emb = self.language_embedding(input_lang)
                    emb = emb + lang_emb.unsqueeze(0)

        else:
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
            mems = None

        if self.unidirectional:
            qlen = input.size(0)
            klen = qlen + mem_len
            attn_mask_src = torch.triu(
                emb.new_ones(qlen, klen), diagonal=1 + mem_len).byte()[:, :, None]

            # pad_mask = mask_src

            # mask_src = pad_mask + attn_mask_src
            # dec_attn_mask = dec_attn_mask + pad_mask.unsqueeze(0)
            # mask_src = mask_src.gt(0)

            # with right padding, causal mask covers the mask pad
            mask_src = attn_mask_src

        if onmt.constants.torch_version >= 1.2:
            mask_src = mask_src.bool()

        """ Scale the emb by sqrt(d_model) """
        emb = emb * math.sqrt(self.model_size)

        """ positional encoding """
        qlen = input.size(0)
        klen = qlen + mem_len

        # Asynchronous positions: 2K+1 positions instead of K+1
        if self.unidirectional:
            pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)
        else:
            pos = torch.arange(klen - 1, -klen, -1.0, device=emb.device, dtype=emb.dtype)

        # pos_emb has size 2T+1 x 1 x H
        pos_emb = self.positional_encoder(pos, bsz=input.size(1) if self.fast_self_attn else None)

        # B x T x H -> T x B x H
        context = emb

        # Apply dropout to both context and pos_emb
        context = self.preprocess_layer(context)

        pos_emb = self.preprocess_layer(pos_emb)

        for i, layer in enumerate(self.layer_modules):
            # src_len x batch_size x d_model

            context = layer(context, pos_emb, mask_src)

        # final layer norm
        context = self.postprocess_layer(context)

        output_dict = defaultdict(lambda: None, {'context': context, 'src_mask': dec_attn_mask, 'src': input})

        return output_dict

    def log_prior(self):

        log_prior = 0
        for module in self.layer_modules:
            log_prior += module.log_prior()
        return log_prior

    def log_variational_posterior(self):

        log_variational_posterior = 0
        for module in self.layer_modules:
            log_variational_posterior += module.log_variational_posterior()
        return log_variational_posterior


class RelativeTransformerDecoder(TransformerDecoder):

    def __init__(self, opt, dicts, positional_encoder, language_embeddings=None, ignore_source=False):

        self.death_rate = opt.death_rate
        self.max_memory_size = opt.max_memory_size
        self.extra_context_size = opt.extra_context_size
        self.n_heads = opt.n_heads
        self.fast_self_attn = opt.fast_self_attention

        # build_modules will be called from the inherited constructor
        super(RelativeTransformerDecoder, self).__init__(opt, dicts,
                                                         positional_encoder,
                                                         language_embeddings,
                                                         ignore_source,
                                                         allocate_positions=False)
        self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)
        self.d_head = self.model_size // self.n_heads
        # Parameters for the position biases - deprecated. kept for backward compatibility
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))

    def renew_buffer(self, new_len):
        return

    def build_modules(self):

        e_length = expected_length(self.layers, self.death_rate)
        self.opt.ignore_source = self.ignore_source
        print("* Bayes-By-Backprop Relative Transformer Decoder with %.2f expected layers" % e_length)

        self.layer_modules = nn.ModuleList()

        for l in range(self.layers):
            # linearly decay the death rate
            death_r = (l + 1.0) / self.layers * self.death_rate

            block = TransformerDecoderLayer(self.opt, death_rate=death_r)

            self.layer_modules.append(block)

    def process_embedding(self, input, input_lang=None):

        return input

    def log_prior(self):
        log_prior = 0
        for module in self.layer_modules:
            log_prior += module.log_prior()
        return log_prior

    def log_variational_posterior(self):

        log_variational_posterior = 0
        for module in self.layer_modules:
            log_variational_posterior += module.log_variational_posterior()
        return log_variational_posterior

    def forward(self, input, context, src, input_pos=None, input_lang=None, **kwargs):
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
        klen = qlen + mem_len
        # preparing self-attention mask. The input is either left or right aligned

        dec_attn_mask = torch.triu(
            emb.new_ones(qlen, klen), diagonal=1 + mem_len).byte()[:, :, None]
        # pad_mask = input.eq(onmt.constants.PAD).byte()  # L x B
        #
        # dec_attn_mask = dec_attn_mask + pad_mask.unsqueeze(0)
        # dec_attn_mask = dec_attn_mask.gt(0)
        dec_attn_mask = dec_attn_mask.bool()

        pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)

        pos_emb = self.positional_encoder(pos, bsz=input.size(1) if self.fast_self_attn else None)

        output = self.preprocess_layer(emb.contiguous())

        pos_emb = self.preprocess_layer(pos_emb)

        for i, layer in enumerate(self.layer_modules):

            output, coverage, _ = layer(output, context, pos_emb, dec_attn_mask, mask_src)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        output = self.postprocess_layer(output)

        output_dict = {'hidden': output, 'coverage': coverage, 'context': context}
        output_dict = defaultdict(lambda: None, output_dict)

        return output_dict

    def step(self, input, decoder_state, **kwargs):
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
        buffering = decoder_state.buffering

        if decoder_state.concat_input_seq:
            if decoder_state.input_seq is None:
                decoder_state.input_seq = input
            else:
                # concatenate the last input to the previous input sequence
                decoder_state.input_seq = torch.cat([decoder_state.input_seq, input], 0)
            input = decoder_state.input_seq.transpose(0, 1)  # B x T

        src = decoder_state.src.transpose(0, 1) if decoder_state.src is not None else None

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

            if buffering:
                output, coverage, buffer = layer(output, context, pos_emb, dec_attn_mask, mask_src,
                                                 incremental=True, incremental_cache=buffer)
                decoder_state.update_attention_buffer(buffer, i)
            else:
                output, coverage, _ = layer(output, context, pos_emb, dec_attn_mask, mask_src)

        # normalize and take the last time step
        output = self.postprocess_layer(output)
        output = output[-1].unsqueeze(0)

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = output
        output_dict['coverage'] = coverage
        output_dict['context'] = context

        return output_dict


class BayesianTransformer(Transformer):

    def log_prior(self):

        return self.encoder.log_prior() + self.decoder.log_prior()

    def log_variational_posterior(self):

        return self.encoder.log_variational_posterior() + self.decoder.log_variational_posterior()