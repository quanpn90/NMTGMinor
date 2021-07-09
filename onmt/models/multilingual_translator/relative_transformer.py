import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.models.transformer_layers import PositionalEncoding, PrePostProcessing
from onmt.models.transformer_layers import EncoderLayer, DecoderLayer
from onmt.models.transformers import TransformerEncoder, TransformerDecoder, Transformer, TransformerDecodingState
import onmt
from onmt.modules.base_seq2seq import NMTModel, Reconstructor, DecoderState
from onmt.modules.dropout import embedded_dropout
from onmt.modules.sinusoidal_positional_encoding import SinusoidalPositionalEmbedding, FastSinusoidalPositionalEncoding
from onmt.models.transformer_layers import PrePostProcessing
from .relative_transformer_layers import RelativeTransformerEncoderLayer, RelativeTransformerDecoderLayer
from .reversible_transformers import ReversibleTransformerEncoderLayer, reversible_encoder
from .reversible_transformers import ReversibleTransformerDecoderLayer, reversible_decoder
from onmt.modules.identity import Identity
from onmt.utils import flip, expected_length
from collections import defaultdict
import math
import sys
from torch.utils.checkpoint import checkpoint

torch.set_printoptions(threshold=500000)


def create_forward_function(module):
    def forward_pass(*inputs):
        return module(*inputs)

    return forward_pass


class RelativeTransformerEncoder(TransformerEncoder):

    def __init__(self, opt, dicts, positional_encoder, encoder_type='text', language_embeddings=None):
        self.death_rate = opt.death_rate
        self.learnable_position_encoding = opt.learnable_position_encoding
        self.layer_modules = list()
        self.unidirectional = opt.unidirectional
        self.n_heads = opt.n_heads
        self.n_languages = opt.n_languages
        self.checkpointing = opt.checkpointing
        self.absolute_position_encoding = opt.absolute_position_encoding
        self.early_emb_scale = opt.encoder_early_emb_scale
        self.learnable_position_encoding = opt.learnable_position_encoding
        self.rotary_position_encoding = opt.rotary_position_encoding
        self.max_pos_length = opt.max_pos_length
        self.reversible = opt.src_reversible

        # build_modules will be called from the inherited constructor
        super(RelativeTransformerEncoder, self).__init__(opt, dicts, positional_encoder, encoder_type,
                                                         language_embeddings)

        if not self.early_emb_scale and (self.use_language_embedding or self.absolute_position_encoding):
            print("[INFO] Embedding will be scaled after being added with embedding and position encoding."
                  "\n[INFO] For multilingual models its advisable to use -encoder_early_emb_scale")

        # learnable position encoding
        if self.learnable_position_encoding:
            assert not self.rotary_position_encoding
            self.positional_encoder = None
        elif self.rotary_position_encoding:
            from onmt.modules.rotary_postional_encodings import SinusoidalEmbeddings
            self.positional_encoder = SinusoidalEmbeddings(opt.model_size // opt.n_heads)
        else:
            self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)

        if opt.rezero or opt.post_norm:
            self.postprocess_layer = Identity()

        self.d_head = self.model_size // self.n_heads

    def build_modules(self):

        e_length = expected_length(self.layers, self.death_rate)
        if self.reversible:
            print("* Relative Reversible Encoder with %.2f expected layers" % e_length)
        else:
            print("* Relative Translation Encoder with %.2f expected layers" % e_length)

        self.layer_modules = nn.ModuleList()

        for _l in range(self.layers):
            # linearly decay the death rate
            death_r = (_l + 1.0) / self.layers * self.death_rate
            if self.reversible:
                block = ReversibleTransformerEncoderLayer(self.opt, death_rate=death_r)
            else:
                block = RelativeTransformerEncoderLayer(self.opt, death_rate=death_r)

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
        bsz_first_input = input
        input = input.transpose(0, 1)

        dec_attn_mask = bsz_first_input.eq(onmt.constants.PAD).unsqueeze(1)

        mem_len = 0

        mems = None

        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        if self.early_emb_scale:
            """ Scale the emb by sqrt(d_model) """
            emb = emb * math.sqrt(self.model_size)

        """ Adding language embeddings """
        if self.use_language_embedding:
            assert self.language_embedding is not None
            # There is no "unsqueeze" here because the input is T x B x H and lang_emb is B x H
            if self.language_embedding_type in ['sum', 'all_sum']:
                lang_emb = self.language_embedding(input_lang)
                emb = emb + lang_emb.unsqueeze(0)

        """ Adding positional encoding """
        qlen = input.size(0)
        klen = qlen + mem_len

        # Asynchronous positions: 2K+1 positions instead of K+1
        if not self.rotary_position_encoding:
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

            mask_src = input.eq(onmt.constants.PAD).unsqueeze(0)  # 1 x src_len x batch_size for broadcasting
        elif self.rotary_position_encoding:
            # generate rotary position encodings as sinusoidal
            pos_emb = self.positional_encoder(length=qlen)
            mask_src = input.eq(onmt.constants.PAD).transpose(0, 1)

        if onmt.constants.torch_version >= 1.2:
            mask_src = mask_src.bool()

        if not self.early_emb_scale:
            """ Scale the emb by sqrt(d_model) """
            emb = emb * math.sqrt(self.model_size)

        # context size is now T x B x H
        context = self.preprocess_layer(emb)

        if self.reversible:
            context = reversible_encoder(self.layer_modules, context, pos_emb, mask_src)
        else:
            for i, layer in enumerate(self.layer_modules):
                # src_len x batch_size x d_model
                context = layer(context, pos_emb, mask_src, src_lang=input_lang)
                # if self.checkpointing == 0 or self.training is False:
                #     context = layer(context, pos_emb, mask_src, src_lang=input_lang)
                # else:
                #     context = checkpoint(create_forward_function(layer), context, pos_emb, mask_src, input_lang)

        # final layer norm. we can consider this layer norm as a part of the output layer/function
        context = self.postprocess_layer(context)

        output_dict = defaultdict(lambda: None, {'context': context, 'src_mask': dec_attn_mask, 'src': input})

        return output_dict


class RelativeTransformerDecoder(TransformerDecoder):

    def __init__(self, opt, dicts, positional_encoder, language_embeddings=None, ignore_source=False):

        self.death_rate = opt.death_rate
        self.n_heads = opt.n_heads
        self.checkpointing = opt.checkpointing
        self.late_emb_scale = opt.decoder_late_emb_scale
        self.learnable_position_encoding = opt.learnable_position_encoding
        self.max_pos_length = opt.max_pos_length
        self.reversible = opt.tgt_reversible
        self.rotary_position_encoding = opt.rotary_position_encoding

        # build_modules will be called from the inherited constructor
        super(RelativeTransformerDecoder, self).__init__(opt, dicts,
                                                         positional_encoder,
                                                         language_embeddings,
                                                         ignore_source,
                                                         allocate_positions=False)

        if self.learnable_position_encoding:
            assert self.rotary_position_encoding is False
            self.positional_encoder = None
        elif self.rotary_position_encoding:
            from onmt.modules.rotary_postional_encodings import SinusoidalEmbeddings
            self.positional_encoder = SinusoidalEmbeddings(opt.model_size // opt.n_heads)
        else:
            self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)
        self.d_head = self.model_size // self.n_heads

        if opt.rezero or opt.post_norm:
            self.postprocess_layer = Identity()

    def renew_buffer(self, new_len):
        return

    def build_modules(self):

        e_length = expected_length(self.layers, self.death_rate)
        self.opt.ignore_source = self.ignore_source
        if self.reversible:
            print("* Transformer Reversible Decoder with Relative Attention with %.2f expected layers" % e_length)
        else:
            print("* Transformer Decoder with Relative Attention with %.2f expected layers" % e_length)

        self.layer_modules = nn.ModuleList()

        for l in range(self.layers):
            # linearly decay the death rate
            death_r = (l + 1.0) / self.layers * self.death_rate

            if not self.reversible:
                block = RelativeTransformerDecoderLayer(self.opt, death_rate=death_r)
            else:
                block = ReversibleTransformerDecoderLayer(self.opt)

            self.layer_modules.append(block)

    def process_embedding(self, input, input_lang=None):

        return input

    # TODO: merging forward_stream and forward
    # TODO: write a step function for encoder

    def forward(self, input, context, src, input_pos=None, src_lang=None, tgt_lang=None,
                streaming=False, **kwargs):
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
        if not self.late_emb_scale:
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
            mask_src = src.eq(onmt.constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        qlen = input.size(0)
        klen = qlen + mem_len

        # preparing self-attention mask. The input is left aligned so we do not need to add the pad mask
        dec_attn_mask = torch.triu(
            emb.new_ones(qlen, klen), diagonal=1 + mem_len).byte()

        dec_attn_mask = dec_attn_mask.bool()

        # relative positions
        if self.rotary_position_encoding:
            length = qlen
            pos_emb = self.positional_encoder(length=length)
            pos_emb_src = self.positional_encoder(context)
        elif not self.learnable_position_encoding:
            pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)
            pos_emb = self.positional_encoder(pos, bsz=input.size(1))
            pos_emb = self.preprocess_layer(pos_emb)
        else:
            range_vec = torch.arange(klen, device=emb.device)
            range_mat = range_vec.unsqueeze(-1).expand(-1, klen).transpose(0, 1)
            distance_mat = range_vec - range_mat.transpose(0, 1)
            distance_mat.clamp_(-self.max_pos_length, self.max_pos_length).add_(self.max_pos_length)
            pos_emb = distance_mat
            # pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype).long()
            # pos.clamp_(-self.max_pos_length, self.max_pos_length).add_(self.max_pos_length)
            # pos_emb = pos.unsqueeze(1)

        if self.late_emb_scale:
            emb = emb * math.sqrt(self.model_size)

        output = self.preprocess_layer(emb.contiguous())

        if self.reversible:
            # TODO: add src lang and tgt lang to reversible
            output, coverage = reversible_decoder(self.layer_modules, output, pos_emb, context,
                                                  dec_attn_mask.squeeze(-1), mask_src,
                                                  False, None)  # incremental variables
        else:
            for i, layer in enumerate(self.layer_modules):
                output, coverage = layer(output, context, pos_emb, dec_attn_mask, mask_src,
                                         src_lang=src_lang, tgt_lang=tgt_lang,
                                         pos_emb_src=pos_emb_src)
                # if self.checkpointing == 0 or self.training is False:
                #
                #     output, coverage = layer(output, context, pos_emb, dec_attn_mask, mask_src,
                #                                 src_lang=src_lang, tgt_lang=tgt_lang)
                #
                # else:
                #     output, coverage = checkpoint(create_forward_function(layer), output, context, pos_emb,
                #                                      dec_attn_mask,
                #                                      mask_src, src_lang, tgt_lang)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
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

        if buffering:
            # use the last value of input to continue decoding
            input_ = input[:, -1].unsqueeze(1).transpose(0, 1)
            # input_ = input.transpose(0, 1)
        else:
            input_ = input.transpose(0, 1)  # from B x T to T x B

        """ Embedding: batch_size x 1 x d_model """
        emb = self.word_lut(input_)
        if not self.late_emb_scale:
            emb = emb * math.sqrt(self.model_size)
        input = input.transpose(0, 1)
        klen = input.size(0)

        if self.use_language_embedding:
            lang_emb = self.language_embeddings(lang)  # B x H

            if self.language_embedding_type in ['sum', 'all_sum']:
                emb = emb + lang_emb
            elif self.language_embedding_type == 'concat':
                lang_emb = lang_emb.unsqueeze(0).expand_as(emb)
                concat_emb = torch.cat([emb, lang_emb], dim=-1)
                emb = torch.relu(self.projector(concat_emb))
            else:
                raise NotImplementedError

        # prepare position encoding
        qlen = emb.size(0)
        mlen = klen - qlen

        if not self.absolute_position_encoding:
            if self.learnable_position_encoding:
                if buffering:
                    distance_mat = torch.arange(-klen + 1, 1, 1, device=emb.device).unsqueeze(0)
                else:
                    range_vec = torch.arange(klen, device=emb.device)
                    range_mat = range_vec.unsqueeze(-1).expand(-1, klen).transpose(0, 1)
                    distance_mat = range_vec - range_mat.transpose(0, 1)
                distance_mat.clamp_(-self.max_pos_length, self.max_pos_length).add_(self.max_pos_length)
                pos_emb = distance_mat
                # pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)
                # pos.clamp_(-self.max_pos_length, self.max_pos_length).add_(self.max_pos_length)
                # pos_emb = pos.unsqueeze(1)
            else:
                pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)
                pos_emb = self.positional_encoder(pos)
        else:
            if buffering:
                emb = self.positional_encoder(emb.transpose(0, 1), t=input.size(1)).transpose(0, 1)
            else:
                emb = self.positional_encoder(emb.transpose(0, 1)).transpose(0, 1)

        dec_attn_mask = torch.triu(
            emb.new_ones(qlen, klen), diagonal=1 + mlen).byte()[:, :, None]

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

        if self.late_emb_scale:
            emb = emb * math.sqrt(self.model_size)

        output = emb.contiguous()
        if self.reversible:
            incremental = True
            incremental_cache = buffer
            output, coverage = reversible_decoder.apply(self.layer_modules, output, pos_emb, context,
                                                        dec_attn_mask, mask_src,
                                                        incremental, incremental_cache)
        else:
            for i, layer in enumerate(self.layer_modules):
                buffer = buffers[i] if i in buffers else None

                if buffering:
                    output, coverage, buffer = layer(output, context, pos_emb, dec_attn_mask, mask_src,
                                                     tgt_lang=lang, src_lang=src_lang,
                                                     incremental=True, incremental_cache=buffer)
                    decoder_state.update_attention_buffer(buffer, i)
                else:
                    output, coverage = layer(output, context, pos_emb, dec_attn_mask, mask_src,
                                                src_lang=src_lang, tgt_lang=lang)

        # normalize and take the last time step
        output = self.postprocess_layer(output)
        output = output[-1].unsqueeze(0)

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = output
        output_dict['coverage'] = coverage
        output_dict['context'] = context

        return output_dict
