import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.checkpoint import checkpoint

import onmt
from onmt.models.transformer_layers import EncoderLayer, DecoderLayer, PositionalEncoding, \
    PrePostProcessing
from onmt.modules.base_seq2seq import NMTModel, Reconstructor, DecoderState
from onmt.modules.dropout import embedded_dropout, switchout
from onmt.modules.linear import FeedForward, FeedForwardSwish
from onmt.reversible_models.transformers import ReversibleTransformerEncoderLayer, ReversibleEncoderFunction, \
    ReversibleDecoderFunction, ReversibleTransformerDecoderLayer
from onmt.utils import flip, expected_length

torch_version = float(torch.__version__[:3])


class MixedEncoder(nn.Module):

    def __init(self, text_encoder, audio_encoder):
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder

    def forward(self, input, **kwargs):
        """
        Inputs Shapes:
            input: batch_size x len_src (to be transposed)

        Outputs Shapes:
            out: batch_size x len_src x d_model
            mask_src

        """

        """ Embedding: batch_size x len_src x d_model """

        if input.dim() == 2:
            return self.text_encoder.forward(input)
        else:
            return self.audio_encoder.forward(input)


class TransformerEncoder(nn.Module):
    """Encoder in 'Attention is all you need'

    Args:
        opt: list of options ( see train.py )
        dicts : dictionary (for source language)

    """

    def __init__(self, opt, embedding, positional_encoder, encoder_type='text', language_embeddings=None):

        super(TransformerEncoder, self).__init__()

        self.opt = opt
        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        if hasattr(opt, 'encoder_layers') and opt.encoder_layers != -1:
            self.layers = opt.encoder_layers
        else:
            self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout

        self.input_type = encoder_type
        self.cnn_downsampling = opt.cnn_downsampling
        self.death_rate = opt.death_rate

        self.switchout = opt.switchout
        self.varitional_dropout = opt.variational_dropout
        self.use_language_embedding = opt.use_language_embedding
        self.language_embedding_type = opt.language_embedding_type

        self.time = opt.time
        self.lsh_src_attention = opt.lsh_src_attention
        self.reversible = opt.src_reversible

        feature_size = opt.input_size
        self.channels = 1  # n. audio channels

        if opt.upsampling:
            feature_size = feature_size // 4

        if encoder_type != "text":
            if not self.cnn_downsampling:
                self.audio_trans = nn.Linear(feature_size, self.model_size)
                torch.nn.init.xavier_uniform_(self.audio_trans.weight)
            else:
                channels = self.channels  # should be 1

                if not opt.no_batch_norm:
                    cnn = [nn.Conv2d(channels, 32, kernel_size=(3, 3), stride=2), nn.ReLU(True), nn.BatchNorm2d(32),
                           nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2), nn.ReLU(True), nn.BatchNorm2d(32)]
                else:
                    cnn = [nn.Conv2d(channels, 32, kernel_size=(3, 3), stride=2), nn.ReLU(True),
                           nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2), nn.ReLU(True)]

                feat_size = (((feature_size // channels) - 3) // 4) * 32
                self.audio_trans = nn.Sequential(*cnn)
                self.linear_trans = nn.Linear(feat_size, self.model_size)
                # assert self.model_size == feat_size, \
                #     "The model dimension doesn't match with the feature dim, expecting %d " % feat_size
        else:
            self.word_lut = embedding

        self.time_transformer = positional_encoder
        self.language_embedding = language_embeddings

        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d',
                                                  variational=self.varitional_dropout)

        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')

        self.positional_encoder = positional_encoder

        self.layer_modules = nn.ModuleList()
        self.build_modules()

    def build_modules(self):

        e_length = expected_length(self.layers, self.death_rate)

        if self.reversible:
            print("* Reversible Transformer Encoder with Absolute Attention with %.2f expected layers" % e_length)
        else:
            print("* Transformer Encoder with Absolute Attention with %.2f expected layers" % e_length)

        for _l in range(self.layers):
            # linearly decay the death rate
            death_r = (_l + 1.0) / self.layers * self.death_rate

            if not self.reversible:
                block = EncoderLayer(self.opt, death_rate=death_r)
            else:
                block = ReversibleTransformerEncoderLayer(self.opt, death_rate=death_r)

            self.layer_modules.append(block)

    def forward(self, input, input_lang=None, **kwargs):
        """
        Inputs Shapes:
            input: batch_size x len_src (to be transposed)

        Outputs Shapes:
            out: batch_size x len_src x d_model
            mask_src

        """

        """ Embedding: batch_size x len_src x d_model """
        if self.input_type == "text":
            mask_src = input.eq(onmt.constants.SRC_PAD).unsqueeze(1)  # batch_size x 1 x len_src for broadcasting

            emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        else:
            if not self.cnn_downsampling:
                mask_src = input.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.SRC_PAD).unsqueeze(1)
                input = input.narrow(2, 1, input.size(2) - 1)
                emb = self.audio_trans(input.contiguous().view(-1, input.size(2))).view(input.size(0),
                                                                                        input.size(1), -1)
                emb = emb.type_as(input)
            else:
                long_mask = input.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.SRC_PAD)
                input = input.narrow(2, 1, input.size(2) - 1)

                # first resizing to fit the CNN format
                input = input.view(input.size(0), input.size(1), -1, self.channels)
                input = input.permute(0, 3, 1, 2)

                input = self.audio_trans(input)
                input = input.permute(0, 2, 1, 3).contiguous()
                input = input.view(input.size(0), input.size(1), -1)
                input = self.linear_trans(input)

                mask_src = long_mask[:, 0:input.size(1) * 4:4].unsqueeze(1)
                # the size seems to be B x T ?
                emb = input

        mask_src = mask_src.bool()

        """ Scale the emb by sqrt(d_model) """
        emb = emb * math.sqrt(self.model_size)

        """ Adding positional encoding """
        emb = self.time_transformer(emb)

        """ Adding language embeddings """
        if self.use_language_embedding:
            assert self.language_embedding is not None

            if self.language_embedding_type in ['sum', 'all_sum']:
                lang_emb = self.language_embedding(input_lang)
                emb = emb + lang_emb.unsqueeze(1)

        # B x T x H -> T x B x H
        context = emb.transpose(0, 1)

        context = self.preprocess_layer(context)

        if self.reversible:
            # x_1 and x_2 are the same at first for reversible
            context = torch.cat([context, context], dim=-1)

            context = ReversibleEncoderFunction.apply(context, self.layer_modules, mask_src)
        else:
            for i, layer in enumerate(self.layer_modules):
                context = layer(context, mask_src)  # batch_size x len_src x d_model

        context = self.postprocess_layer(context)

        output_dict = {'context': context, 'src_mask': mask_src}

        # return context, mask_src
        return output_dict


class TransformerDecoder(nn.Module):
    """Decoder in 'Attention is all you need'"""

    def __init__(self, opt, embedding, positional_encoder,
                 language_embeddings=None, ignore_source=False, allocate_positions=True):
        """
        :param opt:
        :param embedding:
        :param positional_encoder:
        :param attribute_embeddings:
        :param ignore_source:
        """
        super(TransformerDecoder, self).__init__()
        opt.ignore_source = ignore_source
        self.opt = opt

        self.model_size = opt.model_size
        self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout
        self.encoder_type = opt.encoder_type
        self.ignore_source = ignore_source
        self.encoder_cnn_downsampling = opt.cnn_downsampling
        self.variational_dropout = opt.variational_dropout
        self.switchout = opt.switchout
        self.death_rate = opt.death_rate
        self.time = opt.time
        self.use_language_embedding = opt.use_language_embedding
        self.language_embedding_type = opt.language_embedding_type
        self.reversible = opt.tgt_reversible

        self.time_transformer = positional_encoder

        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d',
                                                  variational=self.variational_dropout)

        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')

        self.word_lut = embedding

        # Using feature embeddings in models
        self.language_embeddings = language_embeddings

        if self.language_embedding_type == 'concat':
            self.projector = nn.Linear(opt.model_size * 2, opt.model_size)

        self.positional_encoder = positional_encoder

        if allocate_positions:
            if hasattr(self.positional_encoder, 'len_max'):
                len_max = self.positional_encoder.len_max
                mask = torch.ByteTensor(np.triu(np.ones((len_max, len_max)), k=1).astype('uint8'))
                self.register_buffer('mask', mask)

        self.layer_modules = nn.ModuleList()
        self.build_modules()

    def build_modules(self):

        e_length = expected_length(self.layers, self.death_rate)

        if self.reversible:
            print("* Reversible Transformer Decoder with Absolute Attention with %.2f expected layers" % e_length)
        else:
            print("* Transformer Decoder with Absolute Attention with %.2f expected layers" % e_length)

        for _l in range(self.layers):
            # linearly decay the death rate
            death_r = (_l + 1.0) / self.layers * self.death_rate
            if not self.reversible:
                # block = DecoderLayer(self.n_heads, self.model_size,
                #                      self.dropout, self.inner_size, self.attn_dropout,
                #                      variational=self.variational_dropout, death_rate=death_r)
                block = DecoderLayer(self.opt, death_rate=death_r)
            else:
                block = ReversibleTransformerDecoderLayer(self.opt, death_rate=_l)

            self.layer_modules.append(block)

    def renew_buffer(self, new_len):

        self.positional_encoder.renew(new_len)
        mask = torch.ByteTensor(np.triu(np.ones((new_len + 1, new_len + 1)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)

    def process_embedding(self, input, input_lang=None):

        input_ = input

        emb = embedded_dropout(self.word_lut, input_, dropout=self.word_dropout if self.training else 0)
        if self.time == 'positional_encoding':
            emb = emb * math.sqrt(self.model_size)
        """ Adding positional encoding """
        emb = self.time_transformer(emb)

        if self.use_language_embedding:
            lang_emb = self.language_embeddings(input_lang)  # B x H or 1 x H
            if self.language_embedding_type == 'sum':
                emb = emb + lang_emb.unsqueeze(1)
            elif self.language_embedding_type == 'concat':
                lang_emb = lang_emb.unsqueeze(1).expand_as(emb)
                concat_emb = torch.cat([emb, lang_emb], dim=-1)
                emb = torch.relu(self.projector(concat_emb))
            else:
                raise NotImplementedError
        return emb

    def forward(self, input, context, src, tgt_lang=None, **kwargs):
        """
        Inputs Shapes:
            input: (Variable) batch_size x len_tgt (to be transposed)
            context: (Variable) batch_size x len_src x d_model
            mask_src (Tensor) batch_size x len_src
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src

        """

        """ Embedding: batch_size x len_tgt x d_model """

        emb = self.process_embedding(input, tgt_lang)

        if context is not None:
            if self.encoder_type == "audio":
                if not self.encoder_cnn_downsampling:
                    mask_src = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.SRC_PAD).unsqueeze(1)
                else:
                    long_mask = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.SRC_PAD)
                    mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
            else:

                mask_src = src.data.eq(onmt.constants.SRC_PAD).unsqueeze(1)
        else:
            mask_src = None

        len_tgt = input.size(1)
        # mask_tgt = input.eq(onmt.constants.PAD).byte().unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        # mask_tgt = torch.gt(mask_tgt, 0)
        mask_tgt = torch.triu(
                emb.new_ones(len_tgt, len_tgt), diagonal=1).byte().unsqueeze(0)

        mask_tgt = mask_tgt.bool()

        output = self.preprocess_layer(emb.transpose(0, 1).contiguous())

        if self.reversible:
            # x_1 and x_2 are the same at first for reversible
            output = torch.cat([output, output], dim=-1)

            output = ReversibleDecoderFunction.apply(output, context, self.layer_modules,
                                                     mask_tgt, mask_src)
            coverage = None
        else:
            for i, layer in enumerate(self.layer_modules):
                output, coverage, _ = layer(output, context, mask_tgt, mask_src)  # batch_size x len_src x d_model

        # From Google T2T: normalization to control network output magnitude
        output = self.postprocess_layer(output)

        output_dict = defaultdict(lambda: None, {'hidden': output, 'coverage': coverage, 'context': context})

        # return output, None
        return output_dict

    def step(self, input, decoder_state, **kwargs):
        """
        Inputs Shapes:
            input: (Variable) batch_size x len_tgt (to be transposed)
            context: (Variable) batch_size x len_src x d_model
            mask_src (Tensor) batch_size x len_src
            buffer (List of tensors) List of batch_size * len_tgt-1 * d_model for self-attention recomputing
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src

        """
        context = decoder_state.context
        buffers = decoder_state.attention_buffers
        lang = decoder_state.tgt_lang
        # mask_src = decoder_state.src_mask

        if decoder_state.concat_input_seq:
            if decoder_state.input_seq is None:
                decoder_state.input_seq = input
            else:
                # concatenate the last input to the previous input sequence
                decoder_state.input_seq = torch.cat([decoder_state.input_seq, input], 0)
            input = decoder_state.input_seq.transpose(0, 1)

        src = decoder_state.src.transpose(0, 1) if decoder_state.src is not None else None

        if input.size(1) > 1:
            input_ = input[:, -1].unsqueeze(1)
        else:
            input_ = input
        """ Embedding: batch_size x 1 x d_model """
        check = input_.gt(self.word_lut.num_embeddings)
        emb = self.word_lut(input_)

        """ Adding positional encoding """
        emb = emb * math.sqrt(self.model_size)
        emb = self.time_transformer(emb, t=input.size(1))
        # emb should be batch_size x 1 x dim

        if self.use_language_embedding:
            if self.use_language_embedding:
                lang_emb = self.language_embeddings(lang)  # B x H or 1 x H
                if self.language_embedding_type == 'sum':
                    emb = emb + lang_emb
                elif self.language_embedding_type == 'concat':
                    # replace the bos embedding with the language
                    if input.size(1) == 1:
                        bos_emb = lang_emb.expand_as(emb[:, 0, :])
                        emb[:, 0, :] = bos_emb

                    lang_emb = lang_emb.unsqueeze(1).expand_as(emb)
                    concat_emb = torch.cat([emb, lang_emb], dim=-1)
                    emb = torch.relu(self.projector(concat_emb))
                else:
                    raise NotImplementedError

        emb = emb.transpose(0, 1)

        # batch_size x 1 x len_src
        if context is not None:
            if self.encoder_type == "audio":
                if src.dim() == 3:
                    if self.encoder_cnn_downsampling:
                        long_mask = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.SRC_PAD)
                        mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
                    else:
                        mask_src = src.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.SRC_PAD).unsqueeze(1)
                elif self.encoder_cnn_downsampling:
                    long_mask = src.eq(onmt.constants.SRC_PAD)
                    mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
                else:
                    mask_src = src.eq(onmt.constants.SRC_PAD).unsqueeze(1)
            else:
                mask_src = src.eq(onmt.constants.SRC_PAD).unsqueeze(1)
        else:
            mask_src = None

        len_tgt = input.size(1)
        mask_tgt = torch.triu(
            emb.new_ones(len_tgt, len_tgt), diagonal=1).byte().unsqueeze(0)
        # only get the final step of the mask during decoding (because the input of the network is only the last step)
        mask_tgt = mask_tgt[:, -1, :].unsqueeze(1)

        if torch_version >= 1.2:
            mask_tgt = mask_tgt.bool()

        output = emb.contiguous()

        if self.reversible:
            # x_1 and x_2 are the same at first for reversible
            # output = torch.cat([output, output], dim=-1)
            output1, output2 = output, output

        for i, layer in enumerate(self.layer_modules):
            buffer = buffers[i] if i in buffers else None
            assert (output.size(0) == 1)

            if self.reversible:
                output1, output2, coverage, buffer = layer(output1, output2, context, mask_tgt, mask_src,
                                                           incremental=True, incremental_cache=buffer)
            else:
                output, coverage, buffer = layer(output, context, mask_tgt, mask_src,
                                                 incremental=True, incremental_cache=buffer)

            decoder_state.update_attention_buffer(buffer, i)

        if self.reversible:
            output = output1 + output2

        output = self.postprocess_layer(output)

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = output
        output_dict['coverage'] = coverage
        output_dict['context'] = context

        return output_dict


class Transformer(NMTModel):
    """Main model in 'Attention is all you need' """

    def __init__(self, encoder, decoder, generator=None, rec_decoder=None, rec_generator=None,
                 mirror=False, ctc=False):
        super().__init__(encoder, decoder, generator, rec_decoder, rec_generator, ctc=ctc)
        self.model_size = self.decoder.model_size
        self.switchout = self.decoder.switchout
        if hasattr(self.decoder, 'word_lut'):
            self.tgt_vocab_size = self.decoder.word_lut.weight.size(0)

        if self.encoder.input_type == 'text':
            self.src_vocab_size = self.encoder.word_lut.weight.size(0)
        else:
            self.src_vocab_size = 0

        if mirror:
            self.mirror_decoder = copy.deepcopy(self.decoder)
            self.mirror_g = nn.Linear(decoder.model_size, decoder.model_size)
            self.mirror_generator = copy.deepcopy(self.generator)
            self.mirror_generator[0].linear.weight = self.decoder.word_lut.weight

        if self.reconstruct:
            self.rec_linear = nn.Linear(decoder.model_size, decoder.model_size)

        if self.ctc:
            self.ctc_linear = nn.Linear(encoder.model_size, self.tgt_vocab_size)

    def reset_states(self):
        return

    def forward(self, batch, target_mask=None, streaming=False, zero_encoder=False,
                mirror=False, streaming_state=None, nce=False, factorize=True,
                pretrained_layer_states=None, **kwargs):
        """
        :param pretrained_layer_states:
        :param nce: use noise contrastive estimation
        :param streaming_state:
        :param streaming:
        :param mirror: if using mirror network for future anticipation
        :param batch: data object sent from the dataset
        :param target_mask:
        :param zero_encoder: zero out the encoder output (if necessary)
        :return:
        """
        if self.switchout > 0 and self.training:
            batch.switchout(self.switchout, self.src_vocab_size, self.tgt_vocab_size)

        src = batch.get('source')
        tgt = batch.get('target_input')
        src_pos = batch.get('source_pos')
        tgt_pos = batch.get('target_pos')
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')
        src_lengths = batch.src_lengths
        tgt_lengths = batch.tgt_lengths

        org_src = src
        org_tgt = tgt
        src = src.transpose(0, 1)  # transpose to have batch first
        tgt = tgt.transpose(0, 1)

        encoder_output = self.encoder(src, input_pos=src_pos, input_lang=src_lang, streaming=streaming,
                                      src_lengths=src_lengths, streaming_state=streaming_state, factorize=factorize,
                                      pretrained_layer_states=pretrained_layer_states)

        encoder_output = defaultdict(lambda: None, encoder_output)
        context = encoder_output['context']

        # the state is changed if streaming
        streaming_state = encoder_output['streaming_state']

        # zero out the encoder part for pre-training
        if zero_encoder:
            context.zero_()

        decoder_output = self.decoder(tgt, context, src,
                                      src_lang=src_lang, tgt_lang=tgt_lang, input_pos=tgt_pos, streaming=streaming,
                                      src_lengths=src_lengths, tgt_lengths=tgt_lengths,
                                      streaming_state=streaming_state, factorize=factorize)

        # update the streaming state again
        decoder_output = defaultdict(lambda: None, decoder_output)
        streaming_state = decoder_output['streaming_state']
        output = decoder_output['hidden']

        # build the output dict based on decoder output
        output_dict = defaultdict(lambda: None, decoder_output)
        output_dict['hidden'] = output
        output_dict['context'] = context
        output_dict['src_mask'] = encoder_output['src_mask']
        output_dict['src'] = src
        output_dict['target_mask'] = target_mask
        output_dict['streaming_state'] = streaming_state
        output_dict['target'] = batch.get('target_output')
        # output_dict['lid_logits'] = decoder_output['lid_logits']

        # final layer: computing softmax
        if self.training and nce:
            output_dict = self.generator[0](output_dict)
        else:
            logprobs = self.generator[0](output_dict)['logits']
            output_dict['logprobs'] = logprobs

        # Mirror network: reverse the target sequence and perform backward language model
        if mirror:
            # tgt_reverse = torch.flip(batch.get('target_input'), (0, ))
            tgt_pos = torch.flip(batch.get('target_pos'), (0,))
            tgt_reverse = torch.flip(batch.get('target'), (0,))
            tgt_reverse_input = tgt_reverse[:-1]
            tgt_reverse_output = tgt_reverse[1:]

            tgt_reverse_input = tgt_reverse_input.transpose(0, 1)
            # perform an additional backward pass
            reverse_decoder_output = self.mirror_decoder(tgt_reverse_input, context, src, src_lang=src_lang,
                                                         tgt_lang=tgt_lang, input_pos=tgt_pos)

            reverse_decoder_output['src'] = src
            reverse_decoder_output['context'] = context
            reverse_decoder_output['target_mask'] = target_mask

            reverse_logprobs = self.mirror_generator[0](reverse_decoder_output)['logits']

            output_dict['reverse_target'] = tgt_reverse_output
            output_dict['reverse_hidden'] = reverse_decoder_output['hidden']
            output_dict['reverse_logprobs'] = reverse_logprobs
            output_dict['target_input'] = batch.get('target_input')
            output_dict['target_lengths'] = batch.tgt_lengths

            # learn weights for mapping (g in the paper)
            output_dict['hidden'] = self.mirror_g(output_dict['hidden'])

        # Reconstruction network
        if self.reconstruct:
            bos = org_tgt[0].unsqueeze(0)  # 1 x B
            src_input = torch.cat([bos, org_src[:-1]], dim=0)  # T x B
            src_output = org_src

            src_input = src_input.transpose(0, 1)
            rec_context = self.rec_linear(output_dict['hidden'])  # T x B x H
            rec_decoder_output = self.rec_decoder(src_input, rec_context, tgt, tgt_lang=src_lang, input_pos=src_pos)
            rec_output = rec_decoder_output['hidden']
            rec_logprobs = self.rec_generator[0](rec_decoder_output)['logits']

            output_dict['rec_logprobs'] = rec_logprobs
            output_dict['rec_hidden'] = rec_output
            output_dict['reconstruct'] = True
            output_dict['rec_target'] = src_output
        else:
            output_dict['reconstruct'] = False

        # compute the logits for each encoder step
        if self.ctc:
            output_dict['encoder_logits'] = self.ctc_linear(output_dict['context'])

        return output_dict

    def load_encoder_weights(self, pretrained_model):

        pretrained_model.encoder.language_embedding = None

        enc_language_embedding = self.encoder.language_embedding
        self.encoder.language_embedding = None
        encoder_state_dict = pretrained_model.encoder.state_dict()

        self.encoder.load_state_dict(encoder_state_dict)
        self.encoder.language_embedding = enc_language_embedding

    def decode(self, batch, pretrained_layer_states=None):
        """
        :param batch: (onmt.Dataset.Batch) an object containing tensors needed for training
        :return: gold_scores (torch.Tensor) log probs for each sentence
                 gold_words  (Int) the total number of non-padded tokens
                 allgold_scores (list of Tensors) log probs for each word in the sentence
        """

        src = batch.get('source')
        src_pos = batch.get('source_pos')
        tgt_input = batch.get('target_input')
        tgt_output = batch.get('target_output')
        tgt_pos = batch.get('target_pos')
        # tgt_atb = batch.get('target_atb')  # a dictionary of attributes
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')

        # transpose to have batch first
        src = src.transpose(0, 1)
        tgt_input = tgt_input.transpose(0, 1)
        batch_size = tgt_input.size(0)

        context = self.encoder(src, input_pos=src_pos, input_lang=src_lang,
                               pretrained_layer_states=pretrained_layer_states)['context']

        if hasattr(self, 'autoencoder') and self.autoencoder \
                and self.autoencoder.representation == "EncoderHiddenState":
            context = self.autoencoder.autocode(context)

        gold_scores = context.new(batch_size).zero_()
        gold_words = 0
        allgold_scores = list()
        decoder_output = self.decoder(tgt_input, context, src, tgt_lang=tgt_lang, src_lang=src_lang,
                                      input_pos=tgt_pos)['hidden']

        output = decoder_output

        if hasattr(self, 'autoencoder') and self.autoencoder and \
                self.autoencoder.representation == "DecoderHiddenState":
            output = self.autoencoder.autocode(output)

        for dec_t, tgt_t in zip(output, tgt_output):

            dec_out = defaultdict(lambda: None)
            dec_out['hidden'] = dec_t.unsqueeze(0)
            dec_out['src'] = src
            dec_out['context'] = context

            if isinstance(self.generator, nn.ModuleList):
                dec_out = self.generator[0](dec_out)
                # gen_t = self.generator[0](dec_out)['logits']
            else:
                dec_out = self.generator(dec_out)
            gen_t = dec_out['logits']
            if dec_out['softmaxed'] is False:
                gen_t = F.log_softmax(gen_t, dim=-1, dtype=torch.float32)
            gen_t = gen_t.squeeze(0)
            tgt_t = tgt_t.unsqueeze(1)
            scores = gen_t.gather(1, tgt_t)
            scores.masked_fill_(tgt_t.eq(onmt.constants.TGT_PAD), 0)
            gold_scores += scores.squeeze(1).type_as(gold_scores)
            gold_words += tgt_t.ne(onmt.constants.TGT_PAD).sum().item()
            allgold_scores.append(scores.squeeze(1).type_as(gold_scores))

        return gold_words, gold_scores, allgold_scores

    def renew_buffer(self, new_len):
        self.decoder.renew_buffer(new_len)

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

        # squeeze to remove the time step dimension
        if isinstance(self.generator, nn.ModuleList):
            output_dict = self.generator[0](output_dict)
        else:
            output_dict = self.generator(output_dict)
        log_prob = output_dict['logits'].squeeze(0)

        # the key 'softmaxed' should be included in generators.
        # The 'normal linear + CE' doesn't need softmax
        if output_dict['softmaxed'] is False:
            log_prob = F.log_softmax(log_prob, dim=-1, dtype=torch.float32)

        coverage = output_dict['coverage']

        try:
            last_coverage = coverage[:, -1, :].squeeze(1)
        except TypeError:
            last_coverage = None

        output_dict['log_prob'] = log_prob
        output_dict['coverage'] = last_coverage

        return output_dict

    def create_decoder_state(self, batch, beam_size=1, type=1, buffering=True,
                             pretrained_classifier=None, pretrained_layer_states=None, **kwargs):
        """
        Generate a new decoder state based on the batch input
        :param pretrained_classifier: model to create mixtures
        :param buffering:
        :param type:
        :param batch: Batch object (may not contain target during decoding)
        :param beam_size: Size of beam used in beam search
        :return:
        """
        src = batch.get('source')
        src_pos = batch.get('source_pos')
        tgt_atb = batch.get('target_atb')
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')

        src_transposed = src.transpose(0, 1)

        if pretrained_classifier is not None:
            mixture = pretrained_classifier(src_transposed)

        encoder_output = self.encoder(src_transposed, input_pos=src_pos, input_lang=src_lang, atb=tgt_atb,
                                      pretrained_layer_states=pretrained_layer_states)

        print("[INFO] create Transformer decoding state with buffering", buffering)
        decoder_state = TransformerDecodingState(src, tgt_lang, encoder_output['context'], src_lang,
                                                 beam_size=beam_size, model_size=self.model_size,
                                                 type=type, buffering=buffering, tgt_atb=tgt_atb)

        return decoder_state

    def init_stream(self):

        pass

    def set_memory_size(self, src_memory_size, tgt_memory_size):

        pass


class TransformerDecodingState(DecoderState):

    def __init__(self, src, tgt_lang, context, src_lang, beam_size=1, model_size=512, type=2,
                 cloning=True, buffering=False, src_mask=None, tgt_atb=None,
                 dec_pretrained_model="", ):

        """
        :param src:
        :param tgt_lang:
        :param context:
        :param src_lang:
        :param beam_size:
        :param model_size:
        :param type: Type 1 is for old translation code. Type 2 is for fast buffering. (Type 2 default).
        :param cloning:
        :param buffering:
        """

        self.beam_size = beam_size
        self.model_size = model_size
        self.attention_buffers = dict()
        self.buffering = buffering
        self.dec_pretrained_model = dec_pretrained_model
        self.tgt_atb = tgt_atb

        if type == 1:
            # if audio only take one dimension since only used for mask
            raise NotImplementedError
            # self.original_src = src  # TxBxC
            # self.concat_input_seq = True
            #
            # if src is not None:
            #     if src.dim() == 3:
            #         # print(self.src.size())
            #         self.src = src.narrow(2, 0, 1).squeeze(2).repeat(1, beam_size)
            #         # self.src = src.repeat(1, beam_size, 1)
            #         # print(self.src.size())
            #         # self.src = src.repeat(1, beam_size, 1) # T x Bb x c
            #     else:
            #         self.src = src.repeat(1, beam_size)
            # else:
            #     self.src = None
            #
            # if context is not None:
            #     self.context = context.repeat(1, beam_size, 1)
            # else:
            #     self.context = None
            #
            # self.input_seq = None
            # self.src_lang = src_lang
            # self.tgt_lang = tgt_lang

        elif type == 2:
            bsz = src.size(1)  # src is T x B
            new_order = torch.arange(bsz).view(-1, 1).repeat(1, self.beam_size).view(-1)
            new_order = new_order.to(src.device)

            if cloning:
                self.src = src.index_select(1, new_order)  # because src is time first

                if context is not None:
                    self.context = context.index_select(1, new_order)
                else:
                    self.context = None

                if src_mask is not None:
                    self.src_mask = src_mask.index_select(0, new_order)
                else:
                    self.src_mask = None
            else:
                self.context = context
                self.src = src
                # self.src_mask = src_mask

            self.concat_input_seq = False
            self.tgt_lang = tgt_lang
            self.src_lang = src_lang

        else:
            raise NotImplementedError

    def update_attention_buffer(self, buffer, layer):

        self.attention_buffers[layer] = buffer  # dict of 2 keys (k, v) : T x B x H

    def update_beam(self, beam, b, remaining_sents, idx):

        if self.beam_size == 1:
            return

        for tensor in [self.src, self.input_seq]:

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

        def update_active_with_hidden(t):
            if t is None:
                return t
            dim = t.size(-1)
            # select only the remaining active sentences
            view = t.data.view(-1, remaining_sents, dim)
            new_size = list(t.size())
            new_size[-2] = new_size[-2] * len(active_idx) // remaining_sents
            return view.index_select(1, active_idx).view(*new_size)

        def update_active_without_hidden(t):
            if t is None:
                return t
            view = t.view(-1, remaining_sents)
            new_size = list(t.size())
            new_size[-1] = new_size[-1] * len(active_idx) // remaining_sents
            new_t = view.index_select(1, active_idx).view(*new_size)
            return new_t

        self.context = update_active_with_hidden(self.context)

        self.input_seq = update_active_without_hidden(self.input_seq)

        if self.src.dim() == 2:
            self.src = update_active_without_hidden(self.src)
        elif self.src.dim() == 3:
            t = self.src
            dim = t.size(-1)
            view = t.view(-1, remaining_sents, dim)
            new_size = list(t.size())
            new_size[-2] = new_size[-2] * len(active_idx) // remaining_sents
            new_t = view.index_select(1, active_idx).view(*new_size)
            self.src = new_t

        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]

            for k in buffer_:
                buffer_[k] = update_active_with_hidden(buffer_[k])

    # For the new decoder version only
    def _reorder_incremental_state(self, reorder_state):

        if self.context is not None:
            self.context = self.context.index_select(1, reorder_state)

        if self.src_mask is not None:
            self.src_mask = self.src_mask.index_select(0, reorder_state)
        self.src = self.src.index_select(1, reorder_state)

        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]
            if buffer_ is not None:
                for k in buffer_.keys():
                    t_, br_, d_ = buffer_[k].size()
                    buffer_[k] = buffer_[k].index_select(1, reorder_state)
                    # if not self.dec_pretrained_model:
                    #     buffer_[k] = buffer_[k].index_select(1, reorder_state)  # beam/batch is the 2nd dim
                    # elif self.dec_pretrained_model in ["bert", "roberta", "bart"]:
                    #     buffer_[k] = buffer_[k].index_select(0, reorder_state)  # beam/batch is the first dim
                    # elif self.dec_pretrained_model in ["mbart", "mbart50"]:
                    #     buffer_[k] = buffer_[k].index_select(1, reorder_state)  # beam/batch is the 2nd dim
                    # else:
                    #     print("Warning: check dec_pretrained_model type")
                    #     raise NotImplementedError
