import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import defaultdict
import math
import onmt
from onmt.modules.base_seq2seq import NMTModel, DecoderState
from onmt.models.transformer_layers import PrePostProcessing
from onmt.modules.attention import MultiHeadAttention
from onmt.modules.optimized.encdec_attention import EncdecMultiheadAttn
from onmt.modules.dropout import embedded_dropout, switchout
from onmt.models.transformer_layers import EncoderLayer, DecoderLayer
import random
import time


class SpeechLSTMEncoder(nn.Module):

    def __init__(self, opt, embedding, encoder_type='audio'):
        super(SpeechLSTMEncoder, self).__init__()
        self.opt = opt
        self.model_size = opt.model_size

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

        self.switchout = 0.0  # for speech it has to be
        self.varitional_dropout = opt.variational_dropout
        self.use_language_embedding = opt.use_language_embedding
        self.language_embedding_type = opt.language_embedding_type

        self.time = opt.time
        self.lsh_src_attention = opt.lsh_src_attention
        self.reversible = opt.src_reversible
        self.multilingual_factorized_weights = opt.multilingual_factorized_weights
        self.mfw_rank = opt.mfw_rank

        feature_size = opt.input_size
        self.channels = 1

        if opt.upsampling:
            feature_size = feature_size // 4

        if not self.cnn_downsampling:
            self.audio_trans = nn.Linear(feature_size, self.model_size)
            torch.nn.init.xavier_uniform_(self.audio_trans.weight)
        else:
            channels = self.channels
            cnn = [nn.Conv2d(channels, 32, kernel_size=(3, 3), stride=2), nn.ReLU(True), nn.BatchNorm2d(32),
                   nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2), nn.ReLU(True), nn.BatchNorm2d(32)]

            feat_size = (((feature_size // channels) - 3) // 4) * 32
            # cnn.append()
            self.audio_trans = nn.Sequential(*cnn)
            self.linear_trans = nn.Linear(feat_size, self.model_size)

        self.unidirect = False

        self.rnn = nn.LSTM(input_size=self.model_size, hidden_size=self.model_size, num_layers=self.layers,
                           bidirectional=(not self.unidirect), bias=False, dropout=self.dropout, batch_first=True)

        if self.multilingual_factorized_weights:
            from onmt.modules.weight_control_lstm import WeightFactoredLSTM
            self.rnn = WeightFactoredLSTM(self.rnn, dropout=opt.weight_drop, n_languages=opt.n_languages,
                                          rank=self.mfw_rank)

        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d',
                                                  variational=self.varitional_dropout)
        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')

    def rnn_fwd(self, seq, mask, hid, src_lang=None):
        """
        :param src_lang:
        :param seq:
        :param mask:
        :param hid:
        :return:
        """
        if mask is not None:
            lengths = mask.sum(-1).float()
            seq = pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=False)
            if self.multilingual_factorized_weights:
                seq, hid = self.rnn(seq, hid, indices=src_lang)
            else:
                seq, hid = self.rnn(seq, hid)
            seq = pad_packed_sequence(seq, batch_first=True)[0]
        else:
            if self.multilingual_factorized_weights:
                seq, hid = self.rnn(seq, hid, indices=src_lang)
            else:
                seq, hid = self.rnn(seq, hid)

        return seq, hid

    def forward(self, input, input_pos=None, input_lang=None, hid=None, **kwargs):

        if not self.cnn_downsampling:
            mask_src = input.narrow(2, 0, 1).squeeze(2).gt(onmt.constants.PAD)
            dec_attn_mask = input.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD)
            input = input.narrow(2, 1, input.size(2) - 1)
            emb = self.audio_trans(input.contiguous().view(-1, input.size(2))).view(input.size(0),
                                                                                    input.size(1), -1)
            emb = emb.type_as(input)
        else:
            long_mask = input.narrow(2, 0, 1).squeeze(2).gt(onmt.constants.PAD)

            input = input.narrow(2, 1, input.size(2) - 1)
            # first resizing to fit the CNN format
            input = input.view(input.size(0), input.size(1), -1, self.channels)
            input = input.permute(0, 3, 1, 2)

            input = self.audio_trans(input)
            input = input.permute(0, 2, 1, 3).contiguous()
            input = input.view(input.size(0), input.size(1), -1)
            input = self.linear_trans(input)

            mask_src = long_mask[:, 0:input.size(1) * 4:4]
            dec_attn_mask = ~mask_src
            # the size seems to be B x T ?
            emb = input

        seq, hid = self.rnn_fwd(emb, mask_src, hid, src_lang=input_lang)

        if not self.unidirect:
            hidden_size = seq.size(2) // 2
            seq = seq[:, :, :hidden_size] + seq[:, :, hidden_size:]

        # layer norm
        seq = self.postprocess_layer(seq)

        output_dict = {'context': seq.transpose(0, 1), 'src_mask': dec_attn_mask}

        return output_dict


class SpeechLSTMDecoder(nn.Module):
    def __init__(self, opt, embedding, language_embeddings=None, **kwargs):
        super(SpeechLSTMDecoder, self).__init__()

        # Keep for reference

        # Define layers
        self.model_size = opt.model_size
        self.layers = opt.layers
        self.dropout = opt.dropout

        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout
        self.variational_dropout = opt.variational_dropout
        self.multilingual_factorized_weights = opt.multilingual_factorized_weights
        self.mfw_rank = opt.mfw_rank
        self.encoder_type = opt.encoder_type
        self.n_languages = opt.n_languages

        self.lstm = nn.LSTM(self.model_size, self.model_size, self.layers, dropout=self.dropout, batch_first=True)
        if self.multilingual_factorized_weights:
            from onmt.modules.weight_control_lstm import WeightFactoredLSTM
            self.lstm = WeightFactoredLSTM(self.lstm, dropout=opt.weight_drop, n_languages=opt.n_languages,
                                           rank=self.mfw_rank)

        self.fast_xattention = opt.fast_xattention
        self.n_head = 1  # fixed to always use 1 head
        # also fix attention dropout to 0.0

        if self.multilingual_factorized_weights:
            self.fast_xattention = True
            from onmt.modules.multilingual_factorized.encdec_attention import MFWEncdecMultiheadAttn
            self.multihead_tgt = MFWEncdecMultiheadAttn(self.n_head, opt.model_size, 0.0, n_languages=opt.n_languages,
                                                        rank=opt.mfw_rank, weight_drop=0.0)
        else:
            if opt.fast_xattention:
                self.multihead_tgt = EncdecMultiheadAttn(self.n_head, opt.model_size, 0.0)
            else:
                self.multihead_tgt = MultiHeadAttention(self.n_head, opt.model_size, attn_p=0.0, share=3)

        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d',
                                                  variational=self.variational_dropout)

        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')
        self.preprocess_attn = PrePostProcessing(self.model_size, 0, sequence='n')

        self.word_lut = embedding

        self.encoder_cnn_downsampling = opt.cnn_downsampling
        self.language_embeddings = language_embeddings
        self.use_language_embedding = opt.use_language_embedding
        self.language_embedding_type = opt.language_embedding_type

        if self.language_embedding_type == 'concat':
            self.projector = nn.Linear(opt.model_size * 2, opt.model_size)

        print("* Create LSTM Decoder with %d layers." % self.layers)

    def process_embedding(self, input, input_lang=None):

        return input

    def step(self, input, decoder_state, **kwargs):
        context = decoder_state.context
        buffer = decoder_state.lstm_buffer
        attn_buffer = decoder_state.attention_buffers
        hid = buffer["hidden_state"]
        cell = buffer["cell_state"]
        tgt_lang = decoder_state.tgt_lang

        buffering = decoder_state.buffering

        if hid is not None:
            hid_cell = (hid, cell)
        else:
            hid_cell = None

        lang = decoder_state.tgt_lang

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

        emb = self.word_lut(input_)

        emb = emb * math.sqrt(self.model_size)

        if self.use_language_embedding:
            # print("Using language embedding")
            lang_emb = self.language_embeddings(lang)  # B x H or 1 x H
            if self.language_embedding_type == 'sum':

                dec_emb = emb + lang_emb.unsqueeze(1)
            elif self.language_embedding_type == 'concat':
                # replace the bos embedding with the language
                bos_emb = lang_emb.expand_as(emb[0])
                emb[0] = bos_emb

                lang_emb = lang_emb.unsqueeze(0).expand_as(emb)
                concat_emb = torch.cat([emb, lang_emb], dim=-1)
                dec_emb = torch.relu(self.projector(concat_emb))
            else:
                raise NotImplementedError

        if context is not None:
            if self.encoder_type == "audio":
                if src.data.dim() == 3:
                    if self.encoder_cnn_downsampling:
                        long_mask = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD)
                        mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
                    else:
                        mask_src = src.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD).unsqueeze(1)
                elif self.encoder_cnn_downsampling:
                    long_mask = src.eq(onmt.constants.PAD)
                    mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
                else:
                    mask_src = src.eq(onmt.constants.PAD).unsqueeze(1)
            else:
                mask_src = src.eq(onmt.constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        # if input_.size(0) > 1 and input_.size(1) > 1:
        #
        #     lengths = input.gt(onmt.constants.PAD).sum(-1)
        #
        #     dec_in = pack_padded_sequence(dec_emb, lengths, batch_first=True, enforce_sorted=False)
        #
        #     dec_out, hidden = self.lstm(dec_in, hid_cell)
        #     dec_out = pad_packed_sequence(dec_out, batch_first=True)[0]
        # else:
        if self.multilingual_factorized_weights:
            dec_out, hid_cell = self.lstm(dec_emb, hid_cell, indices=tgt_lang)
        else:
            dec_out, hid_cell = self.lstm(dec_emb, hid_cell)

        decoder_state.update_lstm_buffer(hid_cell)

        lt = input_.size(1)
        attn_mask = mask_src.expand(-1, lt, -1) if not self.fast_xattention else mask_src.squeeze(1)
        # dec_out = self.postprocess_layer(dec_out)

        dec_out = self.preprocess_attn(dec_out)
        dec_out = dec_out.transpose(0, 1)

        if buffering:
            buffer = attn_buffer[0]
            if buffer is None:
                buffer = dict()
            if self.multilingual_factorized_weights:
                output, coverage = self.multihead_tgt(dec_out, context, context, tgt_lang, tgt_lang, attn_mask,
                                                      incremental=True, incremental_cache=buffer)
            else:
                output, coverage = self.multihead_tgt(dec_out, context, context, attn_mask,
                                                      incremental=True, incremental_cache=buffer)

            decoder_state.update_attention_buffer(buffer, 0)
        else:
            if self.multilingual_factorized_weights:
                output, coverage = self.multihead_tgt(dec_out, context, context, tgt_lang, tgt_lang, attn_mask)
            else:
                output, coverage = self.multihead_tgt(dec_out, context, context, attn_mask)

        output = (output + dec_out)
        output = self.postprocess_layer(output)

        output_dict = defaultdict(lambda: None, {'hidden': output, 'coverage': coverage, 'context': context})

        return output_dict

    def forward(self, dec_seq, enc_out, src, tgt_lang=None, hid=None, **kwargs):

        emb = embedded_dropout(self.word_lut, dec_seq, dropout=self.word_dropout if self.training else 0)
        emb = emb * math.sqrt(self.model_size)

        if self.use_language_embedding:
            # print("Using language embedding")
            lang_emb = self.language_embeddings(tgt_lang)  # B x H or 1 x H
            if self.language_embedding_type == 'sum':

                dec_emb = emb + lang_emb.unsqueeze(1)
            elif self.language_embedding_type == 'concat':
                # replace the bos embedding with the language
                bos_emb = lang_emb.expand_as(emb[0])
                emb[0] = bos_emb

                lang_emb = lang_emb.unsqueeze(0).expand_as(emb)
                concat_emb = torch.cat([emb, lang_emb], dim=-1)
                dec_emb = torch.relu(self.projector(concat_emb))
            else:
                raise NotImplementedError

        if enc_out is not None:
            if self.encoder_type == "audio":
                if not self.encoder_cnn_downsampling:
                    mask_src = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD).unsqueeze(1)
                else:
                    long_mask = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD)
                    mask_src = long_mask[:, 0: enc_out.size(0) * 4:4].unsqueeze(1)
            else:

                mask_src = src.data.eq(onmt.constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        # if dec_seq.size(0) > 1 and dec_seq.size(1) > 1:
        #     lengths = dec_seq.gt(onmt.constants.PAD).sum(-1)
        #     dec_in = pack_padded_sequence(dec_emb, lengths, batch_first=True, enforce_sorted=False)
        #     dec_out, hid = self.lstm(dec_in, hid)
        #     dec_out = pad_packed_sequence(dec_out, batch_first=True)[0]
        # else:
        if self.multilingual_factorized_weights:
            dec_out, hid = self.lstm(dec_emb, hid, indices=tgt_lang)
        else:
            dec_out, hid = self.lstm(dec_emb, hid)

        lt = dec_seq.size(1)
        attn_mask = mask_src.expand(-1, lt, -1) if not self.fast_xattention else mask_src.squeeze(1)
        # dec_out = self.postprocess_layer(dec_out)
        dec_out = self.preprocess_attn(dec_out)
        dec_out = dec_out.transpose(0, 1).contiguous()
        enc_out = enc_out.contiguous()

        if self.multilingual_factorized_weights:
            output, coverage = self.multihead_tgt(dec_out, enc_out, enc_out, tgt_lang, tgt_lang, attn_mask)
        else:
            output, coverage = self.multihead_tgt(dec_out, enc_out, enc_out, attn_mask)

        output = (output + dec_out)
        output = self.postprocess_layer(output)

        output_dict = defaultdict(lambda: None, {'hidden': output, 'coverage': coverage, 'context': enc_out})
        return output_dict


class SpeechLSTMSeq2Seq(NMTModel):

    def __init__(self, encoder, decoder, generator=None, rec_decoder=None, rec_generator=None,
                 mirror=False, ctc=False):
        super().__init__(encoder, decoder, generator, rec_decoder, rec_generator, ctc=ctc)

        self.model_size = self.decoder.model_size
        self.tgt_vocab_size = self.decoder.word_lut.weight.size(0)
        if self.encoder.input_type == 'text':
            self.src_vocab_size = self.encoder.word_lut.weight.size(0)
        else:
            self.src_vocab_size = 0

        if self.ctc:
            self.ctc_linear = nn.Linear(encoder.model_size, self.tgt_vocab_size)

    def reset_states(self):
        return

    def forward(self, batch, target_mask=None, streaming=False, zero_encoder=False,
                mirror=False, streaming_state=None, nce=False):

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

        encoder_output = self.encoder(src, input_pos=src_pos, input_lang=src_lang, src_lengths=src_lengths)
        encoder_output = defaultdict(lambda: None, encoder_output)

        context = encoder_output['context']

        if zero_encoder:
            context.zero_()

        src_mask = encoder_output['src_mask']
        decoder_output = self.decoder(tgt, context, src,
                                      tgt_lang=tgt_lang, input_pos=tgt_pos, streaming=streaming,
                                      src_lengths=src_lengths, tgt_lengths=tgt_lengths,
                                      streaming_state=streaming_state)

        decoder_output = defaultdict(lambda: None, decoder_output)
        output = decoder_output['hidden']

        output_dict = defaultdict(lambda: None, decoder_output)
        output_dict['hidden'] = output
        output_dict['context'] = context
        output_dict['src_mask'] = encoder_output['src_mask']
        output_dict['src'] = src
        output_dict['target_mask'] = target_mask
        output_dict['reconstruct'] = None
        output_dict['target'] = batch.get('target_output')

        if self.training and nce:
            output_dict = self.generator[0](output_dict)
        else:
            logprobs = self.generator[0](output_dict)['logits']
            output_dict['logprobs'] = logprobs

        if self.ctc:
            output_dict['encoder_logits'] = self.ctc_linear(output_dict['context'])

        return output_dict

    def step(self, input_t, decoder_state):

        output_dict = self.decoder.step(input_t, decoder_state)
        output_dict['src'] = decoder_state.src.transpose(0, 1)

        # squeeze to remove the time step dimension
        log_prob = self.generator[0](output_dict)['logits'].squeeze(0)
        log_prob = F.log_softmax(log_prob, dim=-1, dtype=torch.float32)

        coverage = output_dict['coverage']
        last_coverage = coverage[:, -1, :].squeeze(1)

        output_dict['log_prob'] = log_prob
        output_dict['coverage'] = last_coverage

        return output_dict

    def create_decoder_state(self, batch, beam_size=1, type=1, buffering=True, **kwargs):
        """
        Generate a new decoder state based on the batch input
        :param buffering:
        :param streaming:
        :param type:
        :param batch: Batch object (may not contain target during decoding)
        :param beam_size: Size of beam used in beam search
        :return:
        """
        src = batch.get('source')
        src_pos = batch.get('source_pos')
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')
        src_lengths = batch.src_lengths

        # TxB -> BxT
        src_transposed = src.transpose(0, 1)

        encoder_output = self.encoder(src_transposed, input_pos=src_pos, input_lang=src_lang, src_lengths=src_lengths)
        decoder_state = LSTMDecodingState(src, tgt_lang, encoder_output['context'],
                                                 beam_size=beam_size, model_size=self.model_size,
                                                 type=type, buffering=buffering)

        return decoder_state

    def decode(self, batch):
        src = batch.get('source')
        src_pos = batch.get('source_pos')
        tgt_input = batch.get('target_input')
        tgt_output = batch.get('target_output')
        tgt_pos = batch.get('target_pos')
        # tgt_atb = batch.get('target_atb')  # a dictionary of attributes
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')

        src = src.transpose(0, 1)
        tgt_input = tgt_input.transpose(0, 1)
        batch_size = tgt_input.size(0)

        context = self.encoder(src)['context']

        gold_scores = context.new(batch_size).zero_()
        gold_words = 0
        allgold_scores = list()

        decoder_output = self.decoder(tgt_input, context, src, tgt_lang=tgt_lang, src_lang=src_lang,
                                      input_pos=tgt_pos)['hidden']

        output = decoder_output

        for dec_t, tgt_t in zip(output, tgt_output):

            dec_out = defaultdict(lambda: None)
            dec_out['hidden'] = dec_t.unsqueeze(0)
            dec_out['src'] = src
            dec_out['context'] = context

            if isinstance(self.generator, nn.ModuleList):
                gen_t = self.generator[0](dec_out)['logits']
            else:
                gen_t = self.generator(dec_out)['logits']
            gen_t = F.log_softmax(gen_t, dim=-1, dtype=torch.float32)
            gen_t = gen_t.squeeze(0)
            tgt_t = tgt_t.unsqueeze(1)
            scores = gen_t.gather(1, tgt_t)
            scores.masked_fill_(tgt_t.eq(onmt.constants.PAD), 0)
            gold_scores += scores.squeeze(1).type_as(gold_scores)
            gold_words += tgt_t.ne(onmt.constants.PAD).sum().item()
            allgold_scores.append(scores.squeeze(1).type_as(gold_scores))

        return gold_words, gold_scores, allgold_scores


class LSTMDecodingState(DecoderState):

    def __init__(self, src, tgt_lang, context, beam_size=1, model_size=512, type=2,
                 cloning=True, buffering=False):
        self.beam_size = beam_size
        self.model_size = model_size
        self.lstm_buffer = dict()
        self.lstm_buffer["hidden_state"] = None
        self.lstm_buffer["cell_state"] = None
        self.buffering = buffering
        self.attention_buffers = defaultdict(lambda: None)

        if type == 1:
            # if audio only take one dimension since only used for mask
            # raise NotImplementedError
            self.original_src = src  # TxBxC
            self.concat_input_seq = True

            if src is not None:
                if src.dim() == 3:
                    self.src = src.narrow(2, 0, 1).squeeze(2).repeat(1, beam_size)
                    # self.src = src.repeat(1, beam_size, 1) # T x Bb x c
                else:
                    self.src = src.repeat(1, beam_size)
            else:
                self.src = None

            if context is not None:
                self.context = context.repeat(1, beam_size, 1)
            else:
                self.context = None

            self.input_seq = None
            self.tgt_lang = tgt_lang

        elif type == 2:
            bsz = src.size(1)  # src is T x B
            new_order = torch.arange(bsz).view(-1, 1).repeat(1, self.beam_size).view(-1)
            new_order = new_order.to(src.device)

            if cloning:
                self.src = src.index_select(1, new_order)  # because src is batch first

                if context is not None:
                    self.context = context.index_select(1, new_order)
                else:
                    self.context = None

            else:
                self.context = context
                self.src = src
            self.input_seq = None
            self.concat_input_seq = False
            self.tgt_lang = tgt_lang
        else:
            raise NotImplementedError

    def update_lstm_buffer(self, buffer):

        hid, cell = buffer
        # hid and cell should have size [n_layer, batch_size, hidden_size]

        self.lstm_buffer["hidden_state"] = hid
        self.lstm_buffer["cell_state"] = cell

    def update_attention_buffer(self, buffer, layer):

        self.attention_buffers[layer] = buffer

    def update_beam(self, beam, b, remaining_sents, idx):

        if self.beam_size == 1:
            return
        # print(self.input_seq)
        # print(self.src.shape)
        for tensor in [self.src, self.input_seq]:

            if tensor is None:
                continue

            t_, br = tensor.size()
            sent_states = tensor.view(t_, self.beam_size, remaining_sents)[:, :, idx]

            sent_states.copy_(sent_states.index_select(1, beam[b].getCurrentOrigin()))

        for l in self.lstm_buffer:
            buffer_ = self.lstm_buffer[l]

            t_, br_, d_ = buffer_.size()
            sent_states = buffer_.view(t_, self.beam_size, remaining_sents, d_)[:, :, idx, :]

            sent_states.data.copy_(sent_states.data.index_select(1, beam[b].getCurrentOrigin()))

        for l in self.attention_buffers:
            buffers = self.attention_buffers[l]
            if buffers is not None:
                for k in buffers.keys():
                    buffer_ = buffers[k]
                    t_, br_, d_ = buffer_.size()
                    sent_states = buffer_.view(t_, self.beam_size, remaining_sents, d_)[:, :, idx, :]

                    sent_states.data.copy_(sent_states.data.index_select(1, beam[b].getCurrentOrigin()))

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

        for l in self.lstm_buffer:
            buffer_ = self.lstm_buffer[l]

            buffer = update_active_with_hidden(buffer_)

            self.lstm_buffer[l] = buffer

        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]
            if buffer_ is not None:
                for k in buffer_.keys():
                    buffer_[k] = update_active_with_hidden(buffer_[k])

    # For the new decoder version only
    def _reorder_incremental_state(self, reorder_state):

        if self.context is not None:
            self.context = self.context.index_select(1, reorder_state)

        # if self.src_mask is not None:
        #     self.src_mask = self.src_mask.index_select(0, reorder_state)
        self.src = self.src.index_select(1, reorder_state)

        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]
            if buffer_ is not None:
                for k in buffer_.keys():
                    t_, br_, d_ = buffer_[k].size()
                    buffer_[k] = buffer_[k].index_select(1, reorder_state)  # 1 for time first

        for k in self.lstm_buffer:
            buffer_ = self.lstm_buffer[k]
            if buffer_ is not None:
                self.lstm_buffer[k] = buffer_.index_select(1, reorder_state)  # 1 because the first dim is n_layer

