import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.models.transformers import TransformerEncoder, TransformerDecoder, Transformer, TransformerDecodingState
from onmt.modules.sinusoidal_positional_encoding import SinusoidalPositionalEmbedding
import onmt
from onmt.modules.base_seq2seq import NMTModel, DecoderState
from onmt.models.speech_recognizer.lstm import SpeechLSTMDecoder, LSTMDecodingState
from onmt.modules.convolution import Conv2dSubsampling
from onmt.models.transformer_layers import PrePostProcessing
from .relative_transformer_layers import RelativeTransformerEncoderLayer, RelativeTransformerDecoderLayer
from .conformer_layers import ConformerEncoderLayer
from onmt.utils import flip, expected_length
from collections import defaultdict
import math
import sys


class ConformerEncoder(TransformerEncoder):

    def __init__(self, opt, dicts, positional_encoder, encoder_type='text', language_embeddings=None):
        self.death_rate = opt.death_rate
        self.learnable_position_encoding = opt.learnable_position_encoding
        self.layer_modules = list()
        self.unidirectional = opt.unidirectional
        self.reversible = opt.src_reversible
        self.n_heads = opt.n_heads

        # build_modules will be called from the inherited constructor
        super().__init__(opt, dicts, positional_encoder, encoder_type, language_embeddings)

        # position encoding sin/cos
        self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)

        # self.audio_trans = Conv2dSubsampling(opt.input_size, opt.model_size)
        channels = self.channels
        feature_size = opt.input_size
        cnn = [nn.Conv2d(channels, 32, kernel_size=(3, 3), stride=2), nn.ReLU(True), nn.BatchNorm2d(32),
               nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2), nn.ReLU(True), nn.BatchNorm2d(32)]
        # cnn = [nn.Conv2d(channels, 32, kernel_size=(3, 3), stride=2), nn.ReLU(True),
        #        nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2), nn.ReLU(True)]

        nn.init.kaiming_normal_(cnn[0].weight, nonlinearity="relu")
        nn.init.kaiming_normal_(cnn[3].weight, nonlinearity="relu")

        feat_size = (((feature_size // channels) - 3) // 4) * 32
        # cnn.append()
        self.audio_trans = nn.Sequential(*cnn)
        self.linear_trans = nn.Linear(feat_size, self.model_size)

        self.d_head = self.model_size // self.n_heads

    def build_modules(self):

        e_length = expected_length(self.layers, self.death_rate)
        print("* Conformer Encoder with %.2f expected layers" % e_length)
        if self.unidirectional:
            print("* Running a unidirectional Encoder.")

        self.layer_modules = nn.ModuleList()

        for _l in range(self.layers):
            # linearly decay the death rate
            death_r = (_l + 1.0) / self.layers * self.death_rate

            block = ConformerEncoderLayer(self.opt, death_rate=death_r)
            self.layer_modules.append(block)

    def forward(self, input, input_pos=None, input_lang=None, streaming=False, **kwargs):
        """
        :param input: [B x T x Input_Size]
        :param input_pos: [B x T] positions
        :param input_lang: [B] language ids of each sample
        :param streaming: connect different segments in transformer-xl style
        :param kwargs:
        :return:
        """

        long_mask = input.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD)
        input = input.narrow(2, 1, input.size(2) - 1)

        # first subsampling
        input = input.view(input.size(0), input.size(1), -1, self.channels)
        input = input.permute(0, 3, 1, 2)  # [bsz, channels, time, f]
        input = self.audio_trans(input)
        input = input.permute(0, 2, 1, 3).contiguous()
        input = input.view(input.size(0), input.size(1), -1)
        input = self.linear_trans(input)
        emb = input

        mask_src = long_mask[:, 0:emb.size(1) * 4:4].transpose(0, 1).unsqueeze(0)
        dec_attn_mask = None

        emb = emb.transpose(0, 1)
        input = input.transpose(0, 1)
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

        if onmt.constants.torch_version >= 1.2:
            mask_src = mask_src.bool()

        """ Scale the emb by sqrt(d_model) """
        # emb = emb * math.sqrt(self.model_size)

        """ Adding positional encoding """
        qlen = input.size(0)
        klen = qlen + mem_len

        # Asynchronous positions: 2K+1 positions instead of K+1
        if self.unidirectional:
            pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)
        else:
            pos = torch.arange(klen - 1, -klen, -1.0, device=emb.device, dtype=emb.dtype)

        # pos_emb has size 2T+1 x 1 x H
        pos_emb = self.positional_encoder(pos, bsz=input.size(1))

        if self.learnable_position_encoding:
            raise NotImplementedError

        context = emb

        # Apply dropout to pos_emb
        # context = self.preprocess_layer(context)

        pos_emb = self.preprocess_layer(pos_emb)

        for i, layer in enumerate(self.layer_modules):
            # src_len x batch_size x d_model

            context = layer(context, pos_emb, mask_src)

        # final layer norm
        context = self.postprocess_layer(context)

        output_dict = defaultdict(lambda: None, {'context': context, 'src_mask': dec_attn_mask, 'src': input})

        return output_dict


class Conformer(NMTModel):

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

        encoder_output = self.encoder(src)
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

        # compute the logits for each encoder step
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

        # TxB -> BxT
        src_transposed = src.transpose(0, 1)

        encoder_output = self.encoder(src_transposed)
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