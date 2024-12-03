import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from onmt.models.transformers import Transformer, TransformerDecodingState, TransformerDecodingStateMemory
from collections import defaultdict
import onmt
from onmt.modules.optimized.linear import Linear
import math

import copy
import numpy as np

from itertools import groupby

from .wav2vec2 import Wav2vecTransformer

class WhisperModel(Wav2vecTransformer):

    def __init__(self, encoder, decoder, generator=None,
                 ctc=False,
                 mirror=False,
                 **kwargs):
        super().__init__(encoder, decoder, generator, mirror=mirror, ctc=False)
        self.generator = generator

        self.src_vocab_size = 0
        self.model_size = encoder.model_size

        # this model cannot use CTC
        self.ctc = False
        self.ctc_compress = False
        self.ctc_char = False

        self.model_size = None
        self.tgt_vocab_size = 0

        # TODO: add ctc loss function?

    def forward(self, batch, zero_encoder=False, factorize=False, target_mask=None, mirror=False,
                checkpointing_ffn=False,
                checkpointing_cross_attn=False,
                checkpointing_self_attn=False,
                ctc_loss_function=None,
                ctc_labels=None,
                grad_scaler=None,
                ctc_coeff=None,
                **kwargs):
        """
        :param checkpointing_self_attn:
        :param checkpointing_cross_attn:
        :param checkpointing_ffn:
        :param batch:
        :param zero_encoder:
        :param factorize:
        :param target_mask:
        :param mirror:
        :param kwargs:
        :param ctc_coeff:
        :param grad_scaler:
        :param ctc_labels:
        :param ctc_loss_function:
        :return:
        """
        if self.switchout > 0 and self.training:
            batch.switchout(self.switchout, self.src_vocab_size, self.tgt_vocab_size)

        src = batch.get('source')
        tgt = batch.get('target_input')

        src = src.transpose(0, 1)  # transpose to have batch first
        tgt = tgt.transpose(0, 1)

        #
        batch_first_output = False
        # step 1: extract the actual data and then forward encoder

        # during training mixture is always None
        # encoder_output = self.encoder(src,
        #                               batch_first_output=batch_first_output,
        #                               output_attentions=False)
        encoder_output = self.encoder(src)

        context = encoder_output[0]
        src_attention_mask = None
        tgt_attention_mask = None

        # tgt_attention_mask = torch.logical_not(batch.get('target_input_selfattn_mask'))
        tgt_attention_mask = batch.get('target_input_selfattn_mask')

        decoder_outputs = self.decoder( input_ids=tgt,
                                        attention_mask=tgt_attention_mask,
                                        encoder_hidden_states=context)

        # B x T x H -> T x B x H
        # decoder_output = decoder_outputs[0].transpose(0, 1).contiguous()
        # contrastive_loss = decoder_outputs[-1]

        output_dict = defaultdict(lambda: None)

        output_dict['hidden'] = decoder_output
        output_dict['context'] = context
        output_dict['src'] = src

        # final layer: computing softmax
        logprobs = self.generator[0](output_dict)['logits']
        output_dict['logprobs'] = logprobs

        return output_dict

    def create_decoder_state(self, batch, beam_size=1, type=2, buffering=True,
                             pretrained_layer_states=None, **kwargs):
        """
        Generate a new decoder state based on the batch input
        :param pretrained_layer_states:
        :param buffering:
        :param type:
        :param batch: Batch object (may not contain target during decoding)
        :param beam_size: Size of beam used in beam search
        :return:
        """
        src = batch.get('source')
        tgt_lang = batch.get('target_lang')
        src_lang = batch.get('source_lang')

        src_transposed = src.transpose(0, 1)  # transpose -> batch first

        batch_first_output = False
        encoder_output = self.encoder(src_transposed)

        context = encoder_output[0]
        src_mask = src_transposed  # B x T x H but we don't really need it

        print("[INFO] create Transformer decoding state with buffering", buffering)
        decoder_state = TransformerDecodingState(src, tgt_lang, context, src_lang,
                                                 beam_size=beam_size, model_size=self.model_size,
                                                 type=2, buffering=buffering, src_mask=src_mask)

        return decoder_state

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

        # print(input_t[0].tolist())

        output_dict = self.decoder.step(input_t, decoder_state, streaming=streaming)
        output_dict['src'] = decoder_state.src.transpose(0, 1)

        log_prob = self.generator[0](output_dict)['logits'].squeeze(0)
        log_prob = torch.nn.functional.log_softmax(log_prob, dim=-1, dtype=torch.float32)

        coverage = output_dict['coverage']
        last_coverage = coverage[:, -1, :].squeeze(1)

        output_dict['log_prob'] = log_prob
        output_dict['coverage'] = last_coverage

        return output_dict



    #TODO: create decoding state