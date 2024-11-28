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
        tgt_pos = batch.get('target_pos')
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')
        src_atb = batch.get('source_atbs')
        tgt_atb = batch.get('target_atbs')
        src_lengths = batch.src_lengths
        tgt_lengths = batch.tgt_lengths

        org_src = src
        org_tgt = tgt
        src = src.transpose(0, 1)  # transpose to have batch first
        tgt = tgt.transpose(0, 1)

        #
        batch_first_output = False
        # step 1: extract the actual data and then forward encoder

        # during training mixture is always None
        encoder_output = self.encoder(src, batch_first_output=batch_first_output)

        context = encoder_output[0]
        src_attention_mask = None
        tgt_attention_mask = None

        # decoder_outputs = self.decoder(input_ids=tgt,
        #                                attention_mask=tgt_attention_mask,
        #                                encoder_hidden_states=context,
        #                                encoder_attention_mask=src_attention_mask,
        #                                sub_encoder_hidden_states=None,
        #                                sub_encoder_attention_mask=None,
        #                                lang=tgt_lang, atb=tgt_atb,
        #                                src_lang=_src_lang,
        #                                checkpointing_ffn=checkpointing_ffn,
        #                                checkpointing_cross_attn=checkpointing_cross_attn,
        #                                checkpointing_self_attn=checkpointing_self_attn)

        decoder_outputs = self.decoder( input_ids=tgt,
                                        encoder_hidden_states=context)

        decoder_output = decoder_outputs[0]
        # contrastive_loss = decoder_outputs[-1]

        output_dict = defaultdict(lambda: None)

        output_dict['hidden'] = decoder_output
        output_dict['context'] = context
        output_dict['src'] = src

        # final layer: computing softmax
        logprobs = self.generator[0](output_dict)['logits']
        output_dict['logprobs'] = logprobs

        return output_dict

    #TODO: create decoding state