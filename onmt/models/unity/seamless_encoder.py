import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
from collections import defaultdict
import onmt
import math

import copy
import numpy as np


class SeamlessEncoder(nn.Module):

    def __init__(self, opt, model_path="seamless_encoder.pt",
                 **kwargs):

        super().__init__()
        # from onmt.models.speech_recognizer.w2v_bert.w2vbert_config import conformer_shaw_600m
        # from onmt.models.speech_recognizer.w2v_bert.w2vbert_builder import create_conformer_shaw_model
        # config = conformer_shaw_600m()
        #
        # self.wav2vec_encoder = create_conformer_shaw_model(config)
        from onmt.models.unity.unity_builder import create_unity_encoder
        from onmt.models.unity.unity_builder import _base_v2

        config = _base_v2()
        self.wav2vec_encoder = create_unity_encoder(config)

        if len(model_path) > 0:

            # todo: try catch loading and load_state_dicts
            cpt = torch.load(model_path, map_location=torch.device('cpu'))
            weights = cpt
            print("[INFO] Loaded pretrained seamless encoder w/ adaptor model")
            self.wav2vec_encoder.load_state_dict(weights)

        self.opt = opt
        self.input_type = self.opt.encoder_type
        self.model_size = self.wav2vec_encoder.model_dim
        self.time = None
        self.quantize = opt.wav2vec2_quantize
        self.dual_output = opt.wav2vec2_dual_output and self.quantize

    def convert_fast_attention(self):
        pass

    def freeze_ffn_params(self):
        pass

    def forward(self, input, batch_first_output=False,
                lang=None, atb=None,
                **kwargs):
        """
        :param atb:
        :param lang:
        :param batch_first_output: [bsz, seq_len, hidden_size] as output size, else transpose(0, 1)
        :param input: torch.Tensor [batch_size, sequence_length, 2]
        :param kwargs:
        :param only_extra_layers: no_grad until the extra layers
        :return:
        """

        input = input.contiguous()
        # The data has been constructed that the first dimension is padding mask
        # 0 for tokens that are not masked, 1 for tokens that are masked
        with torch.no_grad():
            long_mask = input.narrow(2, 0, 1).squeeze(2).eq(0).long()
            input = input.narrow(2, 1, input.size(2) - 1)

        attn_mask = long_mask

        input = input.contiguous()

        wav2vec_output, padding_mask = self.wav2vec_encoder(input, attn_mask.byte())

        # output size is always B x T x C (with the current implementation)
        continuous_output = wav2vec_output
        time, batch_size = continuous_output.size(1), continuous_output.size(0)

        # mask size is B x T (1 for padded positions, 0 for unpadded)
        dec_attn_mask = padding_mask
        context = continuous_output

        if dec_attn_mask is None:
            dec_attn_mask = context.new_zeros(batch_size, time).byte()
        else:
            dec_attn_mask = dec_attn_mask.byte()

        # wav2vec_context = wav2vec_context.transpose(0, 1).contiguous()
        context = context.transpose(0, 1).contiguous()
        wav2vec_context = context
        wav2vec_padding_mask = dec_attn_mask

        output_dict = defaultdict(lambda: None, {'source': input, 'context': context, 'src_mask': dec_attn_mask,
                                                 'src': dec_attn_mask, 'pos_emb': None,
                                                 'wav2vec_context': wav2vec_context,
                                                 'wav2vec_padding_mask': wav2vec_padding_mask,
                                                 'enc_pred_lang': None})

        return output_dict