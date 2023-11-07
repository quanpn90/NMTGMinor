import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from onmt.models.transformers import Transformer, TransformerDecodingState, TransformerDecodingStateMemory
from typing import List, Optional, Union
from collections import defaultdict
import onmt
from onmt.modules.optimized.linear import Linear
import math
from .fairseq_wav2vec2.file_io import PathManager
from omegaconf import DictConfig, open_dict, OmegaConf
from .fairseq_wav2vec2.utils import overwrite_args_by_name

import copy
import numpy as np
from onmt.modules.loss import CrossEntropyLossBase
from onmt.modules.layer_norm import LayerNorm


def average_features(features, pad_mask):
    """
    This function computes the mean of features across the time dimension

    Args:
        features: torch.Tensor [T x B x H]
        pad_mask: [B x T]

    Returns:

    """

    features_t = features.transpose(0, 1).contiguous()

    features_m = features_t.masked_fill_(pad_mask.unsqueeze(2).expand_as(features_t), 0)

    lengths = (1 - pad_mask.long()).sum(dim=-1)

    mean = features_m.sum(dim=1).div(lengths.unsqueeze(1))

    return mean


class Wav2VecCLIP(nn.Module):

    def __init__(self, acoustic_encoder, text_encoder, **kwargs):
        """
        Args:
            acoustic_encoder: Wav2vec2 Encoder
            text_encoder:
            **kwargs:
        """

        super(Wav2VecCLIP, self).__init__()

        self.acoustic_encoder = acoustic_encoder
        self.text_encoder = text_encoder

        # TODO: freezing the weights in the acoustic encoder

    def forward(self, batch):
        src = batch.get('source')
        tgt = batch.get('target')

        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')
        src_atb = batch.get('source_atbs')
        tgt_atb = batch.get('target_atbs')

        # transpose to have batch first
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        # first we run the wav2vec encoder
        batch_first_output = False
        # TODO: run the acoustic encoder in no_grad mode except for the extra layers
        encoder_output = self.acoustic_encoder(src, batch_first_output=batch_first_output,
                                               lang=src_lang, atb=src_atb, extra_layers_only=True)

        acoustic_features = encoder_output['context']
        acoustic_pad_mask = encoder_output['src']

        # averaged features from the acoustic encoder [B x H]
        acoustic_features = average_features(acoustic_features, acoustic_pad_mask)

        text_pad_mask = batch.get("tgt_selfattn_mask")
        # freeze the mbart encoder so:
        with torch.no_grad():
            text_output = self.text_encoder(tgt, attention_mask=text_pad_mask)

        text_features = text_output[0]

        # averaged features from the MBART encoder [B x H]
        text_features = average_features(text_features, text_pad_mask)

        output_dict = defaultdict(lambda: None)
        output_dict['acoustic_features'] = acoustic_features
        output_dict['text_features'] = text_features

        return output_dict
