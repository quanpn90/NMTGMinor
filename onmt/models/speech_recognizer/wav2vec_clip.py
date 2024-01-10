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
from itertools import groupby


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

        self.model_size = acoustic_encoder.model_size

        # this model cannot use CTC
        self.ctc = False
        self.ctc_char = False
        self.ctc_compress = False

        # TODO: freezing the weights in the acoustic

    def create_ctc_char(self, char_data, ctc_compress="None"):

        id2char = char_data['id2char']
        char_vocab_size = len(id2char)
        self.char_vocab_size = char_vocab_size
        self.char_ctc_linear = nn.Linear(self.model_size, char_vocab_size)
        print(self.char_vocab_size)

        if ctc_compress != "None":
            from .ctc_compressor import CTCCompressStrategy
            self.ctc_compress = getattr(CTCCompressStrategy, ctc_compress)
        else:
            self.ctc_compress = None

        self.ctc_char = True

    def forward(self, batch, ctc_loss_function=None,
                ctc_coeff=None, **kwargs):
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

        if self.ctc_char:
            # what is the ctc_labels here?
            ctc_labels = batch.get("char_target").transpose(0, 1)
            assert (ctc_loss_function.padding_idx == onmt.constants.TGT_PAD)

            # compute the logits for each encoder step
            # run the ctcoutput via the wav2vec context (not context)
            # ctc output should have the mbart vocabulary
            # encoder_hidden = output_dict['wav2vec_context'].
            # output_dict['encoder_logits'] = self.ctc_linear(output_dict['wav2vec_context'])
            # how should we proceed from this?

            encoder_logits = self.char_ctc_linear(acoustic_features)

            ctc_loss_inputs = dict()
            ctc_loss_inputs['encoder_logits'] = encoder_logits
            ctc_loss_inputs['wav2vec_padding_mask'] = encoder_output['wav2vec_padding_mask']
            ctc_loss_inputs['src_mask'] = acoustic_pad_mask

            ctc_loss, n_ctc_targets = ctc_loss_function(ctc_loss_inputs, ctc_labels)
            ctc_loss_data = ctc_loss.item()
            ctc_loss = ctc_loss * ctc_coeff

            if self.ctc_compress is not None:
                src_lengths = (1 - acoustic_pad_mask.long()).sum(dim=1)

                # TODO: Ctc compression
                with torch.no_grad():
                    x_ctc = encoder_logits
                    batch_predicted = []
                    prob_ctc = F.softmax(x_ctc, dim=-1).transpose(0, 1)  # from T x B x D to B x T x D
                    for b in range(prob_ctc.shape[0]):
                        predicted = prob_ctc[b][: src_lengths[b]].argmax(-1).tolist()
                        batch_predicted.append([(p[0], len(list(p[1]))) for p in groupby(predicted)])

                    new_lengths = [len(p) for p in batch_predicted]

                    # TODO: compress_method
                    weights_matrix = self.ctc_compress(prob_ctc, batch_predicted, new_lengths, x_ctc.dtype,
                                                       x_ctc.device)

                context = acoustic_features.permute(1, 2, 0).bmm(weights_matrix).permute(2, 0, 1)

                # creating a new padding mask
                max_len = max(new_lengths)
                _src_mask = context.new_zeros(len(new_lengths), max_len).bool()
                for i, l in enumerate(new_lengths):
                    _src_mask[i, l:] = 1

                acoustic_features = context
                acoustic_pad_mask = _src_mask
                del encoder_logits

        else:
            ctc_loss = None
            ctc_loss_data = None
            n_ctc_targets = 0

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

        output_dict['ctc_loss'] = ctc_loss
        output_dict['ctc_loss_data'] = ctc_loss_data
        output_dict['n_ctc_targets'] = n_ctc_targets

        return output_dict
