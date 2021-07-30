import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class TransformerClassifier(nn.Module):
    """Main model in 'Attention is all you need' """

    def __init__(self, encoder, generator=None, mpc=False, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        self.num_classes = self.generator[0].linear.weight.size(0)
        self.mpc = mpc

        if mpc:
            input_size = self.encoder.opt.input_size
            model_size = self.encoder.opt.model_size
            self.mpc_linear = nn.Linear(model_size, input_size)

        if self.encoder.input_type == 'text':
            self.src_vocab_size = self.encoder.word_lut.weight.size(0)
        else:
            self.src_vocab_size = 0

    def forward(self, batch, *args, **kwargs):

        if self.mpc and self.training:
            # mask inputs with p=20%
            batch.mask_mpc(p=0.2)

        src = batch.get('source')

        src_pos = batch.get('source_pos')
        src_lang = batch.get('source_lang')

        src_lengths = batch.src_lengths

        src = src.transpose(0, 1)  # transpose to have batch first
        encoder_output = self.encoder(src, input_pos=src_pos, input_lang=src_lang, src_lengths=src_lengths)

        # feed the encoder output to generator? Or average per frame?

        encoder_output = defaultdict(lambda: None, encoder_output)
        context = encoder_output['context']

        # build the output dict based on decoder output
        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = context
        output_dict['context'] = context
        output_dict['src_mask'] = encoder_output['src_mask']
        output_dict['src'] = src
        output_dict['target_mask'] = encoder_output['src_mask']

        logprobs = self.generator[0](output_dict)['logits']
        output_dict['logprobs'] = logprobs

        # masked predictive coding
        if self.mpc:
            # mpc reconstruction
            mpc_rec = self.mpc_linear(context)
            output_dict['mpc'] = mpc_rec
            output_dict['masked_positions'] = batch.get('masked_positions')
            output_dict['original_source'] = batch.get('original_source')

        return output_dict




