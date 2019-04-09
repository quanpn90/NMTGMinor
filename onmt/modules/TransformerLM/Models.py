import numpy as np
import torch, math
import torch.nn as nn
from onmt.modules.Transformer.Layers import EncoderLayer, DecoderLayer, PositionalEncoding, variational_dropout, PrePostProcessing
from onmt.modules.BaseModel import NMTModel, Reconstructor, DecoderState
import onmt
from onmt.modules.WordDrop import embedded_dropout
#~ from onmt.modules.Checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint
from collections import defaultdict


def custom_layer(module):
    def custom_forward(*args):
        output = module(*args)
        return output
    return custom_forward


class TransformerLM(NMTModel):
    """Main model in 'Attention is all you need' """

    def forward(self, batch):
        """
        Inputs Shapes:
            src: len_src x batch_size
            tgt: len_tgt x batch_size

        Outputs Shapes:
            out:      batch_size*len_tgt x model_size


        """
        # we only need target for language model
        tgt = batch.get('target_input')
        tgt = tgt.transpose(0, 1)

        context = None
        src = None

        output, _ = self.decoder(tgt, context, src)

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = output

        return output_dict

