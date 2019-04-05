import numpy as np
import torch, math
import torch.nn as nn
from onmt.modules.Transformer.Layers import EncoderLayer, DecoderLayer, PositionalEncoding, variational_dropout, PrePostProcessing
from onmt.modules.BaseModel import NMTModel, Reconstructor, DecoderState
import onmt
from onmt.modules.WordDrop import embedded_dropout
#~ from onmt.modules.Checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint
from torch.autograd import Variable



def custom_layer(module):
    def custom_forward(*args):
        output = module(*args)
        return output
    return custom_forward


class TransformerLM(NMTModel):
    """Main model in 'Attention is all you need' """

    def forward(self, input, grow=False):
        """
        Inputs Shapes:
            src: len_src x batch_size
            tgt: len_tgt x batch_size

        Outputs Shapes:
            out:      batch_size*len_tgt x model_size


        """
        # src = input[0]
        tgt = input[1][:-1]  # exclude last target from inputs

        src = src.transpose(0, 1)  # transpose to have batch first
        tgt = tgt.transpose(0, 1)

        context, src_mask = self.encoder(src, grow=grow)

        output, coverage = self.decoder(tgt, context, src, grow=grow)

        return output, context, src_mask

