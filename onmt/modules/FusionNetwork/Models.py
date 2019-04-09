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
from onmt.modules.BaseModel import DecoderState
from collections import defaultdict



class FusionNetwork(nn.Module):
    """Main model in 'Attention is all you need' """

    def __init__(self, tm_model, lm_model):
        self.tm_model = tm_model
        self.lm_model = lm_model


    def forward(self, batch):
        """
        Inputs Shapes:
            src: len_src x batch_size
            tgt: len_tgt x batch_size

        Outputs Shapes:
            out:      batch_size*len_tgt x model_size


        """
        # we only need target for language model
        # tgt = batch.get('target_input')
        # tgt = tgt.transpose(0, 1)

        nmt_output_dict = self.tm_model(batch)

        lm_output_dict = self.lm_model(batch)


        output_dict = defaultdict(lambda: None)

        output_dict['nmt'] = nmt_output_dict
        output_dict['lm']  = lm_output_dict
        # context = None
        # src = None
        #
        # output, _ = self.decoder(tgt, context, src)
        #
        # output_dict = defaultdict(lambda: None)
        # output_dict['hidden'] = output
        #
        # return output, _, _




class FusionDecodingState(DecoderState):

    def __init__(self, tm_state, lm_state):

        self.tm_state = tm_state
        self.lm_state = lm_state

