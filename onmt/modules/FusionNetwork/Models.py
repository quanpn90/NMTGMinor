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
        super(FusionNetwork, self).__init__()
        self.tm_model = tm_model
        self.lm_model = lm_model

        # freezing the parameters for the language model
        for param in self.lm_model.parameters():
            param.requires_grad = False


    def forward(self, batch):
        """
        Inputs Shapes:
            src: len_src x batch_size
            tgt: len_tgt x batch_size

        Outputs Shapes:
            out:      batch_size*len_tgt x model_size


        """

        nmt_output_dict = self.tm_model(batch)

        # no gradient for the LM side
        with torch.no_grad():
            lm_output_dict = self.lm_model(batch)


        output_dict = defaultdict(lambda: None)

        output_dict['tm'] = nmt_output_dict
        output_dict['lm']  = lm_output_dict

        return output_dict



class FusionDecodingState(DecoderState):

    def __init__(self, tm_state, lm_state):

        self.tm_state = tm_state
        self.lm_state = lm_state

