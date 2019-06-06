import numpy as np
import torch, math
import torch.nn as nn
import onmt
from collections import defaultdict
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder, TransformerDecodingState
from onmt.modules.Transformer.Layers import EncoderLayer, DecoderLayer
from onmt.modules.BaseModel import NMTModel
from onmt.modules.WordDrop import embedded_dropout
from torch.utils.checkpoint import checkpoint
from onmt.modules.Utilities import mean_with_mask_backpropable as mean_with_mask
from onmt.modules.Utilities import max_with_mask
from onmt.modules.Transformer.Layers import PrePostProcessing
import copy
from onmt.modules.Transformer.Models import Transformer

# to do: create an external decoder and synchronize it every parameter update

class ParallelTransformer(Transformer):
    """Main model in 'Attention is all you need' """

    def __init__(self, encoder, decoder, generator=None, tgt_encoder=None, tgt_decoder=None):
        super().__init__(encoder, decoder, generator=generator)
        self.tgt_encoder = tgt_encoder
        self.tgt_decoder = tgt_decoder

        # freeze the parameters for non-embedding (which is shared anyways)
        for child in tgt_decoder.children():
            if not (isinstance(child, torch.nn.modules.sparse.Embedding)):
                for param in child.parameters():
                    param.requires_grad = False


    def _synchronize(self):

        # every time model parameters are updated, we synchronize the paramters of the target decoder
        # with the regular decoder
        self.tgt_decoder.load_state_dict(self.decoder.state_dict())
        return


    def forward(self, batch, fast_mode=False, **kwargs):
        """
        Inputs Shapes:
            src: len_src x batch_size
            tgt: len_tgt x batch_size

        Outputs Shapes:
            out:   a dictionary containing output hidden state and coverage

        """
        src = batch.get('source')
        tgt = batch.get('target_input')
        tgt_attbs = batch.get('tgt_attbs')  # vector of length B

        src = src.transpose(0, 1)  # transpose to have batch first
        tgt = tgt.transpose(0, 1)

        context, src_mask = self.encoder(src)

        # the first token is <BOS>
        # the encoder side is not patched with <BOS>
        tgt_ = tgt[:, 1:]

        # because the context size does not depend on the input size
        # and the context does not have masking any more
        # so we create a 'fake' input sequence for the decoder
        dec_output = self.decoder(tgt, tgt_attbs, context, src)

            if not fast_mode:
            if self.tgt_encoder is not None:
                # don't look at the first token of the target
                tgt_context, _ = self.tgt_encoder(tgt_)

                # generate the target distribution on top of the tgt hiddens
                tgt_dec_output = self.tgt_decoder(tgt, tgt_attbs, tgt_context, tgt_, freeze_embedding=True)
            else:
                tgt_dec_output = defaultdict(lambda: None)

        else:
            tgt_dec_output = defaultdict(lambda: None)
            tgt_dec_output['final_state'] = \
                dec_output['final_state'].new(*dec_output['final_state'].size()).copy_( dec_output['final_state'])

        output_dict = dict()
        output_dict['hiddens'] = dec_output['final_state']
        output_dict['coverage'] = dec_output['coverage']
        output_dict['tgt_hiddens'] = tgt_dec_output['final_state']

        return output_dict