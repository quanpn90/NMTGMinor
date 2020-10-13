import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.models.transformers import TransformerEncoder, TransformerDecoder, Transformer, TransformerDecodingState
from onmt.modules.sinusoidal_positional_encoding import SinusoidalPositionalEmbedding
import onmt
from onmt.modules.base_seq2seq import NMTModel, Reconstructor, DecoderState
from onmt.modules.convolution import Conv2dSubsampling
from onmt.models.transformer_layers import PrePostProcessing
from .relative_transformer_layers import RelativeTransformerEncoderLayer, RelativeTransformerDecoderLayer
from .conformer_layers import ConformerEncoderLayer
from onmt.utils import flip, expected_length
from collections import defaultdict
import math
import sys


class ConformerEncoder(TransformerEncoder):

    def __init__(self, opt, dicts, positional_encoder, encoder_type='text', language_embeddings=None):
        self.death_rate = opt.death_rate
        self.learnable_position_encoding = opt.learnable_position_encoding
        self.layer_modules = list()
        self.unidirectional = opt.unidirectional
        self.reversible = opt.src_reversible
        self.n_heads = opt.n_heads

        # build_modules will be called from the inherited constructor
        super().__init__(opt, dicts, positional_encoder, encoder_type, language_embeddings)

        # position encoding sin/cos
        self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)

        # self.audio_trans = Conv2dSubsampling(opt.input_size, opt.model_size)
        channels = self.channels
        feature_size = opt.input_size
        cnn = [nn.Conv2d(channels, 32, kernel_size=(3, 3), stride=2), nn.ReLU(True), nn.BatchNorm2d(32),
               nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2), nn.ReLU(True), nn.BatchNorm2d(32)]
        # cnn = [nn.Conv2d(channels, 32, kernel_size=(3, 3), stride=2), nn.ReLU(True),
        #        nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2), nn.ReLU(True)]

        nn.init.kaiming_normal_(cnn[0].weight, nonlinearity="relu")
        nn.init.kaiming_normal_(cnn[3].weight, nonlinearity="relu")

        feat_size = (((feature_size // channels) - 3) // 4) * 32
        # cnn.append()
        self.audio_trans = nn.Sequential(*cnn)
        self.linear_trans = nn.Linear(feat_size, self.model_size)

        self.d_head = self.model_size // self.n_heads

    def build_modules(self):

        e_length = expected_length(self.layers, self.death_rate)
        print("* Conformer Encoder with %.2f expected layers" % e_length)
        if self.unidirectional:
            print("* Running a unidirectional Encoder.")

        self.layer_modules = nn.ModuleList()

        for _l in range(self.layers):
            # linearly decay the death rate
            death_r = (_l + 1.0) / self.layers * self.death_rate

            block = ConformerEncoderLayer(self.opt, death_rate=death_r)
            self.layer_modules.append(block)

    def forward(self, input, input_pos=None, input_lang=None, streaming=False, **kwargs):
        """
        :param input: [B x T x Input_Size]
        :param input_pos: [B x T] positions
        :param input_lang: [B] language ids of each sample
        :param streaming: connect different segments in transformer-xl style
        :param kwargs:
        :return:
        """

        long_mask = input.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD)
        input = input.narrow(2, 1, input.size(2) - 1)

        # first subsampling
        input = input.view(input.size(0), input.size(1), -1, self.channels)
        input = input.permute(0, 3, 1, 2)  # [bsz, channels, time, f]
        input = self.audio_trans(input)
        input = input.permute(0, 2, 1, 3).contiguous()
        input = input.view(input.size(0), input.size(1), -1)
        input = self.linear_trans(input)
        emb = input

        mask_src = long_mask[:, 0:emb.size(1) * 4:4].transpose(0, 1).unsqueeze(0)
        dec_attn_mask = None

        emb = emb.transpose(0, 1)
        input = input.transpose(0, 1)
        mem_len = 0
        mems = None

        if self.unidirectional:
            qlen = input.size(0)
            klen = qlen + mem_len
            attn_mask_src = torch.triu(
                emb.new_ones(qlen, klen), diagonal=1 + mem_len).byte()[:, :, None]

            pad_mask = mask_src

            mask_src = pad_mask + attn_mask_src
            # dec_attn_mask = dec_attn_mask + pad_mask.unsqueeze(0)
            mask_src = mask_src.gt(0)

        if onmt.constants.torch_version >= 1.2:
            mask_src = mask_src.bool()

        """ Adding positional encoding """
        qlen = input.size(0)
        klen = qlen + mem_len

        # Asynchronous positions: 2K+1 positions instead of K+1
        if self.unidirectional:
            pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)
        else:
            pos = torch.arange(klen - 1, -klen, -1.0, device=emb.device, dtype=emb.dtype)

        # pos_emb has size 2T+1 x 1 x H
        pos_emb = self.positional_encoder(pos, bsz=input.size(1))

        if self.learnable_position_encoding:
            raise NotImplementedError

        context = emb

        # Apply dropout to both context and pos_emb
        context = self.preprocess_layer(context)

        pos_emb = self.preprocess_layer(pos_emb)

        for i, layer in enumerate(self.layer_modules):
            # src_len x batch_size x d_model

            context = layer(context, pos_emb, mask_src)

        # final layer norm
        context = self.postprocess_layer(context)

        output_dict = defaultdict(lambda: None, {'context': context, 'src_mask': dec_attn_mask, 'src': input})

        return output_dict

