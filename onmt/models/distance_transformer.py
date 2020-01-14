import torch
import torch.nn as nn
from onmt.models.transformer_layers import PositionalEncoding, PrePostProcessing
from onmt.models.transformer_layers import EncoderLayer, DecoderLayer
from onmt.models.transformers import TransformerEncoder, TransformerDecoder
import onmt
from onmt.modules.dropout import embedded_dropout
from onmt.models.transformer_layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing
from onmt.models.relative_transformer_layers import RelativeTransformerEncoderLayer, RelativeTransformerDecoderLayer
from onmt.utils import flip, expected_length
from collections import defaultdict
import math


class DistanceTransformerEncoder(TransformerEncoder):
    """
    Self-attention with learnable past and future relative positions (with embeddings)
    """

    def __init__(self, opt, dicts, positional_encoder, encoder_type='text'):
        self.death_rate = opt.death_rate
        self.double_position = opt.double_position
        self.max_pos_length = opt.max_pos_length
        self.layer_modules = list()

        # build_modules will be called from the inherited constructor
        super(RelativeTransformerEncoder, self).__init__(opt, dicts, positional_encoder, encoder_type)

        print("Encoder type: %s", encoder_type)
        # self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)

        # embedding for the positions
        # 2N - 1 positions because it runs from -N -> 0 -> N
        self.positional_encoder = nn.Embedding(2 * self.max_pos_length + 1, self.model_size)
        self.d_head = self.model_size // self.n_heads

        e_length = expected_length(self.layers, self.death_rate)

        print("* Transformer Encoder with Relative Attention with %.2f expected layers" % e_length)

    def build_modules(self):
        self.layer_modules = nn.ModuleList()

        for l in range(self.layers):
            # linearly decay the death rate
            death_r = (l + 1.0) / self.layers * self.death_rate

            block = RelativeTransformerEncoderLayer(self.n_heads, self.model_size,
                                                    self.dropout, self.inner_size, self.attn_dropout,
                                                    variational=self.varitional_dropout, death_rate=death_r)

            self.layer_modules.append(block)

    def forward(self, input, input_pos=None, **kwargs):
        """
        Inputs Shapes:
            input: batch_size x src_len (wanna tranpose)
        Outputs Shapes:
            out: batch_size x src_len x d_model
            mask_src
        """

        """ Embedding: batch_size x src_len x d_model """
        if self.input_type == "text":
            bsz_first_input = input
            input = input.transpose(0, 1)
            # mask_src = input.eq(onmt.constants.PAD).unsqueeze(1)  # batch_size x src_len x 1 for broadcasting
            mask_src = input.eq(onmt.constants.PAD).unsqueeze(0)
            dec_attn_mask = bsz_first_input.eq(onmt.constants.PAD).unsqueeze(1)

            emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)

            # if self.double_position:
            #     assert input_pos is not None
            #     # flatten
            #     src_len, bsz = input_pos.size(0), input_pos.size(1)
            #     input_pos_ = input_pos.contiguous().view(-1).type_as(emb)
            #     abs_pos = self.positional_encoder(input_pos_)
            #     abs_pos = abs_pos.squeeze(1).view(src_len, bsz, -1)
            #
            # else:
            #     abs_pos = None
        else:
            if not self.cnn_downsampling:
                mask_src = input.narrow(2, 0, 1).squeeze(2).transpose(0, 1).eq(onmt.constants.PAD).unsqueeze(0)
                dec_attn_mask = input.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD).unsqueeze(1)
                input = input.narrow(2, 1, input.size(2) - 1)
                emb = self.audio_trans(input.contiguous().view(-1, input.size(2))).view(input.size(0),
                                                                                        input.size(1), -1)
            else:
                long_mask = input.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD)
                input = input.narrow(2, 1, input.size(2) - 1)

                # first resizing to fit the CNN format
                input = input.view(input.size(0), input.size(1), -1, self.channels)
                input = input.permute(0, 3, 1, 2)

                input = self.audio_trans(input)
                input = input.permute(0, 2, 1, 3).contiguous()
                input = input.view(input.size(0), input.size(1), -1)
                # print(input.size())
                input = self.linear_trans(input)

                mask_src = long_mask[:, 0:input.size(1) * 4:4].transpose().unsqueeze(0)
                dec_attn_mask = long_mask[:, 0:input.size(1) * 4:4].unsqueeze(1)
                # the size seems to be B x T ?
                emb = input

            emb = emb.transpose(0, 1)
            input = input.transpose(0, 1)
            abs_pos = None

        if onmt.constants.torch_version >= 1.2:
            mask_src = mask_src.bool()

        # Scale the emb by sqrt(d_model)
        emb = emb * math.sqrt(self.model_size)

        # if self.double_position and abs_pos is not None:
        #     # adding position encoding
        #     emb = emb + abs_pos
        klen = input.size(0)

        # allocate positions: from L - 1 to -L + 1
        pos = torch.arange(klen - 1, -klen + 1, -1.0, device=emb.device)

        # clamp the positions (all postions from afar are treated equally, maybe?)
        pos = torch.clamp(pos, -self.max_pos_length, self.max_pos_length)

        # L x 1 x H
        pos_emb = self.positional_encoder(pos.unsqueeze(1))

        # Apply dropout to both context and pos_emb
        context = self.preprocess_layer(emb)

        pos_emb = self.preprocess_layer(pos_emb)

        for i, layer in enumerate(self.layer_modules):
            # src_len x batch_size x d_model
            context = layer(context, pos_emb, mask_src)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        context = self.postprocess_layer(context)

        output_dict = {'context': context, 'src_mask': dec_attn_mask, 'src': input}

        # return context, mask_src
        return output_dict

