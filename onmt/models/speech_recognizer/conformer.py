import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.models.transformers import TransformerEncoder, TransformerDecoder, Transformer, TransformerDecodingState
from onmt.modules.sinusoidal_positional_encoding import SinusoidalPositionalEmbedding
import onmt
from onmt.modules.base_seq2seq import NMTModel, Reconstructor, DecoderState
from onmt.modules.dropout import embedded_dropout
from onmt.models.transformer_layers import PrePostProcessing
from .relative_transformer_layers import RelativeTransformerEncoderLayer, RelativeTransformerDecoderLayer
from onmt.utils import flip, expected_length
from collections import defaultdict
import math
import sys


class ConformerEncoder(TransformerEncoder):

    def __init__(self, opt, dicts, positional_encoder, encoder_type='text', language_embeddings=None):
        self.death_rate = opt.death_rate
        self.learnable_position_encoding = opt.learnable_position_encoding
        self.layer_modules = list()
        self.asynchronous = opt.asynchronous
        self.max_memory_size = opt.max_memory_size
        self.extra_context_size = opt.extra_context_size
        self.experimental = opt.experimental
        self.unidirectional = opt.unidirectional
        self.reversible = opt.src_reversible
        self.n_heads = opt.n_heads
        self.fast_self_attn = opt.fast_self_attention

        # build_modules will be called from the inherited constructor
        super().__init__(opt, dicts, positional_encoder, encoder_type, language_embeddings)

        # learnable position encoding
        if self.learnable_position_encoding:
            raise NotImplementedError
        else:
            # or using pre-set sinusoidal
            self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)

        self.d_head = self.model_size // self.n_heads

    def build_modules(self):

        e_length = expected_length(self.layers, self.death_rate)
        print("* Transformer Encoder with Relative Attention with %.2f expected layers" % e_length)
        if self.unidirectional:
            print("* Running a unidirectional Encoder.")

        self.layer_modules = nn.ModuleList()

        for _l in range(self.layers):
            # linearly decay the death rate
            death_r = (_l + 1.0) / self.layers * self.death_rate

            block = RelativeTransformerEncoderLayer(self.opt, death_rate=death_r)
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

        if not self.cnn_downsampling:
            mask_src = input.narrow(2, 0, 1).squeeze(2).transpose(0, 1).eq(onmt.constants.PAD).unsqueeze(0)
            dec_attn_mask = input.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD).unsqueeze(1)
            input = input.narrow(2, 1, input.size(2) - 1)
            emb = self.audio_trans(input.contiguous().view(-1, input.size(2))).view(input.size(0),
                                                                                    input.size(1), -1)
            emb = emb.type_as(input)
        else:
            long_mask = input.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD)
            input = input.narrow(2, 1, input.size(2) - 1)

            # first resizing to fit the CNN format
            # note that this is actually conv2d so channel=1, f=40
            input = input.view(input.size(0), input.size(1), -1, self.channels)
            input = input.permute(0, 3, 1, 2)  # [bsz, channels, time, f]

            # apply CNN
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

        """ Scale the emb by sqrt(d_model) """
        emb = emb * math.sqrt(self.model_size)

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

        # B x T x H -> T x B x H
        context = emb

        if streaming:
            hids = [context]

        # Apply dropout to both context and pos_emb
        context = self.preprocess_layer(context)

        pos_emb = self.preprocess_layer(pos_emb)

        if self.reversible:
            context = torch.cat([context, context], dim=-1)

            assert streaming is not True, "Streaming and Reversible is not usable yet."
            # print(context.size(), pos_emb.size())
            context = ReversibleEncoderFunction.apply(context, pos_emb, self.layer_modules, mask_src)
        else:
            for i, layer in enumerate(self.layer_modules):
                # src_len x batch_size x d_model

                mems_i = mems[i] if mems is not None and streaming and self.max_memory_size > 0 else None
                context = layer(context, pos_emb, mask_src, mems=mems_i)

                if streaming:
                    hids.append(context)

        # final layer norm
        context = self.postprocess_layer(context)

        output_dict = defaultdict(lambda: None, {'context': context, 'src_mask': dec_attn_mask, 'src': input})

        if streaming:
            # streaming_state.prev_src_mem_size += sum(input_length.tolist())
            # streaming_state.prune_source_memory(self.max_memory_size)
            streaming_state.update_src_mems(hids, qlen)
            output_dict['streaming_state'] = streaming_state

        return output_dict