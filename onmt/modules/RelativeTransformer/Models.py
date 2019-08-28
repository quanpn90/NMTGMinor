import numpy as np
import torch, math
import torch.nn as nn
from onmt.modules.Transformer.Layers import PositionalEncoding, PrePostProcessing
from onmt.modules.Transformer.Layers import EncoderLayer, DecoderLayer
from onmt.modules.RelativeTransformer.Layers import RelativeTransformerDecoderLayer
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder
from onmt.modules.BaseModel import NMTModel, Reconstructor
import onmt
from onmt.modules.WordDrop import embedded_dropout
from onmt.modules.Checkpoint import checkpoint

from onmt.modules.Transformer.Layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing


class RelativeTransformerDecoder(TransformerDecoder):
    """Encoder in 'Attention is all you need'

    Args:
        opt
        dicts


    """

    def __init__(self, opt, dicts, positional_encoder, attribute_embeddings=None, ignore_source=False):
        self.death_rate = opt.death_rate
        self.layer_modules = None

        # build_modules will be called from the inherited constructor
        super(RelativeTransformerDecoder, self).__init__(opt, dicts,
                                                         positional_encoder,
                                                         attribute_embeddings,
                                                         ignore_source)
        self.positional_encoder = positional_encoder

    def build_modules(self):
        self.layer_modules = nn.ModuleList([RelativeTransformerDecoderLayer
                                            (self.n_heads, self.model_size,
                                             self.dropout, self.inner_size,
                                             self.attn_dropout,
                                             ignore_source=self.ignore_source) for _ in range(self.layers)])

    def forward(self, input, context, src, atbs=None, **kwargs):
        """
        Inputs Shapes:
            input: (Variable) batch_size x len_tgt (wanna tranpose)
            context: (Variable) batch_size x len_src x d_model
            mask_src (Tensor) batch_size x len_src
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src

        """

        """ Embedding: batch_size x len_tgt x d_model """

        input = input.transpose(0, 1)  # B x T to T x B
        klen, batch_size = input.size()
        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)

        # Adding dropout
        emb = self.preprocess_layer(emb)

        emb = emb * math.sqrt(self.model_size)

        # Prepare positional encoding:
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)
        # pos_seq = torch.arange(0, klen, device=emb.device, dtype=emb.dtype)
        pos_emb = self.preprocess_layer(self.positional_encoder(pos_seq))

        if self.use_feature:
            raise NotImplementedError

        if context is not None:
            if self.encoder_type == "audio":
                if not self.encoder_cnn_downsampling:
                    mask_src = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD).unsqueeze(1)
                else:
                    long_mask = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD)
                    mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
            else:

                mask_src = src.data.eq(onmt.Constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        # attention masking
        qlen = klen
        mask_tgt = torch.triu(emb.new_ones(qlen, klen), diagonal=1).unsqueeze(-1).byte()
        mask_tgt = mask_tgt + input.eq(onmt.Constants.PAD).byte().unsqueeze(0)
        mask_tgt = torch.gt(mask_tgt, 0)  # convert all 2s to 1
        mask_tgt = mask_tgt.bool()

        output = emb

        for i, layer in enumerate(self.layer_modules):
            output, coverage = layer(output, pos_emb, context, mask_tgt, mask_src)  # batch_size x len_src x d_model

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        output = self.postprocess_layer(output)

        output_dict = {'hidden': output, 'coverage': coverage}

        # return output, None
        return output_dict

    def step(self, input, decoder_state):
        """
        Inputs Shapes:
            input: (Variable) batch_size x len_tgt (wanna tranpose)
            context: (Variable) batch_size x len_src x d_model
            mask_src (Tensor) batch_size x len_src
            buffer (List of tensors) List of batch_size * len_tgt-1 * d_model for self-attention recomputing
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src

        """
        context = decoder_state.context
        buffers = decoder_state.attention_buffers
        src = decoder_state.src.transpose(0, 1) if decoder_state.src is not None else None
        atbs = decoder_state.tgt_atb

        if decoder_state.input_seq is None:
            decoder_state.input_seq = input
        else:
            # concatenate the last input to the previous input sequence
            decoder_state.input_seq = torch.cat([decoder_state.input_seq, input], 0)
        # input = decoder_state.input_seq.transpose(0, 1)

        input = decoder_state.input_seq  # no need to transpose because time first
        input_ = input[-1, :].unsqueeze(0)

        """ Embedding: batch_size x 1 x d_model """
        emb = self.word_lut(input_)

        emb = emb * math.sqrt(self.model_size)

        if isinstance(emb, tuple):
            emb = emb[0]
        # emb should be batch_size x 1 x dim

        # Prepare positional encoding:
        klen = input.size(0)
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)
        # pos_seq = torch.arange(0, klen, device=emb.device, dtype=emb.dtype)
        pos_emb = self.preprocess_layer(self.positional_encoder(pos_seq))

        if self.use_feature:
            raise NotImplementedError
            # atb_emb = self.attribute_embeddings(atbs).unsqueeze(1)  # B x H to B x 1 x H
            # emb = torch.cat([emb, atb_emb], dim=-1)
            # emb = torch.relu(self.feature_projector(emb))

        # batch_size x 1 x len_src
        if context is not None:
            if self.encoder_type == "audio":
                if src.data.dim() == 3:
                    if self.encoder_cnn_downsampling:
                        long_mask = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD)
                        mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
                    else:
                        mask_src = src.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD).unsqueeze(1)
                elif self.encoder_cnn_downsampling:
                    long_mask = src.eq(onmt.Constants.PAD)
                    mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
                else:
                    mask_src = src.eq(onmt.Constants.PAD).unsqueeze(1)
            else:
                mask_src = src.eq(onmt.Constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        # attention masking
        klen, batch_size = input.size()
        qlen = klen
        mask_tgt = torch.triu(emb.new_ones(qlen, klen), diagonal=1).unsqueeze(-1).byte()
        mask_tgt = mask_tgt + input.eq(onmt.Constants.PAD).byte().unsqueeze(0)
        mask_tgt = torch.gt(mask_tgt, 0)  # convert all 2s to 1
        mask_tgt = mask_tgt.bool()

        output = emb.contiguous()

        for i, layer in enumerate(self.layer_modules):
            buffer = buffers[i] if i in buffers else None
            assert (output.size(0) == 1)

            output, coverage, buffer = layer.step(output, pos_emb, context, mask_tgt, mask_src, buffer=buffer)

            decoder_state.update_attention_buffer(buffer, i)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        output = self.postprocess_layer(output)

        return output, coverage