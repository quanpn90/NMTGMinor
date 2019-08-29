import torch, math
import torch.nn as nn
from onmt.modules.Transformer.Layers import PositionalEncoding, PrePostProcessing
from onmt.modules.Transformer.Layers import EncoderLayer, DecoderLayer
from onmt.modules.RelativeTransformer.Layers import RelativeTransformerDecoderLayer, RelativeTransformerEncoderLayer
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder
import onmt
from onmt.modules.WordDrop import embedded_dropout
from onmt.modules.Transformer.Layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing
from onmt.utils import flip


class RelativeTransformerEncoder(TransformerEncoder):

    def __init__(self, opt, embedding, positional_encoder, encoder_type='text'):
        """
        :param opt:
        :param embedding:
        :param positional_encoder:
        :param encoder_type:
        """
        self.death_rate = opt.death_rate
        self.layer_modules = None

        # build_modules will be called from the inherited constructor
        super(RelativeTransformerEncoder, self).__init__(opt, embedding, positional_encoder, encoder_type=encoder_type)

        self.positional_encoder = positional_encoder

    def build_modules(self):
        self.layer_modules = nn.ModuleList(
            [RelativeTransformerEncoderLayer(self.n_heads, self.model_size,
                                             self.dropout, self.inner_size,
                                             self.attn_dropout) for _ in
             range(self.layers)])

    def forward(self, input, **kwargs):
        """
        Inputs Shapes:
            input: batch_size x len_src

        Outputs Shapes:
            out: batch_size x len_src x d_model
            mask_src

        """

        """ Embedding: batch_size x len_src x d_model """
        # if self.input_type == "text":
        #     mask_src = input.eq(onmt.Constants.PAD).byte()  # batch_size x len_src x 1 for broadcasting
        #     emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        # else:
        #     raise NotImplementedError
        # if not self.cnn_downsampling:
        #     mask_src = input.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD)
        #     input = input.narrow(2, 1, input.size(2) - 1)
        #     emb = self.audio_trans(input.contiguous().view(-1, input.size(2))).view(input.size(0),
        #                                                                             input.size(1), -1)
        # else:
        #     long_mask = input.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD)
        #     input = input.narrow(2, 1, input.size(2) - 1)
        #
        #     # first resizing to fit the CNN format
        #     input = input.view(input.size(0), input.size(1), -1, self.channels)
        #     input = input.permute(0, 3, 1, 2)
        #
        #     input = self.audio_trans(input)
        #     input = input.permute(0, 2, 1, 3).contiguous()
        #     input = input.view(input.size(0), input.size(1), -1)
        #
        #     mask_src = long_mask[:, 0:input.size(1) * 4:4]
        #     emb = input

        input = input.transpose(0, 1)  # B x T to T x B
        klen, batch_size = input.size()

        """ Scale the emb by sqrt(d_model) """
        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        emb = emb * (math.sqrt(self.model_size))

        # Adding dropout
        emb = self.preprocess_layer(emb)

        # Prepare positional encoding:
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)
        pos_emb = self.preprocess_layer(self.positional_encoder(pos_seq))

        # attention masking
        qlen = klen
        mask = torch.triu(emb.new_ones(qlen, klen), diagonal=1).byte()
        mask_fwd = input.t().eq(onmt.Constants.PAD).byte().unsqueeze(1) + mask
        mask_fwd = torch.gt(mask_fwd, 0)
        mask_fwd = mask_fwd.bool()

        input_flip = flip(input, 0)
        # mask_bwd = mask + input_flip.eq(onmt.Constants.PAD).unsqueeze(0).byte()
        mask_bwd = input_flip.t().eq(onmt.Constants.PAD).byte().unsqueeze(1) + \
            torch.triu(emb.new_ones(qlen, klen), diagonal=1).byte()
        mask_bwd = torch.gt(mask_bwd, 0)  # convert all 2s to 1
        mask_bwd = mask_bwd.bool()

        context = emb
        for i, layer in enumerate(self.layer_modules):
            context = layer(context, pos_emb, mask_fwd, mask_bwd)  # batch_size x len_src x d_model

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        context = self.postprocess_layer(context)

        output_dict = {'context': context, 'src_mask': None}

        # return context, mask_src
        return output_dict


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
                    mask_src = src.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD).byte().unsqueeze(1)
                else:
                    long_mask = src.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD).byte()
                    mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
            else:
                mask_src = src.eq(onmt.Constants.PAD).byte().unsqueeze(1)
        else:
            mask_src = None

        mask_src = mask_src.bool()

        # attention masking
        qlen = klen
        # mask_tgt = torch.triu(emb.new_ones(qlen, klen), diagonal=1).unsqueeze(-1).byte()
        # mask_tgt = mask_tgt + input.eq(onmt.Constants.PAD).byte().unsqueeze(0)
        # mask_tgt = torch.gt(mask_tgt, 0)  # convert all 2s to 1
        # mask_tgt = mask_tgt.bool()
        mask_tgt = input.t().eq(onmt.Constants.PAD).byte().unsqueeze(1) + \
                   torch.triu(emb.new_ones(qlen, klen), diagonal=1).byte()
        mask_tgt = torch.gt(mask_tgt, 0)
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

        mask_tgt = input.t().eq(onmt.Constants.PAD).byte().unsqueeze(1) + \
            torch.triu(emb.new_ones(qlen, klen), diagonal=1).byte()
        mask_tgt = torch.gt(mask_tgt, 0)
        mask_tgt = mask_tgt.bool()
        # mask_tgt = torch.triu(emb.new_ones(qlen, klen), diagonal=1).unsqueeze(-1).byte()
        # mask_tgt = mask_tgt + input.eq(onmt.Constants.PAD).byte().unsqueeze(0)
        # mask_tgt = torch.gt(mask_tgt, 0)  # convert all 2s to 1
        # mask_tgt = mask_tgt.bool()

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