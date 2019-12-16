import torch
import torch.nn as nn
from onmt.models.transformer_layers import PositionalEncoding, PrePostProcessing
from onmt.models.transformer_layers import EncoderLayer, DecoderLayer
from onmt.models.tranformers import TransformerEncoder, TransformerDecoder
import onmt
from onmt.modules.dropout import embedded_dropout
from onmt.models.transformer_layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing
from onmt.models.relative_transformer_layers import RelativeTransformerEncoderLayer, RelativeTransformerDecoderLayer
from onmt.utils import flip
from collections import defaultdict
import math


#  Positional Embedding with discrete inputs
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(SinusoidalPositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class RelativeTransformerEncoder(TransformerEncoder):

    def __init__(self, opt, dicts, positional_encoder, encoder_type='text'):
        self.death_rate = opt.death_rate

        # build_modules will be called from the inherited constructor
        super(RelativeTransformerEncoder, self).__init__(opt, dicts, positional_encoder, encoder_type)

        self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)
        self.d_head = self.model_size // self.n_heads

        # Parameters for the position biases
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))

        print("* Transformer Encoder with Relative Attention")

    def build_modules(self):
        self.layer_modules = nn.ModuleList()

        for l in range(self.layers):

            block = RelativeTransformerEncoderLayer(self.n_heads, self.model_size,
                                                    self.dropout, self.inner_size, self.attn_dropout)

            self.layer_modules.append(block)

    def forward(self, input, **kwargs):
        """
        Inputs Shapes:
            input: batch_size x len_src (wanna tranpose)

        Outputs Shapes:
            out: batch_size x len_src x d_model
            mask_src

        """

        """ Embedding: batch_size x len_src x d_model """
        if self.input_type == "text":
            # mask_src = input.eq(onmt.constants.PAD).unsqueeze(1)  # batch_size x len_src x 1 for broadcasting
            mask_src = input.transpose(0, 1).eq(onmt.constants.PAD).unsqueeze(0)
            dec_attn_mask = input.eq(onmt.constants.PAD).unsqueeze(1)

            # apply switchout
            # if self.switchout > 0 and self.training:
            #     vocab_size = self.word_lut.weight.size(0)
            #     input = switchout(input, vocab_size, self.switchout)
            emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
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

        if onmt.constants.torch_version >= 1.2:
            mask_src = mask_src.bool()

        """ Scale the emb by sqrt(d_model) """
        emb = emb * math.sqrt(self.model_size)

        """ Adding positional encoding """
        klen = input.size(1)
        pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)

        pos_emb = self.positional_encoder(pos)

        # B x T x H -> T x B x H
        context = emb.transpose(0, 1)

        # Apply dropout to both context and pos_emb
        context = self.preprocess_layer(context)

        pos_emb = self.preprocess_layer(pos_emb)

        for i, layer in enumerate(self.layer_modules):

            # len_src x batch_size x d_model
            context = layer(context, pos_emb, self.r_w_bias, self.r_r_bias, mask_src)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        context = self.postprocess_layer(context)

        output_dict = {'context': context, 'src_mask': dec_attn_mask, 'src': input}

        # return context, mask_src
        return output_dict


class RelativeTransformerDecoder(TransformerDecoder):
    """Encoder in 'Attention is all you need'

    Args:
        opt
        dicts


    """

    def __init__(self, opt, dicts, positional_encoder, attribute_embeddings=None, ignore_source=False):

        # build_modules will be called from the inherited constructor
        super(RelativeTransformerDecoder, self).__init__(opt, dicts,
                                                         positional_encoder,
                                                         attribute_embeddings,
                                                         ignore_source)

        self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)
        self.d_head = self.model_size // self.n_heads

        # Parameters for the position biases
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))

        print("* Transformer Decoder with Relative Attention")

    def renew_buffer(self, new_len):
        return

    def build_modules(self):
        self.layer_modules = nn.ModuleList()

        for l in range(self.layers):
            # linearly decay the death rate
            block = RelativeTransformerDecoderLayer(self.n_heads, self.model_size,
                                                    self.dropout, self.inner_size, self.attn_dropout)

            self.layer_modules.append(block)

    def process_embedding(self, input, atbs=None):

        input_ = input

        emb = embedded_dropout(self.word_lut, input_, dropout=self.word_dropout if self.training else 0)

        emb = emb * math.sqrt(self.model_size)

        if self.use_feature:
            len_tgt = emb.size(1)
            atb_emb = self.attribute_embeddings(atbs).unsqueeze(1).repeat(1, len_tgt, 1)  # B x H to 1 x B x H
            emb = torch.cat([emb, atb_emb], dim=-1)
            emb = torch.relu(self.feature_projector(emb))
        return emb

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
        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        emb = emb * math.sqrt(self.model_size)

        if context is not None:
            if self.encoder_type == "audio":
                if not self.encoder_cnn_downsampling:
                    mask_src = src.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD).unsqueeze(1)
                else:
                    long_mask = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD)
                    mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
            else:

                mask_src = src.eq(onmt.constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        qlen = input.size(1)
        klen = input.size(1)
        mlen = 0  # extra memory if expanded
        # preparing self-attention mask. The input is either left or right aligned
        dec_attn_mask = torch.triu(
            emb.new_ones(qlen, klen), diagonal=1 + mlen).byte()[:, :, None]
        pad_mask = input.transpose(0, 1).eq(onmt.constants.PAD).byte()  # L x B

        # dec_attn_mask = dec_attn_mask + pad_mask.unsqueeze(0)
        # dec_attn_mask = dec_attn_mask.gt(0)
        if onmt.constants.torch_version >= 1.2:
            dec_attn_mask = dec_attn_mask.bool()

        pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)

        pos_emb = self.positional_encoder(pos)

        output = self.preprocess_layer(emb.transpose(0, 1).contiguous())

        pos_emb = self.preprocess_layer(pos_emb)

        for i, layer in enumerate(self.layer_modules):
            # batch_size x len_src x d_model
            output, coverage = layer(output, context, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask, mask_src)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        output = self.postprocess_layer(output)

        output_dict = {'hidden': output, 'coverage': coverage, 'context': context}

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
        atbs = decoder_state.tgt_atb
        mask_src = decoder_state.src_mask

        if decoder_state.concat_input_seq == True:
            if decoder_state.input_seq is None:
                decoder_state.input_seq = input
            else:
                # concatenate the last input to the previous input sequence
                decoder_state.input_seq = torch.cat([decoder_state.input_seq, input], 0)
            input = decoder_state.input_seq.transpose(0, 1)  # B x T

        src = decoder_state.src.transpose(0, 1) if decoder_state.src is not None else None

        # use the last value of input to continue decoding
        # if input.size(1) > 1:
        #     input_ = input[:, -1].unsqueeze(1)
        # else:
        #     input_ = input
        """ Embedding: batch_size x 1 x d_model """
        # emb = self.word_lut(input_)  # * math.sqrt(self.model_size)
        emb = self.word_lut(input) * math.sqrt(self.model_size)

        # prepare position encoding
        klen = input.size(1)
        qlen = input.size(1)
        mlen = 0

        pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)

        pos_emb = self.positional_encoder(pos)

        dec_attn_mask = torch.triu(
            emb.new_ones(qlen, klen), diagonal=1 + mlen).byte()[:, :, None]

        pad_mask = input.transpose(0, 1).eq(onmt.constants.PAD).byte()  # L x B

        # dec_attn_mask = dec_attn_mask + pad_mask.unsqueeze(0)
        # dec_attn_mask = dec_attn_mask.gt(0)

        if onmt.constants.torch_version >= 1.2:
            dec_attn_mask = dec_attn_mask.bool()

        if context is not None:
            if self.encoder_type == "audio":
                if not self.encoder_cnn_downsampling:
                    mask_src = src.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD).unsqueeze(1)
                else:
                    long_mask = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD)
                    mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
            else:

                mask_src = src.eq(onmt.constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        if self.use_feature:
            atb_emb = self.attribute_embeddings(atbs).unsqueeze(1)  # B x H to B x 1 x H
            emb = torch.cat([emb, atb_emb], dim=-1)
            emb = torch.relu(self.feature_projector(emb))

        emb = emb.transpose(0, 1)

        output = emb.contiguous()

        for i, layer in enumerate(self.layer_modules):
            buffer = buffers[i] if i in buffers else None
            # assert (output.size(0) == 1)

            # output, coverage, buffer = layer.step(output, context, pos_emb, self.r_w_bias, self.r_r_bias,
            #                                       dec_attn_mask, mask_src, buffer=buffer)
            output, coverage = layer(output, context, pos_emb, self.r_w_bias, self.r_r_bias,
                                                  dec_attn_mask, mask_src)

            # decoder_state.update_attention_buffer(buffer, i)

        output = self.postprocess_layer(output)

        output = output[-1].unsqueeze(0)

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = output
        output_dict['coverage'] = coverage
        output_dict['context'] = context

        return output_dict

