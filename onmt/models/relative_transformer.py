import torch
import torch.nn as nn
from onmt.models.transformer_layers import PositionalEncoding, PrePostProcessing
from onmt.models.transformer_layers import EncoderLayer, DecoderLayer
from onmt.models.tranformers import TransformerEncoder, TransformerDecoder
import onmt
from onmt.modules.dropout import embedded_dropout
from onmt.models.transformer_layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing
from onmt.models.relative_transformer_layers import RelativeTransformerEncoderLayer, RelativeTransformerDecoderLayer
from onmt.utils import flip, expected_length
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


class LearnablePostionEmbedding(nn.Module):

    def __init__(self, max_pos, demb):
        super(LearnablePostionEmbedding, self).__init__()
        self.max_pos = max_pos
        self.embedding = nn.Embedding(2 * max_pos + 1, demb)

    def forward(self, input):
        pos = torch.clamp(input, -self.max_pos, self.max_pos)
        k = min((pos.size(0) - 1) // 2, self.max_pos)
        return self.embedding(pos + k)


class RelativeTransformerEncoder(TransformerEncoder):

    def __init__(self, opt, dicts, positional_encoder, encoder_type='text'):
        self.death_rate = opt.death_rate
        self.double_position = opt.double_position
        self.learnable_position_encoding = opt.learnable_position_encoding
        self.layer_modules = list()

        # build_modules will be called from the inherited constructor
        super(RelativeTransformerEncoder, self).__init__(opt, dicts, positional_encoder, encoder_type)

        # learnable position encoding
        if self.learnable_position_encoding:
            self.max_pos_length = opt.max_pos_length
            self.positional_encoder = LearnablePostionEmbedding(self.max_pos_length, self.model_size)
            print("Learnable position encoding")
        else:
            # or using pre-set sinusoidal
            self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)

        self.d_head = self.model_size // self.n_heads

        e_length = expected_length(self.layers, self.death_rate)
        print("* Transformer Encoder with Relative Attention with %.2f expected layers" % e_length)

    def build_modules(self):
        self.layer_modules = nn.ModuleList()

        for _l in range(self.layers):
            # linearly decay the death rate
            death_r = (_l + 1.0) / self.layers * self.death_rate

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

            if self.double_position:
                assert input_pos is not None
                # flatten
                src_len, bsz = input_pos.size(0), input_pos.size(1)
                input_pos_ = input_pos.contiguous().view(-1).type_as(emb)
                abs_pos = self.positional_encoder(input_pos_)
                abs_pos = abs_pos.squeeze(1).view(src_len, bsz, -1)

            else:
                abs_pos = None
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

        """ Scale the emb by sqrt(d_model) """
        emb = emb * math.sqrt(self.model_size)

        if self.double_position and abs_pos is not None:
            # adding position encoding
            emb = emb + abs_pos

        """ Adding positional encoding """
        klen = input.size(0)

        if not self.learnable_position_encoding:
            pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)
            # L x 1 x H
            pos_emb = self.positional_encoder(pos)
        else:
            pos = torch.arange(klen - 1, -klen, -1.0, device=emb.device, dtype=input.dtype)
            # clamp the positions (all postions from afar are treated equally, maybe?)
            # (2L-1) x 1 x H
            pos_emb = self.positional_encoder(pos.unsqueeze(1))
            # print(pos_emb.size())

        # B x T x H -> T x B x H
        context = emb

        # Apply dropout to both context and pos_emb
        context = self.preprocess_layer(context)

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

        return output_dict


class RelativeTransformerDecoder(TransformerDecoder):
    """Encoder in 'Attention is all you need'

    Args:
        opt
        dicts


    """

    def __init__(self, opt, dicts, positional_encoder, attribute_embeddings=None, ignore_source=False):

        self.death_rate = opt.death_rate
        self.double_position = opt.double_position

        # build_modules will be called from the inherited constructor
        super(RelativeTransformerDecoder, self).__init__(opt, dicts,
                                                         positional_encoder,
                                                         attribute_embeddings,
                                                         ignore_source,
                                                         allocate_positions=False)
        self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)
        self.d_head = self.model_size // self.n_heads
        # Parameters for the position biases
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))

        e_length = expected_length(self.layers, self.death_rate)
        # # Parameters for the position biases
        # self.r_w_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))
        # self.r_r_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))

        print("* Transformer Decoder with Relative Attention with %.2f expected layers" % e_length)

    def renew_buffer(self, new_len):
        return

    def build_modules(self):
        self.layer_modules = nn.ModuleList()

        for l in range(self.layers):
            # linearly decay the death rate
            death_r = (l + 1.0) / self.layers * self.death_rate

            block = RelativeTransformerDecoderLayer(self.n_heads, self.model_size,
                                                    self.dropout, self.inner_size, self.attn_dropout,
                                                    variational=self.variational_dropout, death_rate=death_r)

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

    def forward(self, input, context, src, input_pos=None, atbs=None, **kwargs):
        """
                Inputs Shapes:
                    input: (Variable) batch_size x len_tgt (wanna tranpose)
                    context: (Variable) batch_size x src_len x d_model
                    mask_src (Tensor) batch_size x src_len
                Outputs Shapes:
                    out: batch_size x len_tgt x d_model
                    coverage: batch_size x len_tgt x src_len

                """

        """ Embedding: batch_size x len_tgt x d_model """
        input = input.transpose(0, 1)
        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        emb = emb * math.sqrt(self.model_size)

        if self.double_position:
            assert input_pos is not None
            tgt_len, bsz = input_pos.size(0), input_pos.size(1)
            input_pos_ = input_pos.view(-1).type_as(emb)
            abs_pos = self.positional_encoder(input_pos_).squeeze(1).view(tgt_len, bsz, -1)

            emb = emb + abs_pos

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

        qlen = input.size(0)
        klen = input.size(0)
        mlen = klen - qlen  # extra memory if expanded
        # preparing self-attention mask. The input is either left or right aligned
        dec_attn_mask = torch.triu(
            emb.new_ones(qlen, klen), diagonal=1 + mlen).byte()[:, :, None]
        pad_mask = input.eq(onmt.constants.PAD).byte()  # L x B

        dec_attn_mask = dec_attn_mask + pad_mask.unsqueeze(0)
        dec_attn_mask = dec_attn_mask.gt(0)
        if onmt.constants.torch_version >= 1.2:
            dec_attn_mask = dec_attn_mask.bool()

        pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)

        pos_emb = self.positional_encoder(pos)

        output = self.preprocess_layer(emb.contiguous())

        pos_emb = self.preprocess_layer(pos_emb)

        for i, layer in enumerate(self.layer_modules):
            # batch_size x src_len x d_model
            # output, coverage = layer(output, context, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask, mask_src)
            output, coverage = layer(output, context, pos_emb, dec_attn_mask, mask_src)

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
            context: (Variable) batch_size x src_len x d_model
            mask_src (Tensor) batch_size x src_len
            buffer (List of tensors) List of batch_size * len_tgt-1 * d_model for self-attention recomputing
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x src_len

        """
        context = decoder_state.context
        buffers = decoder_state.attention_buffers
        atbs = decoder_state.tgt_atb
        mask_src = decoder_state.src_mask

        if decoder_state.concat_input_seq:
            if decoder_state.input_seq is None:
                decoder_state.input_seq = input
            else:
                # concatenate the last input to the previous input sequence
                decoder_state.input_seq = torch.cat([decoder_state.input_seq, input], 0)
            input = decoder_state.input_seq.transpose(0, 1)  # B x T

        src = decoder_state.src.transpose(0, 1) if decoder_state.src is not None else None

        # use the last value of input to continue decoding
        if input.size(1) > 1:
            input_ = input[:, -1].unsqueeze(1).transpose(0, 1)
        else:
            input_ = input.transpose(0, 1)
        """ Embedding: batch_size x 1 x d_model """
        emb = self.word_lut(input_) * math.sqrt(self.model_size)
        input = input.transpose(0, 1)
        klen = input.size(0)
        # emb = self.word_lut(input) * math.sqrt(self.model_size)

        if self.double_position:
            input_pos = torch.arange(input.size(0), dtype=emb.dtype, device=emb.device)
            input_pos = input_pos.unsqueeze(1).repeat(1, input.size(1))
            tgt_len, bsz = input_pos.size(0), input_pos.size(1)
            input_pos_ = input_pos.view(-1).type_as(emb)
            abs_pos = self.positional_encoder(input_pos_).squeeze(1).view(tgt_len, bsz, -1)
            # print(abs_pos.size(), emb.size())
            emb = emb + abs_pos[-1:, :, :]

        # prepare position encoding
        qlen = emb.size(0)
        mlen = klen - qlen

        pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)

        pos_emb = self.positional_encoder(pos)

        dec_attn_mask = torch.triu(
            emb.new_ones(qlen, klen), diagonal=1 + mlen).byte()[:, :, None]

        pad_mask = input.eq(onmt.constants.PAD).byte()  # L x B

        dec_attn_mask = dec_attn_mask + pad_mask.unsqueeze(0)
        dec_attn_mask = dec_attn_mask.gt(0)

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

        output = emb.contiguous()

        for i, layer in enumerate(self.layer_modules):
            buffer = buffers[i] if i in buffers else None
            # assert (output.size(0) == 1)

            output, coverage, buffer = layer.step(output, context, pos_emb,
                                                  dec_attn_mask, mask_src, buffer=buffer)
            # output, coverage = layer(output, context, pos_emb, dec_attn_mask, mask_src)

            decoder_state.update_attention_buffer(buffer, i)

        output = self.postprocess_layer(output)
        # print(output.size())
        output = output[-1].unsqueeze(0)

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = output
        output_dict['coverage'] = coverage
        output_dict['context'] = context

        return output_dict


class RelativeTransformer(TransformerDecoder):
    """
    This class combines the encoder and the decoder into one single sequence
    Joined attention between encoder and decoder parts
    """

    def __init__(self, opt, src_embedding, tgt_embedding, generator, positional_encoder, encoder_type='text', **kwargs):
        self.death_rate = opt.death_rate
        self.layer_modules = []

        # build_modules will be called from the inherited constructor
        super(RelativeTransformer, self).__init__(opt, tgt_embedding,
                                                  positional_encoder,
                                                  allocate_positions=False)
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        # self.language_embedding = nn.Embedding(3, self.model_size, padding_idx=0)
        self.generator = generator
        self.ignore_source = True
        self.encoder_type = opt.encoder_type

        self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)
        self.d_head = self.model_size // self.n_heads

        e_length = expected_length(self.layers, self.death_rate)

        print("* Transformer Decoder with Relative Attention with %.2f expected layers" % e_length)

        self.build_modules()

    def build_modules(self):
        self.layer_modules = nn.ModuleList()

        for l in range(self.layers):
            # linearly decay the death rate
            death_r = (l + 1.0) / self.layers * self.death_rate

            block = RelativeTransformerDecoderLayer(self.n_heads, self.model_size,
                                                    self.dropout, self.inner_size, self.attn_dropout,
                                                    ignore_source=True,
                                                    variational=self.variational_dropout, death_rate=death_r)
            self.layer_modules.append(block)

    def forward(self, batch, target_mask=None, **kwargs):

        src = batch.get('source')  # src_len x batch_size
        tgt = batch.get('target_input')  # len_tgt x batch_size

        tgt_len = tgt.size(0)
        src_len = src.size(0)
        bsz = tgt.size(1)

        # Embedding stage
        src_emb = embedded_dropout(self.src_embedding, src, dropout=self.word_dropout if self.training else 0)
        tgt_emb = embedded_dropout(self.tgt_embedding, tgt, dropout=self.word_dropout if self.training else 0)

        # src_lang = src.new(*src.size()).fill_(1)
        # tgt_lang = tgt.new(*tgt.size()).fill_(1)

        # concatenate embedding
        emb = torch.cat([src_emb, tgt_emb], dim=0)  # L x batch_size x H
        emb = emb * math.sqrt(self.model_size)

        # prepare self-attention mask
        # For the source: we have two different parts
        # [1 x src_len x batch_size]
        mask_src_src = src.eq(onmt.constants.PAD).unsqueeze(0).byte()
        src_pad_mask = mask_src_src
        mask_src_tgt = mask_src_src.new_ones(1, 1, 1).expand(src_len, tgt_len, bsz)
        # [src_len x L x batch_size]
        mask_src = torch.cat([mask_src_src.expand(src_len, src_len, bsz), mask_src_tgt], dim=1)
        mask_src = mask_src.bool()

        # For the target:
        qlen = tgt_len
        klen = tgt_len + src_len

        mlen = klen - qlen  # extra memory if expanded
        # preparing self-attention mask. The input is either left or right aligned
        dec_attn_mask = torch.triu(
            emb.new_ones(qlen, klen), diagonal=1 + mlen).byte()[:, :, None]  # tgt_len x L x 1
        dec_pad_mask = tgt.eq(onmt.constants.PAD).byte().unsqueeze(0)  # 1 x len_tgt x batch_size
        dec_pad_mask = torch.cat([src_pad_mask, dec_pad_mask], dim=1)  # 1 x L x batch_size

        dec_attn_mask = dec_attn_mask + dec_pad_mask
        dec_attn_mask = dec_attn_mask.gt(0)
        dec_mask = dec_attn_mask.bool()  # tgt_len x L x batch_size

        attn_mask = torch.cat([mask_src, dec_mask], dim=0)  # L x L x batch_size

        output = emb

        # position encoding:
        pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)
        pos_emb = self.positional_encoder(pos)

        # Applying dropout
        output = self.preprocess_layer(output)
        pos_emb = self.preprocess_layer(pos_emb)

        # FORWARD PASS
        coverage = None
        for i, layer in enumerate(self.layer_modules):
            output, coverage = layer(output, None, pos_emb, attn_mask, None)

        # Final normalization
        output = self.postprocess_layer(output)

        # extract the "source" and "target" parts of the output
        context = output[:src_len, :, :]
        output = output[-tgt_len:, :, :]
        output_dict = {'hidden': output, 'coverage': coverage, 'context': context, 'src': src,
                       'target_mask': target_mask}

        # final layer: computing log probabilities
        logprobs = self.generator[0](output_dict)
        output_dict['logprobs'] = logprobs

        return output_dict

    def decode(self, batch):
        raise NotImplementedError

    def renew_buffer(self, new_len):
        return

    def reset_states(self):
        return

    def step(self, input_t, decoder_state):
        raise NotImplementedError

    def create_decoder_state(self, batch, beam_size=1, type=1):
        raise NotImplementedError

    def tie_weights(self):
        assert self.generator is not None, "The generator needs to be created before sharing weights"
        self.generator[0].linear.weight = self.tgt_embedding.weight

    def share_enc_dec_embedding(self):
        self.src_embedding.weight = self.tgt_embedding.weight


class RelativeTransformer(TransformerDecoder):
    """
    This class combines the encoder and the decoder into one single sequence
    Joined attention between encoder and decoder parts
    """

    def __init__(self, opt, src_embedding, tgt_embedding, generator, positional_encoder, encoder_type='text', **kwargs):
        self.death_rate = opt.death_rate
        self.layer_modules = []

        # build_modules will be called from the inherited constructor
        super(RelativeTransformer, self).__init__(opt, tgt_embedding,
                                                  positional_encoder,
                                                  allocate_positions=False)
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        # self.language_embedding = nn.Embedding(3, self.model_size, padding_idx=0)
        self.generator = generator
        self.ignore_source = True
        self.encoder_type = opt.encoder_type

        self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)
        self.d_head = self.model_size // self.n_heads

        e_length = expected_length(self.layers, self.death_rate)

        print("* Transformer Decoder with Relative Attention with %.2f expected layers" % e_length)

        self.build_modules()

    def build_modules(self):
        self.layer_modules = nn.ModuleList()

        for l in range(self.layers):
            # linearly decay the death rate
            death_r = (l + 1.0) / self.layers * self.death_rate

            block = RelativeTransformerDecoderLayer(self.n_heads, self.model_size,
                                                    self.dropout, self.inner_size, self.attn_dropout,
                                                    ignore_source=True,
                                                    variational=self.variational_dropout, death_rate=death_r)
            self.layer_modules.append(block)

    def forward(self, batch, target_mask=None, **kwargs):

        src = batch.get('source')  # src_len x batch_size
        tgt = batch.get('target_input')  # len_tgt x batch_size

        tgt_len = tgt.size(0)
        src_len = src.size(0)
        bsz = tgt.size(1)

        # Embedding stage
        src_emb = embedded_dropout(self.src_embedding, src, dropout=self.word_dropout if self.training else 0)
        tgt_emb = embedded_dropout(self.tgt_embedding, tgt, dropout=self.word_dropout if self.training else 0)

        # src_lang = src.new(*src.size()).fill_(1)
        # tgt_lang = tgt.new(*tgt.size()).fill_(1)

        # concatenate embedding
        emb = torch.cat([src_emb, tgt_emb], dim=0)  # L x batch_size x H
        emb = emb * math.sqrt(self.model_size)

        # prepare self-attention mask
        # For the source: we have two different parts
        # [1 x src_len x batch_size]
        mask_src_src = src.eq(onmt.constants.PAD).unsqueeze(0).byte()
        src_pad_mask = mask_src_src
        mask_src_tgt = mask_src_src.new_ones(1, 1, 1).expand(src_len, tgt_len, bsz)
        # [src_len x L x batch_size]
        mask_src = torch.cat([mask_src_src.expand(src_len, src_len, bsz), mask_src_tgt], dim=1)
        mask_src = mask_src.bool()

        # For the target:
        qlen = tgt_len
        klen = tgt_len + src_len

        mlen = klen - qlen  # extra memory if expanded
        # preparing self-attention mask. The input is either left or right aligned
        dec_attn_mask = torch.triu(
            emb.new_ones(qlen, klen), diagonal=1 + mlen).byte()[:, :, None]  # tgt_len x L x 1
        dec_pad_mask = tgt.eq(onmt.constants.PAD).byte().unsqueeze(0)  # 1 x len_tgt x batch_size
        dec_pad_mask = torch.cat([src_pad_mask, dec_pad_mask], dim=1)  # 1 x L x batch_size

        dec_attn_mask = dec_attn_mask + dec_pad_mask
        dec_attn_mask = dec_attn_mask.gt(0)
        dec_mask = dec_attn_mask.bool()  # tgt_len x L x batch_size

        attn_mask = torch.cat([mask_src, dec_mask], dim=0)  # L x L x batch_size

        output = emb

        # position encoding:
        pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)
        pos_emb = self.positional_encoder(pos)

        # Applying dropout
        output = self.preprocess_layer(output)
        pos_emb = self.preprocess_layer(pos_emb)

        # FORWARD PASS
        coverage = None
        for i, layer in enumerate(self.layer_modules):
            output, coverage = layer(output, None, pos_emb, attn_mask, None)

        # Final normalization
        output = self.postprocess_layer(output)

        # extract the "source" and "target" parts of the output
        context = output[:src_len, :, :]
        output = output[-tgt_len:, :, :]
        output_dict = {'hidden': output, 'coverage': coverage, 'context': context, 'src': src,
                       'target_mask': target_mask}

        # final layer: computing log probabilities
        logprobs = self.generator[0](output_dict)
        output_dict['logprobs'] = logprobs

        return output_dict

    def decode(self, batch):
        raise NotImplementedError

    def renew_buffer(self, new_len):
        return

    def reset_states(self):
        return

    def step(self, input_t, decoder_state):
        raise NotImplementedError

    def create_decoder_state(self, batch, beam_size=1, type=1):
        raise NotImplementedError

    def tie_weights(self):
        assert self.generator is not None, "The generator needs to be created before sharing weights"
        self.generator[0].linear.weight = self.tgt_embedding.weight

    def share_enc_dec_embedding(self):
        self.src_embedding.weight = self.tgt_embedding.weight
