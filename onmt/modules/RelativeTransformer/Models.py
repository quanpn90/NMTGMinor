import torch, math
import torch.nn as nn
from onmt.modules.Transformer.Layers import PositionalEncoding, PrePostProcessing
from onmt.modules.Transformer.Layers import EncoderLayer, DecoderLayer
from onmt.modules.RelativeTransformer.Layers import RelativeTransformerDecoderLayer
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder
import onmt
from onmt.modules.WordDrop import embedded_dropout
from onmt.modules.Transformer.Layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing
from onmt.utils import flip
from collections import defaultdict

# This model doesn't have decoder or encoder
class RelativeTransformer(nn.Module):
    """Encoder in 'Attention is all you need'

    Args:
        opt
        dicts


    """

    def __init__(self, opt, embeddings, positional_encoder, attribute_embeddings=None, generator=None):
        """
        :param opt: Options
        :param embeddings: a list of two embedding tables [src tgt]
        :param positional_encoder: The sinusoidal positional encoding
        :param attribute_embeddings: To be implemented
        """
        super(RelativeTransformer, self).__init__()

        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout
        self.time = opt.time
        self.version = opt.version
        self.encoder_type = opt.encoder_type
        self.encoder_cnn_downsampling = opt.cnn_downsampling
        self.variational_dropout = opt.variational_dropout
        self.switchout = opt.switchout
        self.death_rate = opt.death_rate
        self.layer_modules = None
        self.use_feature = False

        self.d_head = self.model_size // self.n_heads`

        if self.switchout > 0:
            self.word_dropout = 0

        self.positional_encoder = positional_encoder
        self.relative = True
        # two embedding layers for src and tgt
        self.src_word_lut = embeddings[0]
        self.tgt_word_lut = embeddings[1]
        self.generator = generator

        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d',
                                                  variational=self.variational_dropout)

        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')

        self.build_modules()

    def build_modules(self):
        self.layer_modules = nn.ModuleList([RelativeTransformerDecoderLayer
                                            (self.n_heads, self.model_size,
                                             self.dropout, self.inner_size,
                                             self.attn_dropout,
                                             variational=self.variational_dropout) for _ in range(self.layers)])

        self.encoder = nn.Module()
        self.encoder.word_lut = self.src_word_lut
        self.decoder = nn.Module()
        self.decoder.word_lut = self.tgt_word_lut

        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))

        torch.nn.init.uniform_(self.r_w_bias, -0.1, 0.1)
        torch.nn.init.uniform_(self.r_r_bias, -0.1, 0.1)

    # def forward_seq2seq(self, batch, input, src, atbs=None, **kwargs):
    def forward_seq2seq(self, batch, target_masking=None, zero_encoder=False):
        """
        Inputs Shapes:
            input: (Variable) batch_size x len_tgt (wanna tranpose)
            context: (Variable) batch_size x len_src x d_model
            mask_src (Tensor) batch_size x len_src
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src

        """
        src = batch.get('source')
        tgt = batch.get('target_input')
        input = torch.cat([src, tgt], dim=0)

        """ Embedding: batch_size x len_tgt x d_model """

        # we work with two embeddings at the same time
        src_emb = embedded_dropout(self.src_word_lut, src, dropout=self.word_dropout if self.training else 0)
        tgt_emb = embedded_dropout(self.tgt_word_lut, tgt, dropout=self.word_dropout if self.training else 0)

        # Concatenate the embeddings by time dimension
        emb = torch.cat([src_emb, tgt_emb], dim=0)

        # Add dropout and scale
        emb = self.preprocess_layer(emb)
        emb = emb * math.sqrt(self.model_size)

        klen, batch_size = emb.size(0), emb.size(1)

        # Prepare positional encoding:
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)

        # pos_seq = torch.arange(0, klen, device=emb.device, dtype=emb.dtype)
        pos_emb = self.preprocess_layer(self.positional_encoder(pos_seq))

        if self.use_feature:
            raise NotImplementedError  # No feature/attributes for the moment

        # attention masking
        qlen = klen

        mlen = 0  # we don't have any memory in this mode

        # print(input)
        dec_attn_mask = torch.triu(
            emb.new_ones(qlen, klen), diagonal=1 + mlen).byte()[:, :, None]  #  Size T x T ?
        pad_mask = input.eq(onmt.Constants.PAD).byte().unsqueeze(1)  # Size 1 x T x B
        # pad_mask = input.new(*input.size()).zero_()
        mask = dec_attn_mask + pad_mask

        mask = torch.gt(mask, 0).bool()
        # mask = dec_attn_mask
        mask = mask.bool()
        output = emb

        for i, layer in enumerate(self.layer_modules):
            output, coverage = layer(output, pos_emb, self.r_w_bias, self.r_r_bias, mask)  # batch_size x len_src x d_model

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        output = self.postprocess_layer(output)

        all_output = output

        src_len = src.size(0)
        context = output[src_len:, :, :]

        tgt_len = tgt.size(0)
        tgt_hiddens = output[:tgt_len, :, :]
        # output_dict = {'hidden': output, 'coverage': coverage, 'context': context}

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = tgt_hiddens
        output_dict['encoder'] = context
        output_dict['src_mask'] = mask[src_len:, :, :]

        output = tgt_hiddens

        # This step removes the padding to reduce the load for the final layer
        if target_masking is not None:
            output = output.contiguous().view(-1, output.size(-1))

            mask = target_masking
            """ We remove all positions with PAD """
            flattened_mask = mask.view(-1)

            non_pad_indices = torch.nonzero(flattened_mask).squeeze(1)

            output = output.index_select(0, non_pad_indices)

        # final layer: computing softmax
        logprobs = self.generator[0](output)
        output_dict['logprobs'] = logprobs

        # return output, None
        return output_dict

    def forward(self, batch, target_masking=None, mode="seq2seq", **kwargs):

        if mode == "seq2seq":
            return self.forward_seq2seq(batch, target_masking)
        else:
            print("Streaming mode in progress.")
            raise NotImplementedError

    def step(self, input, decoder_state):
        """
        :param input:
        :param decoder_state:
        :return:
        """
        raise NotImplementedError
        # context = decoder_state.context
        # buffers = decoder_state.attention_buffers
        # src = decoder_state.src.transpose(0, 1) if decoder_state.src is not None else None
        # atbs = decoder_state.tgt_atb
        #
        # if decoder_state.input_seq is None:
        #     decoder_state.input_seq = input
        # else:
        #     # concatenate the last input to the previous input sequence
        #     decoder_state.input_seq = torch.cat([decoder_state.input_seq, input], 0)
        # # input = decoder_state.input_seq.transpose(0, 1)
        #
        # input = decoder_state.input_seq  # no need to transpose because time first
        # input_ = input[-1, :].unsqueeze(0)
        #
        # """ Embedding: batch_size x 1 x d_model """
        # emb = self.word_lut(input_)
        #
        # emb = emb * math.sqrt(self.model_size)
        #
        # if isinstance(emb, tuple):
        #     emb = emb[0]
        # # emb should be batch_size x 1 x dim
        #
        # # Prepare positional encoding:
        # klen = input.size(0)
        # pos_seq = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)
        # # pos_seq = torch.arange(0, klen, device=emb.device, dtype=emb.dtype)
        # pos_emb = self.preprocess_layer(self.positional_encoder(pos_seq))
        #
        # if self.use_feature:
        #     raise NotImplementedError
        #     # atb_emb = self.attribute_embeddings(atbs).unsqueeze(1)  # B x H to B x 1 x H
        #     # emb = torch.cat([emb, atb_emb], dim=-1)
        #     # emb = torch.relu(self.feature_projector(emb))
        #
        # # batch_size x 1 x len_src
        # if context is not None:
        #     if self.encoder_type == "audio":
        #         if src.data.dim() == 3:
        #             if self.encoder_cnn_downsampling:
        #                 long_mask = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD)
        #                 mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
        #             else:
        #                 mask_src = src.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD).unsqueeze(1)
        #         elif self.encoder_cnn_downsampling:
        #             long_mask = src.eq(onmt.Constants.PAD)
        #             mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
        #         else:
        #             mask_src = src.eq(onmt.Constants.PAD).unsqueeze(1)
        #     else:
        #         mask_src = src.eq(onmt.Constants.PAD).unsqueeze(1)
        # else:
        #     mask_src = None
        #
        # # attention masking
        # klen, batch_size = input.size()
        # qlen = klen
        #
        # mask_tgt = input.t().eq(onmt.Constants.PAD).unsqueeze(1) + \
        #     torch.triu(emb.new_ones(qlen, klen), diagonal=1)
        # mask_tgt = torch.gt(mask_tgt, 0)
        # # mask_tgt = mask_tgt.bool()
        # # mask_tgt = torch.triu(emb.new_ones(qlen, klen), diagonal=1).unsqueeze(-1).byte()
        # # mask_tgt = mask_tgt + input.eq(onmt.Constants.PAD).byte().unsqueeze(0)
        # # mask_tgt = torch.gt(mask_tgt, 0)  # convert all 2s to 1
        # # mask_tgt = mask_tgt.bool()
        #
        # output = emb.contiguous()
        #
        # for i, layer in enumerate(self.layer_modules):
        #     buffer = buffers[i] if i in buffers else None
        #     assert (output.size(0) == 1)
        #
        #     output, coverage, buffer = layer.step(output, pos_emb, context, mask_tgt, mask_src, buffer=buffer)
        #
        #     decoder_state.update_attention_buffer(buffer, i)
        #
        # # From Google T2T
        # # if normalization is done in layer_preprocess, then it should also be done
        # # on the output, since the output can grow very large, being the sum of
        # # a whole stack of unnormalized layer outputs.
        # output = self.postprocess_layer(output)

        return output, coverage

    def create_decoder_state(self, batch, beam_size=1, type=1):
        raise NotImplementedError

    def renew_buffer(self, new_len):
        return

    def tie_weights(self):
        assert self.generator is not None, "The generator needs to be created before sharing weights"
        self.generator[0].linear.weight = self.tgt_word_lut.weight

    def share_enc_dec_embedding(self):
        self.src_word_lut.weight = self.tgt_word_lut.weight

    def reset_states(self):
        return

