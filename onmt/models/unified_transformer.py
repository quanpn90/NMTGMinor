import torch
import torch.nn as nn
from onmt.models.transformer_layers import PositionalEncoding, PrePostProcessing
from onmt.models.transformers import TransformerEncoder, TransformerDecoder, TransformerDecodingState
import onmt
from onmt.modules.dropout import embedded_dropout
from onmt.models.transformer_layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing
from onmt.models.universal_transformer_layers import UniversalEncoderLayer, UniversalDecoderLayer
# from onmt.models.relative_transformer_layers import RelativeTransformerEncoderLayer, RelativeTransformerDecoderLayer
from onmt.utils import flip, expected_length
from collections import defaultdict
import math

torch.set_printoptions(profile="full")


class UnifiedTransformer(TransformerDecoder):
    """
    This class combines the encoder and the decoder into one single sequence
    Joined attention between encoder and decoder parts
    """

    def __init__(self, opt, src_embedding, tgt_embedding, generator, positional_encoder,
                 language_embeddings=None, encoder_type='text', **kwargs):
        self.death_rate = opt.death_rate
        self.bidirectional = opt.bidirectional
        self.layer_modules = []

        # build_modules will be called from the inherited constructor
        super(UnifiedTransformer, self).__init__(opt, tgt_embedding,
                                                 positional_encoder,
                                                 language_embeddings=language_embeddings,
                                                 allocate_positions=True)
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        # self.language_embedding = nn.Embedding(3, self.model_size, padding_idx=0)
        self.generator = generator
        self.ignore_source = True
        self.encoder_type = opt.encoder_type

        # self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)
        self.d_head = self.model_size // self.n_heads

        # self.build_modules()

    def gen_mask(self, src, tgt):

        input_seq = torch.cat([src, tgt], dim=-1)
        seq_len = input_seq.size(1)

        if self.bidirectional:
            bsz, src_len = src.size(0), src.size(1)
            tgt_len = tgt.size(1)

            tgt_tgt_mask = torch.triu(src.new_ones(tgt_len, tgt_len), diagonal=1)
            tgt_src_mask = src.new_zeros(tgt_len, src_len)

            tgt_mask = torch.cat([tgt_src_mask, tgt_tgt_mask], dim=-1)

            src_src_mask = src.new_zeros(src_len, src_len)
            src_tgt_mask = src.new_ones(src_len, tgt_len)

            src_mask = torch.cat([src_src_mask, src_tgt_mask], dim=-1)

            attn_mask = torch.cat([src_mask, tgt_mask], dim=0)

            attn_mask = attn_mask.bool()

            pad_mask = input_seq.eq(onmt.constants.PAD).unsqueeze(1)

            attn_mask = attn_mask | pad_mask
            # attn_mask = attn_mask.byte() + input_seq.eq(onmt.constants.PAD).byte().unsqueeze(1)
            # print(attn_mask[0])
            # attn_mask = torch.gt(attn_mask, 0).bool()

        else:
            attn_mask = self.mask[:seq_len, :seq_len] + input_seq.eq(onmt.constants.PAD).byte().unsqueeze(1)
            attn_mask = torch.gt(attn_mask, 0).bool()

        return attn_mask

    def build_modules(self):

        e_length = expected_length(self.layers, self.death_rate)

        print("* Transformer Decoder with Absolute Attention with %.2f expected layers" % e_length)

        self.layer_modules = nn.ModuleList()

        for l in range(self.layers):
            # linearly decay the death rate
            death_r = (l + 1.0) / self.layers * self.death_rate

            block = DecoderLayer(opt, death_rate=death_r)
            self.layer_modules.append(block)

    def forward(self, batch, target_mask=None, **kwargs):

        src = batch.get('source').transpose(0, 1)  # src_len x batch_size -> bsz x src_len
        tgt = batch.get('target_input').transpose(0, 1)  # len_tgt x batch_size -> bsz x tgt_len
        src_pos = batch.get('source_pos')
        tgt_pos = batch.get('target_pos')
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')

        tgt_len = tgt.size(1)
        src_len = src.size(1)
        bsz = tgt.size(0)

        # Embedding stage (and scale the embedding)
        src_emb = embedded_dropout(self.src_embedding, src, dropout=self.word_dropout if self.training else 0) \
                  * math.sqrt(self.model_size)
        tgt_emb = embedded_dropout(self.tgt_embedding, tgt, dropout=self.word_dropout if self.training else 0) \
                  * math.sqrt(self.model_size)

        # Add position encoding
        src_emb = self.time_transformer(src_emb)
        tgt_emb = self.time_transformer(tgt_emb)

        if self.use_language_embedding:
            if self.language_embedding_type in ["sum", "all_sum"]:
                src_lang_emb = self.language_embeddings(src_lang)
                src_emb += src_lang_emb.unsqueeze(1)
                tgt_lang_emb = self.language_embeddings(tgt_lang)
                tgt_emb += tgt_lang_emb.unsqueeze(1)

        # concatenate embedding
        emb = torch.cat([src_emb, tgt_emb], dim=1)  # L x batch_size x H

        # prepare self-attention mask
        # For the source: we have two different parts
        # [1 x src_len x batch_size]
        # mask_src_src = src.eq(onmt.constants.PAD).unsqueeze(0).byte()
        # src_pad_mask = mask_src_src
        # # Attention from src to target: everything is padded
        # mask_src_tgt = mask_src_src.new_ones(1, 1, 1).expand(src_len, tgt_len, bsz)
        # # [src_len x L x batch_size]
        # mask_src = torch.cat([mask_src_src.expand(src_len, src_len, bsz), mask_src_tgt], dim=1)
        # mask_src = mask_src.bool()
        # mask_src_src = src.eq(onmt.constants.PAD).unsqueeze(1).byte()  # B x 1 x src_len
        # mask_src_tgt = mask_src_src.new_ones(bsz, src_len, tgt_len)  # bsz x src_len x tgt_len
        #
        # mask_src = torch.cat([mask_src_src.expand(bsz, src_len, src_len), mask_src_tgt], dim=-1)
        #
        # # For the target:
        # mask_tgt_tgt = tgt.eq(onmt.constants.PAD).byte().unsqueeze(1) + self.mask[:tgt_len, :tgt_len]
        # mask_tgt_tgt = torch.gt(mask_tgt_tgt, 0).byte()  # bsz x tgt_len x tgt_len
        #
        # mask_tgt_src = mask_tgt_tgt.new_zeros(bsz, tgt_len, src_len) + src.eq(onmt.constants.PAD).unsqueeze(1).byte()
        # mask_tgt = torch.cat([mask_tgt_src, mask_tgt_tgt], dim=-1)  # bsz x tgt_len x T
        #
        # attn_mask = torch.cat([mask_src, mask_tgt], dim=1).bool()     # L x L x batch_size

        # lets try to use language modeling style
        # input_seq = torch.cat([src, tgt], dim=-1)
        # seq_len = input_seq.size(1)
        #
        # attn_mask = self.mask[:seq_len, :seq_len] + input_seq.eq(onmt.constants.PAD).byte().unsqueeze(1)
        # attn_mask = torch.gt(attn_mask, 0).bool()
        attn_mask = self.gen_mask(src, tgt)

        output = emb

        # Applying dropout and tranpose to T x B x H
        output = self.preprocess_layer(output).transpose(0, 1)

        # FORWARD PASS
        coverage = None
        for i, layer in enumerate(self.layer_modules):
            output, coverage = layer(output, None, attn_mask, None)  # context and context_mask are None

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

    def encode(self, input, decoder_state, input_pos=None, input_lang=None):

        buffers = decoder_state.attention_buffers
        src_lang = input_lang
        # Embedding stage (and scale the embedding)
        src_emb = embedded_dropout(self.src_embedding, input, dropout=self.word_dropout if self.training else 0) \
                  * math.sqrt(self.model_size)

        # Add position encoding
        src_emb = self.time_transformer(src_emb)

        if self.use_language_embedding:
            if self.language_embedding_type in ["sum", "all_sum"]:
                src_lang_emb = self.language_embeddings(src_lang)
                src_emb += src_lang_emb.unsqueeze(1)

        emb = src_emb
        src_len = input.size(1)
        bsz = input.size(0)
        mask_src_src = input.eq(onmt.constants.PAD).unsqueeze(1).byte()  # B x 1 x src_len

        mask_src = mask_src_src

        attn_mask = mask_src.bool()  # L x L x batch_size

        output = emb

        # Applying dropout and tranpose to T x B x H
        output = self.preprocess_layer(output).transpose(0, 1)

        # FORWARD PASS
        coverage = None
        for i, layer in enumerate(self.layer_modules):
            # context and context_mask are None
            buffer = buffers[i] if i in buffers else None
            output, coverage, buffer = layer.step(output, None, attn_mask, None, buffer)
            decoder_state.update_attention_buffer(buffer, i)

        # Final normalization
        output = self.postprocess_layer(output)

        return output

    def decode(self, batch):
        """
        :param batch: (onmt.Dataset.Batch) an object containing tensors needed for training
        :return: gold_scores (torch.Tensor) log probs for each sentence
                 gold_words  (Int) the total number of non-padded tokens
                 allgold_scores (list of Tensors) log probs for each word in the sentence
        """
        # raise NotImplementedError
        tgt_output = batch.get('target_output')
        output_dict = self.forward(batch, target_mask=None)
        context = output_dict['context']
        logprobs = output_dict['logprobs']

        batch_size = logprobs.size(1)

        gold_scores = context.new(batch_size).zero_()
        gold_words = 0
        allgold_scores = list()

        for gen_t, tgt_t in zip(logprobs, tgt_output):
            tgt_t = tgt_t.unsqueeze(1)
            scores = gen_t.gather(1, tgt_t)
            scores.masked_fill_(tgt_t.eq(onmt.constants.PAD), 0)
            gold_scores += scores.squeeze(1).type_as(gold_scores)
            gold_words += tgt_t.ne(onmt.constants.PAD).sum().item()
            allgold_scores.append(scores.squeeze(1).type_as(gold_scores))

        return gold_words, gold_scores, allgold_scores

    def renew_buffer(self, new_len):

        # This model uses pre-allocated position encoding
        self.positional_encoder.renew(new_len)
        mask = torch.ByteTensor(np.triu(np.ones((new_len + 1, new_len + 1)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)

        return

    def reset_states(self):
        return

    def step(self, input, decoder_state):

        src = decoder_state.src.transpose(0, 1) if decoder_state.src is not None else None
        tgt = input
        tgt_lang = decoder_state.tgt_lang
        src_lang = decoder_state.src_lang
        # print(src.size(), tgt.size())
        # print(src_lang, tgt_lang)

        tgt_len = tgt.size(1)
        src_len = src.size(1)
        bsz = tgt.size(0)

        # Embedding stage (and scale the embedding)
        src_emb = embedded_dropout(self.src_embedding, src, dropout=self.word_dropout if self.training else 0) \
                  * math.sqrt(self.model_size)
        tgt_emb = embedded_dropout(self.tgt_embedding, tgt, dropout=self.word_dropout if self.training else 0) \
                  * math.sqrt(self.model_size)

        # Add position encoding
        src_emb = self.time_transformer(src_emb)
        tgt_emb = self.time_transformer(tgt_emb)

        if self.use_language_embedding:
            if self.language_embedding_type in ["sum", "all_sum"]:
                src_lang_emb = self.language_embeddings(src_lang)
                src_emb += src_lang_emb.unsqueeze(1)
                tgt_lang_emb = self.language_embeddings(tgt_lang)
                tgt_emb += tgt_lang_emb.unsqueeze(1)

        # concatenate embedding
        emb = torch.cat([src_emb, tgt_emb], dim=1)  # L x batch_size x H

        # prepare self-attention mask
        # For the source: we have two different parts
        # [1 x src_len x batch_size]
        # mask_src_src = src.eq(onmt.constants.PAD).unsqueeze(0).byte()
        # src_pad_mask = mask_src_src
        # # Attention from src to target: everything is padded
        # mask_src_tgt = mask_src_src.new_ones(1, 1, 1).expand(src_len, tgt_len, bsz)
        # # [src_len x L x batch_size]
        # mask_src = torch.cat([mask_src_src.expand(src_len, src_len, bsz), mask_src_tgt], dim=1)
        # mask_src = mask_src.bool()
        # mask_src_src = src.eq(onmt.constants.PAD).unsqueeze(1).byte()  # B x 1 x src_len
        # mask_src_tgt = mask_src_src.new_ones(bsz, src_len, tgt_len)  # bsz x src_len x tgt_len
        #
        # mask_src = torch.cat([mask_src_src.expand(bsz, src_len, src_len), mask_src_tgt], dim=-1)
        #
        # # For the target:
        # mask_tgt_tgt = tgt.eq(onmt.constants.PAD).byte().unsqueeze(1) + self.mask[:tgt_len, :tgt_len]
        # mask_tgt_tgt = torch.gt(mask_tgt_tgt, 0).byte()  # bsz x tgt_len x tgt_len
        #
        # mask_tgt_src = mask_tgt_tgt.new_zeros(bsz, tgt_len, src_len) + src.eq(onmt.constants.PAD).unsqueeze(1).byte()
        # mask_tgt = torch.cat([mask_tgt_src, mask_tgt_tgt], dim=-1)  # bsz x tgt_len x T
        # attn_mask = torch.cat([mask_src, mask_tgt], dim=1).bool()  # L x L x batch_size

        attn_mask = self.gen_mask(src, input)
        # seq = torch.cat([src, input], dim=-1)
        # seq_len = seq.size(1)
        # attn_mask = self.mask[:seq_len, :seq_len] + seq.eq(onmt.constants.PAD).byte().unsqueeze(1)
        # attn_mask = torch.gt(attn_mask, 0).bool()

        output = emb

        # Applying dropout and tranpose to T x B x H
        output = self.preprocess_layer(output).transpose(0, 1)

        # FORWARD PASS
        coverage = None
        for i, layer in enumerate(self.layer_modules):
            output, coverage = layer(output, None, attn_mask, None)  # context and context_mask are None

        # Final normalization
        output = self.postprocess_layer(output)

        output = output[-1:, :, :]

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = output

        logprobs = self.generator[0](output_dict).squeeze(0)

        output_dict['src'] = decoder_state.src.transpose(0, 1)
        output_dict['log_prob'] = logprobs
        output_dict['coverage'] = logprobs.new(bsz, tgt_len, src_len).zero_()
        # buffers = decoder_state.attention_buffers
        # tgt_lang = decoder_state.tgt_lang
        # src = decoder_state.src.transpose(0, 1) if decoder_state.src is not None else None
        #
        # if decoder_state.concat_input_seq:
        #     if decoder_state.input_seq is None:
        #         decoder_state.input_seq = input
        #     else:
        #         # concatenate the last input to the previous input sequence
        #         decoder_state.input_seq = torch.cat([decoder_state.input_seq, input], 0)
        #
        #     # For Transformer, both inputs are assumed as B x T (batch first)
        #     input = decoder_state.input_seq.transpose(0, 1)
        #     src = decoder_state.src.transpose(0, 1) if decoder_state.src is not None else None
        #
        # if input.size(1) > 1:
        #     input_ = input[:, -1].unsqueeze(1)
        # else:
        #     input_ = input
        # """ Embedding: batch_size x 1 x d_model """
        # # check = input_.gt(self.word_lut.num_embeddings)
        # print(input.size())
        # emb = self.tgt_embedding(input_) * math.sqrt(self.model_size)
        #
        # """ Adding positional encoding """
        # emb = self.time_transformer(emb, t=input.size(1))
        #
        # if self.use_language_embedding:
        #     if self.language_embedding_type in ["sum", "all_sum"]:
        #
        #         tgt_lang_emb = self.language_embeddings(tgt_lang)
        #         emb += tgt_lang_emb.unsqueeze(1)
        #
        # emb = emb.transpose(0, 1)
        #
        # # attention mask For the target:
        # tgt_len = input.size(1)
        # bsz = input.size(0)
        # src_len = src.size(1)
        # mask_tgt_tgt = input.eq(onmt.constants.PAD).byte().unsqueeze(1) + self.mask[:tgt_len, :tgt_len]
        # mask_tgt_tgt = torch.gt(mask_tgt_tgt, 0).byte()  # bsz x tgt_len x tgt_len
        #
        # mask_tgt_src = mask_tgt_tgt.new_zeros(bsz, tgt_len, src_len) + src.eq(onmt.constants.PAD).unsqueeze(1).byte()
        #
        # mask_tgt = torch.cat([mask_tgt_src, mask_tgt_tgt], dim=-1)  # bsz x tgt_len x T
        #
        # # take the last element of the 'target sequence' for the mask
        # attn_mask = mask_tgt[:, -1, :].unsqueeze(1).bool()
        #
        # output = emb
        #
        # for i, layer in enumerate(self.layer_modules):
        #     buffer = buffers[i] if i in buffers else None
        #     assert (output.size(0) == 1)
        #
        #     output, coverage, buffer = layer.step(output, None, attn_mask, None, buffer=buffer)
        #
        #     decoder_state.update_attention_buffer(buffer, i)
        #
        # # Final normalization

        # output_dict = defaultdict(lambda: None)
        # output_dict['hidden'] = output
        #
        # logprobs = self.generator[0](output_dict).squeeze(0)
        #
        # output_dict['src'] = decoder_state.src.transpose(0, 1)
        # output_dict['log_prob'] = logprobs
        # output_dict['coverage'] = logprobs.new(bsz, tgt_len, src_len).zero_()

        return output_dict

    def create_decoder_state(self, batch, beam_size=1, type=1):

        src = batch.get('source')
        src_pos = batch.get('source_pos')
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')

        src_transposed = src.transpose(0, 1)  # B x T

        decoder_state = TransformerDecodingState(src, tgt_lang, None, None,
                                                 beam_size=beam_size, model_size=self.model_size, type=type)

        # forward pass through the input to get the buffer
        # _ = self.encode(src_transposed, decoder_state, input_pos=src_pos, input_lang=src_lang)

        decoder_state.src_lang = src_lang


        # buffers = decoder_state.attention_buffers
        # bsz = src.size(1)
        # new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        # new_order = new_order.to(src.device)
        #
        # for l in buffers:
        #     buffer_ = buffers[l]
        #     if buffer_ is not None:
        #         for k in buffer_.keys():
        #             t_, br_, d_ = buffer_[k].size()
        #             buffer_[k] = buffer_[k].index_select(1, new_order)  # 1 for time first

        return decoder_state

    def tie_weights(self):
        assert self.generator is not None, "The generator needs to be created before sharing weights"
        self.generator[0].linear.weight = self.tgt_embedding.weight

    def share_enc_dec_embedding(self):
        self.src_embedding.weight = self.tgt_embedding.weight