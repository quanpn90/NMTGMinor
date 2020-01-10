import torch
import torch.nn as nn
from onmt.models.transformer_layers import PositionalEncoding, PrePostProcessing
from onmt.models.transformer_layers import EncoderLayer, DecoderLayer
from onmt.models.tranformers import TransformerEncoder, TransformerDecoder
import onmt
from onmt.modules.dropout import embedded_dropout
from onmt.models.transformer_layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing
from onmt.models.transformer_layers import EncoderLayer, DecoderLayer
# from onmt.models.relative_transformer_layers import RelativeTransformerEncoderLayer, RelativeTransformerDecoderLayer
from onmt.utils import flip, expected_length
from collections import defaultdict
import math


class UnifiedTransformer(TransformerDecoder):
    """
    This class combines the encoder and the decoder into one single sequence
    Joined attention between encoder and decoder parts
    """

    def __init__(self, opt, src_embedding, tgt_embedding, generator, positional_encoder,
                 language_embeddings=None, encoder_type='text', **kwargs):
        self.death_rate = opt.death_rate
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

        e_length = expected_length(self.layers, self.death_rate)

        print("* Transformer Decoder with Relative Attention with %.2f expected layers" % e_length)

        self.build_modules()

    def build_modules(self):
        self.layer_modules = nn.ModuleList()

        for l in range(self.layers):
            # linearly decay the death rate
            death_r = (l + 1.0) / self.layers * self.death_rate

            block = DecoderLayer(self.n_heads, self.model_size,
                                                self.dropout, self.inner_size, self.attn_dropout,
                                                ignore_source=True,
                                                variational=self.variational_dropout, death_rate=death_r)
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
        mask_src_src = src.eq(onmt.constants.PAD).unsqueeze(1).byte()  # B x 1 x src_len
        mask_src_tgt = mask_src_src.new_ones(bsz, src_len, tgt_len)  # bsz x src_len x tgt_len

        mask_src = torch.cat([mask_src_src.expand(bsz, src_len, src_len), mask_src_tgt], dim=-1)

        # For the target:
        mask_tgt_tgt = tgt.eq(onmt.constants.PAD).byte().unsqueeze(1) + self.mask[:tgt_len, :tgt_len]
        mask_tgt_tgt = torch.eq(mask_tgt_tgt, 0).byte()  # bsz x tgt_len x tgt_len

        mask_tgt_src = mask_tgt_tgt.new_zeros(bsz, tgt_len, src_len)
        mask_tgt = torch.cat([mask_tgt_src, mask_tgt_tgt], dim=-1)  # bsz x tgt_len x T

        attn_mask = torch.cat([mask_src, mask_tgt], dim=1).bool()  # L x L x batch_size

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