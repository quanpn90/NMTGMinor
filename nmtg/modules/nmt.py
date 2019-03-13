import itertools
import logging

import torch
from torch import nn

from nmtg.models.encoder_decoder import Encoder, IncrementalDecoder
from nmtg.modules.attention import MultiHeadAttention
from nmtg.modules.dropout import EmbeddingDropout
from nmtg.modules.linear import XavierLinear

logger = logging.getLogger(__name__)


class NMTEncoder(Encoder):
    """Wraps an Encoder and adds embedding"""

    def __init__(self, encoder, embedding, dropout):
        super().__init__()
        self.encoder = encoder
        self.embedded_dropout = EmbeddingDropout(embedding, dropout)

    def forward(self, encoder_inputs, input_mask=None):
        emb = self.embedded_dropout(encoder_inputs)
        return self.encoder(emb, input_mask)


class NMTDecoder(IncrementalDecoder):
    """Wraps a Decoder and adds embedding and projection"""

    def __init__(self, decoder, embedding, dropout, linear, *, copy_decoder=False, batch_first=False,
                 extra_attention=False, masked_layers=False, attention_dropout=0.1, language_embedding=None):
        super().__init__()
        self.decoder = decoder
        self.embedded_dropout = EmbeddingDropout(embedding, dropout)
        self.linear = linear
        self.copy_decoder = copy_decoder
        self.batch_first = batch_first
        self.extra_attention = extra_attention

        if self.copy_decoder:
            model_dim = linear.weight.size(1)
            self.gate_layer = XavierLinear(model_dim, 1)
            if extra_attention:
                self.attention = MultiHeadAttention(model_dim, 1, attention_dropout, batch_first, masked_layers)

            self._register_load_state_dict_pre_hook(self._load_nmt_model_compatibility)

        if language_embedding is not None:
            self.language_embedding = language_embedding
            model_dim = self.embedded_dropout.embedding.weight.size(1)
            emb_dim = language_embedding.weight.size(1)
            self.merge_layer = XavierLinear(model_dim + emb_dim, model_dim)
        else:
            self.language_embedding = None

    def forward(self, decoder_inputs, encoder_outputs, decoder_mask=None, encoder_mask=None):
        if self.language_embedding is not None:
            indices, language_id = decoder_inputs

            emb = torch.cat((self.embedded_dropout(indices), self.language_embedding(language_id)), dim=-1)
            emb = self.merge_layer(emb)
        else:
            emb = self.embedded_dropout(decoder_inputs)

        out, attention_weights = self.decoder(emb, encoder_outputs, decoder_mask, encoder_mask)

        if self.copy_decoder:
            if self.extra_attention:
                source_attention_bias = self.get_encoder_attention_bias(encoder_outputs, self.batch_first, encoder_mask)
                _, attention_weights = self.attention(out, encoder_outputs, encoder_outputs,
                                                      source_attention_bias, decoder_mask, encoder_mask)

            gates = torch.sigmoid(self.gate_layer(out)).squeeze(-1)

        if self.training and decoder_mask is not None:
            # Optimize the projection by calculating only those position where
            # the input was not padding
            nonpad_indices = torch.nonzero(decoder_mask.view(-1)).squeeze(1)
            out = out.view(-1, out.size(-1))
            out = out.index_select(0, nonpad_indices)

            # For multihead attention, the batch size dimension will be bigger. That means the results
            # of this operation are garbage
            if attention_weights is not None:
                attention_weights = attention_weights.view(-1, attention_weights.size(-1))
                attention_weights = attention_weights.index_select(0, nonpad_indices)
            if self.copy_decoder:
                gates = gates.masked_select(decoder_mask)

        if self.copy_decoder:
            attention_weights = {'attn': attention_weights, 'gates': gates}

        return self.linear(out), attention_weights

    def _step(self, decoder_inputs, encoder_outputs, incremental_state, decoder_mask=None, encoder_mask=None):
        emb = self.embedded_dropout(decoder_inputs)
        out, attention_weights = self.decoder.step(emb, encoder_outputs, incremental_state, decoder_mask, encoder_mask)

        if self.copy_decoder:
            if self.extra_attention:
                source_attention_bias = self.get_encoder_attention_bias(encoder_outputs, self.batch_first, encoder_mask)
                _, attention_weights = self.attention(out, encoder_outputs, encoder_outputs,
                                                      source_attention_bias, decoder_mask, encoder_mask)

            gates = torch.sigmoid(self.gate_layer(out)).squeeze(-1)
            attention_weights = {'attn': attention_weights, 'gates': gates}

        return self.linear(out), attention_weights

    def get_normalized_probs(self, decoder_outputs, attention_weights, encoder_inputs=None, encoder_mask=None,
                             decoder_mask=None, log_probs=False):
        decoder_probs = self.decoder.get_normalized_probs(decoder_outputs, attention_weights, encoder_inputs,
                                                          encoder_mask, decoder_mask, log_probs)

        if not self.copy_decoder:
            return decoder_probs

        attention_weights, gates = attention_weights['attn'], attention_weights['gates']
        gates = gates.unsqueeze(-1)

        optimized = decoder_outputs.dim() == 2
        if not self.batch_first:
            encoder_inputs = encoder_inputs.transpose(0, 1).unsqueeze(0)  # (1, batch, src_len)
        if optimized:
            # (batch, tgt_len, src_len) | (tgt_len, batch, src_len)
            new_size = list(decoder_mask.size()) + [encoder_inputs.size(-1)]
            nonpad_indices = torch.nonzero(decoder_mask.view(-1)).squeeze(1)
            encoder_inputs = encoder_inputs.expand(new_size).contiguous() \
                .view(-1, encoder_inputs.size(-1)) \
                .index_select(0, nonpad_indices)
            # encoder_inputs is now (decoder_outputs.size(0), src_len)
        else:
            encoder_inputs = encoder_inputs.expand_as(attention_weights)

        assert encoder_inputs.size() == attention_weights.size()

        encoder_probs = decoder_probs.new_full(decoder_probs.size(), 1e-20)
        encoder_probs.scatter_add_(1 if optimized else 2, encoder_inputs, attention_weights)

        if log_probs:
            encoder_probs.log_()
            encoder_probs.add_(torch.log(gates))
            decoder_probs.add_(torch.log(1 - gates))
            # Very important to have it this way around, otherwise we will add -inf + inf = NaN
            res = decoder_probs + torch.log1p(torch.exp(encoder_probs - decoder_probs))
            return res
        else:
            return gates * encoder_probs + (1 - gates) * decoder_probs

    def reorder_incremental_state(self, incremental_state, new_order):
        self.decoder.reorder_incremental_state(incremental_state, new_order)
        if self.extra_attention:
            self.attention.reorder_incremental_state(incremental_state, new_order)

    def _load_nmt_model_compatibility(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs):
        if prefix + 'gate_layer.weight' in state_dict:
            return

        logger.info('Augmenting NMTModel with a copy decoder')
        items = self.gate_layer.state_dict(prefix=prefix + 'gate_layer.').items()
        if self.extra_attention:
            items = itertools.chain(items, self.attention.state_dict(prefix=prefix + 'attention.').items())
        for key, value in items:
            assert key not in state_dict
            state_dict[key] = value
