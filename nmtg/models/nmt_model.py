import itertools
import logging

import torch
from torch import nn

import nmtg.data.data_utils
from nmtg.data import Dictionary
from nmtg.models import Model
from nmtg.models.encoder_decoder import Encoder, IncrementalDecoder, EncoderDecoderModel
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

    def __init__(self, decoder, embedding, dropout, linear):
        super().__init__()
        self.decoder = decoder
        self.embedded_dropout = EmbeddingDropout(embedding, dropout)
        self.linear = linear

    def forward(self, decoder_inputs, encoder_outputs, input_mask=None, encoder_mask=None, optimized=False):
        emb = self.embedded_dropout(decoder_inputs)
        out, attention_weights = self.decoder(emb, encoder_outputs, input_mask, encoder_mask)

        if optimized and input_mask is not None:
            # Optimize the projection by calculating only those position where
            # the input was not padding
            nonpad_indices = torch.nonzero(input_mask.view(-1)).squeeze(1)
            out = out.view(-1, out.size(-1))
            out = out.index_select(0, nonpad_indices)

            # For multihead attention, the batch size dimension will be bigger. That means the results
            # of this operation are garbage
            if attention_weights is not None:
                attention_weights = attention_weights.view(-1, attention_weights.size(-1))
                attention_weights = attention_weights.index_select(0, nonpad_indices)

        return self.linear(out), attention_weights

    def _step(self, decoder_inputs, encoder_outputs, incremental_state, input_mask=None, encoder_mask=None):
        emb = self.embedded_dropout(decoder_inputs)
        out, attention_weights = self.decoder.step(emb, encoder_outputs, incremental_state, input_mask, encoder_mask)

        return self.linear(out), attention_weights

    def reorder_incremental_state(self, incremental_state, new_order):
        self.decoder.reorder_incremental_state(incremental_state, new_order)


class NMTCopyDecoder(NMTDecoder):
    def __init__(self, decoder, embedding, dropout, linear, batch_first=False, extra_attention=False,
                 masked_layers=False, attention_dropout=0.1):
        super().__init__(decoder, embedding, dropout, linear)
        model_dim = linear.weight.size(1)
        self.batch_first = batch_first
        self.gate_layer = XavierLinear(model_dim, 1)
        self.extra_attention = extra_attention
        if extra_attention:
            self.attention = MultiHeadAttention(model_dim, 1, attention_dropout, batch_first, masked_layers)

        self._register_load_state_dict_pre_hook(self._load_nmt_model_compatibility)

    def forward(self, decoder_inputs, encoder_outputs, input_mask=None, encoder_mask=None, optimized=False):
        emb = self.embedded_dropout(decoder_inputs)
        # out: (batch, tgt_len, model_dim) | (tgt_len, batch, model_dim)
        # attn_weights: (batch, tgt_len, src_len) | (tgt_len, batch, src_len)
        out, attention_weights = self.decoder(emb, encoder_outputs, input_mask, encoder_mask)

        if self.extra_attention:
            source_attention_bias = self.get_encoder_attention_bias(encoder_outputs, self.batch_first, encoder_mask)
            _, attention_weights = self.attention(out, encoder_outputs, encoder_outputs,
                                                  source_attention_bias, input_mask, encoder_mask)

        gates = torch.sigmoid(self.gate_layer(out)).squeeze(-1)

        if optimized and input_mask is not None:
            # Optimize the projection by calculating only those position where
            # the input was not padding
            nonpad_indices = torch.nonzero(input_mask.view(-1)).squeeze(1)
            out = out.view(-1, out.size(-1))
            out = out.index_select(0, nonpad_indices)

            # For multihead attention, the batch size dimension will be bigger. That means the results
            # of this operation are garbage
            attention_weights = attention_weights.view(-1, attention_weights.size(-1))
            attention_weights = attention_weights.index_select(0, nonpad_indices)
            gates = gates.masked_select(input_mask)

        attention_weights = {'attn': attention_weights, 'gates': gates}
        return self.linear(out), attention_weights

    def _step(self, decoder_inputs, encoder_outputs, incremental_state, input_mask=None, encoder_mask=None):
        emb = self.embedded_dropout(decoder_inputs)
        out, attention_weights = self.decoder.step(emb, encoder_outputs, incremental_state, input_mask, encoder_mask)

        if self.extra_attention:
            source_attention_bias = self.get_encoder_attention_bias(encoder_outputs, self.batch_first, encoder_mask)
            _, attention_weights = self.attention.step(out, encoder_outputs, encoder_outputs, incremental_state,
                                                       source_attention_bias, input_mask, encoder_mask, static_kv=True)

        gates = torch.sigmoid(self.gate_layer(out)).squeeze(-1)
        attention_weights = {'attn': attention_weights, 'gates': gates}

        return self.linear(out), attention_weights

    def get_normalized_probs(self, decoder_outputs, attention_weights, encoder_inputs=None, encoder_mask=None,
                             decoder_mask=None, log_probs=False):
        decoder_probs = super().get_normalized_probs(decoder_outputs, attention_weights, encoder_inputs, encoder_mask,
                                                     decoder_mask, log_probs)

        attention_weights, gates = attention_weights['attn'], attention_weights['gates']
        gates = gates.unsqueeze(-1)

        optimized = decoder_outputs.dim() == 2
        if not self.batch_first:
            encoder_inputs = encoder_inputs.transpose(0, 1).unsqueeze(0)  # (1, batch, src_len)
        if optimized:
            # (batch, tgt_len, src_len) | (tgt_len, batch, src_len)
            new_size = list(decoder_mask.size()) + [encoder_inputs.size(-1)]
            nonpad_indices = torch.nonzero(decoder_mask.view(-1)).squeeze(1)
            encoder_inputs = encoder_inputs.expand(new_size).contiguous()\
                .view(-1, encoder_inputs.size(-1))\
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

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        if self.extra_attention:
            self.attention.reorder_incremental_state(incremental_state, new_order)


class NMTModel(EncoderDecoderModel):
    def __init__(self, encoder, decoder, src_dict=None, tgt_dict=None, batch_first=False, freeze_old=False):
        super().__init__(encoder, decoder)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.batch_first = batch_first

        if freeze_old:
            logger.info('Freezing model parameters')
            for param in itertools.chain(self.encoder.parameters(), self.decoder.decoder.parameters(),
                                         self.decoder.embedded_dropout.parameters(),
                                         self.decoder.linear.parameters()):
                param.requires_grad_(False)

    @staticmethod
    def add_options(parser):
        # Do not add EncoderDeocderModel parameters, because NMTModel wraps another model
        # that already added those parameters
        # EncoderDecoderModel.add_options(parser)
        parser.add_argument('-tie_weights', action='store_true',
                            help='Share weights between embedding and softmax')
        parser.add_argument('-join_embedding', action='store_true',
                            help='Share encoder and decoder embeddings')
        parser.add_argument('-freeze_embeddings', action='store_true',
                            help='Do not train word embeddings')
        parser.add_argument('-pre_word_vecs_enc', type=str,
                            help='If a valid path is specified, then this will load '
                                 'pretrained word embeddings on the encoder side. '
                                 'See README for specific formatting instructions.')
        parser.add_argument('-pre_word_vecs_dec', type=str,
                            help='If a valid path is specified, then this will load '
                                 'pretrained word embeddings on the decoder side. '
                                 'See README for specific formatting instructions.')
        parser.add_argument('-word_vec_size', type=int,
                            help='Word embedding sizes')
        parser.add_argument('-word_dropout', type=float, default=0.0,
                            help='Dropout probability; applied on embedding indices.')
        parser.add_argument('-init_embedding', default='normal', choices=['xavier', 'normal'],
                            help="How to init the embedding matrices.")
        parser.add_argument('-copy_decoder', action='store_true',
                            help='Use a decoder that will copy tokens from the input when it thinks it appropriate')
        parser.add_argument('-freeze_model', action='store_true',
                            help='Only used when upgrading an NMT Model without copy decoder.'
                                 'Freeze the model and only learn the copy decoder parameters')
        parser.add_argument('-extra_attention', action='store_true',
                            help='Add an extra attention layer at the end of the model to predict alignment for '
                                 'the copy decoder. For models like transformer, that have no clear attention '
                                 'alignment.')

    @classmethod
    def wrap_model(cls, args, model: EncoderDecoderModel, src_dict: Dictionary, tgt_dict: Dictionary, batch_first=None):
        assert isinstance(model.decoder, IncrementalDecoder)
        embedding_size = args.word_vec_size
        if embedding_size is None and hasattr(args, 'model_size'):
            embedding_size = args.model_size
        if embedding_size is None:
            raise ValueError('Could not infer embedding size')

        if hasattr(model, 'batch_first'):
            batch_first = model.batch_first
        if batch_first is None:
            raise ValueError("Could not infer whether the model is batch_first, specify manually")

        if args.copy_decoder and not args.join_vocab:
            raise NotImplementedError('In order to use the copy decoder, the source and target language must '
                                      'use the same vocabulary')

        if hasattr(args, 'model_size'):
            output_size = args.model_size
        else:
            dummy_input = torch.zeros(1, 1, embedding_size)
            dummy_output = model(dummy_input, torch.tensor([[1]], dtype=torch.uint8))
            output_size = dummy_output.size(-1)

        src_embedding = cls.build_embedding(src_dict, embedding_size, path=args.pre_word_vecs_enc,
                                            init_embedding=args.init_embedding,
                                            freeze_embedding=args.freeze_embeddings)

        if args.join_embedding:
            if src_dict is not tgt_dict:
                raise ValueError('Cannot join embeddings, vocabularies are not the same')

            tgt_embedding = src_embedding
        else:
            tgt_embedding = cls.build_embedding(tgt_dict, embedding_size, path=args.pre_word_vecs_dec,
                                                init_embedding=args.init_embedding,
                                                freeze_embedding=args.freeze_embeddings)

        tgt_linear = XavierLinear(output_size, len(tgt_dict))

        if args.tie_weights:
            tgt_linear.weight = tgt_embedding.weight

        encoder = NMTEncoder(model.encoder, src_embedding, args.word_dropout)

        if args.copy_decoder:
            masked_layers = getattr(args, 'masked_layers', False)
            attention_dropout = getattr(args, 'attn_dropout', 0.0)
            decoder = NMTCopyDecoder(model.decoder, tgt_embedding, args.word_dropout, tgt_linear,
                                     batch_first, args.extra_attention, masked_layers, attention_dropout)
        else:
            decoder = NMTDecoder(model.decoder, tgt_embedding, args.word_dropout, tgt_linear)

        return cls(encoder, decoder, src_dict, tgt_dict, batch_first, args.freeze_model)

    @staticmethod
    def build_embedding(dictionary: Dictionary, embedding_size, path=None,
                        init_embedding='xavier', freeze_embedding=False):
        emb = nn.Embedding(len(dictionary), embedding_size, padding_idx=dictionary.pad())
        if path is not None:
            embed_dict = nmtg.data.data_utils.parse_embedding(path)
            nmtg.data.data_utils.load_embedding(embed_dict, dictionary, emb)
        elif init_embedding == 'xavier':
            nn.init.xavier_uniform_(emb.weight)
        elif init_embedding == 'normal':
            nn.init.normal_(emb.weight, mean=0, std=embedding_size ** -0.5)
        else:
            raise ValueError('Unknown initialization {}'.format(init_embedding))

        if freeze_embedding:
            emb.weight.requires_grad_(False)

        return emb

    def forward(self, encoder_inputs, decoder_inputs, encoder_mask=None, decoder_mask=None,
                optimized_decoding=False):
        if encoder_mask is None and self.src_dict is not None:
            encoder_mask = encoder_inputs.ne(self.src_dict.pad())
        if decoder_mask is None and self.tgt_dict is not None:
            decoder_mask = decoder_inputs.ne(self.tgt_dict.pad())

        encoder_out = self.encoder(encoder_inputs, encoder_mask)
        decoder_out, attention_weights = self.decoder(
            decoder_inputs, encoder_out, decoder_mask, encoder_mask, optimized=optimized_decoding)
        return decoder_out, attention_weights

    @staticmethod
    def upgrade_args(args):
        if 'freeze_model' not in args:
            args.freeze_model = False
            args.copy_decoder = False
            args.extra_attention = False
