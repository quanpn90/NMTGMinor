import torch
from torch import nn

import nmtg.data.data_utils
from nmtg.data import Dictionary
from nmtg.models.encoder_decoder import Encoder, IncrementalDecoder, EncoderDecoderModel
from nmtg.modules.dropout import EmbeddingDropout


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

    def forward(self, decoder_inputs, encoder_outputs, input_mask=None,
                encoder_mask=None, optimized=False):
        emb = self.embedded_dropout(decoder_inputs)
        out = self.decoder(emb, encoder_outputs, input_mask, encoder_mask)

        if optimized and input_mask is not None:
            # Optimize the projection by calculating only those position where
            # the input was not padding
            out = out.view(-1, out.size(-1))
            out = out.index_select(0, torch.nonzero(input_mask.view(-1)).squeeze(1))
        return self.linear(out)

    def _step(self, decoder_inputs, encoder_outputs, incremental_state, input_mask=None, encoder_mask=None):
        emb = self.embedded_dropout(decoder_inputs)
        out = self.decoder.step(emb, encoder_outputs, incremental_state, input_mask, encoder_mask)
        return self.linear(out)

    def reorder_incremental_state(self, incremental_state, new_order):
        self.decoder.reorder_incremental_state(incremental_state, new_order)

    def set_beam_size(self, beam_size):
        self.decoder.set_beam_size(beam_size)


class NMTModel(EncoderDecoderModel):
    def __init__(self, encoder, decoder,
                 src_dict=None, tgt_dict=None, batch_first=False):
        super().__init__(encoder, decoder)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.batch_first = batch_first

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

        if hasattr(args, 'model_size'):
            output_size = args.model_size
        else:
            dummy_input = torch.zeros(1, 1, embedding_size)
            dummy_output = model(dummy_input, torch.tensor([[1]], dtype=torch.uint8))
            output_size = dummy_output.size(-1)

        src_embedding = cls.build_embedding(args, src_dict, embedding_size, path=args.pre_word_vecs_enc)

        if args.join_embedding:
            if src_dict is not tgt_dict:
                raise ValueError('Cannot join embeddings, vocabularies are not the same')

            tgt_embedding = src_embedding
        else:
            tgt_embedding = cls.build_embedding(args, tgt_dict, embedding_size, path=args.pre_word_vecs_dec)

        tgt_linear = nn.Linear(output_size, len(tgt_dict))

        if args.tie_weights:
            tgt_linear.weight = tgt_embedding.weight
        else:
            nn.init.xavier_uniform_(tgt_linear.weight)

        encoder = NMTEncoder(model.encoder, src_embedding, args.word_dropout)
        decoder = NMTDecoder(model.decoder, tgt_embedding, args.word_dropout, tgt_linear)

        return cls(encoder, decoder, src_dict, tgt_dict, batch_first)

    @staticmethod
    def build_embedding(args, dictionary: Dictionary, embedding_size, path=None):
        emb = nn.Embedding(len(dictionary), embedding_size, padding_idx=dictionary.pad())
        if path is not None:
            embed_dict = nmtg.data.data_utils.parse_embedding(path)
            nmtg.data.data_utils.load_embedding(embed_dict, dictionary, emb)
        elif args.init_embedding == 'xavier':
            nn.init.xavier_uniform_(emb.weight)
        elif args.init_embedding == 'normal':
            nn.init.normal_(emb.weight, mean=0, std=embedding_size ** -0.5)
        else:
            raise ValueError('Unknown initialization {}'.format(args.init_embedding))

        if args.freeze_embeddings:
            emb.weight.requires_grad_(False)

        return emb

    def forward(self, encoder_inputs, decoder_inputs, encoder_mask=None, decoder_mask=None, optimized_decoding=False):
        if encoder_mask is None and self.src_dict is not None:
            encoder_mask = encoder_inputs.ne(self.src_dict.pad())
        if decoder_mask is None and self.tgt_dict is not None:
            decoder_mask = decoder_inputs.ne(self.tgt_dict.pad())

        encoder_out = self.encoder(encoder_inputs, encoder_mask)
        decoder_out = self.decoder(decoder_inputs, encoder_out, decoder_mask, encoder_mask,
                                   optimized=optimized_decoding)
        return decoder_out

    @staticmethod
    def convert_state_dict(opt, state_dict):
        # No call to super here, because wrapping a model in NMTModel is new
        model_cls = nmtg.models.get_model_type(opt.model)
        res = {
            'encoder': {
                'embedded_dropout': {'embedding': {'weight': state_dict['encoder']['word_lut']['weight']}},
            },
            'decoder': {
                'embedded_dropout': {'embedding': {'weight': state_dict['decoder']['word_lut']['weight']}},
                'linear': {'weight': state_dict['generator']['linear']['weight'],
                           'bias': state_dict['generator']['linear']['bias']}
            }
        }
        model_state_dict = model_cls.convert_state_dict(state_dict)
        res['encoder']['encoder'] = model_state_dict['encoder']
        res['decoder']['decoder'] = model_state_dict['decoder']
        return res
