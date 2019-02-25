import math

from torch import nn
import torch.utils.checkpoint

from nmtg.models import register_model
from nmtg.models.encoder_decoder import Encoder, IncrementalDecoder, EncoderDecoderModel, IncrementalModule
from nmtg.modules.attention import MultiHeadAttention
from nmtg.modules.masking import MaskedFunction
from nmtg.modules.positional_encoding import SinusoidalPositionalEncoding, LearnedPositionalEncoding, \
    RNNPositionalEncoding
from nmtg.modules.transformer_layers import PrePostProcessing, get_feed_forward


@register_model('transformer')
class Transformer(EncoderDecoderModel):
    def __init__(self, *, model_dim=512, num_heads=8, layers=6, feed_forward_dim=2048,
                 feed_forward_dropout=0.1, attn_dropout=0.1, residual_dropout=0.1, embedding_dropout=0.1,
                 weight_norm=False, masked_layers=False, future_masking=True, gated_residuals=False, batch_first=False,
                 feed_forward_type='linear_relu_linear', positional_encoding=None,
                 ignore_context=False, share_encoder_decoder=False, checkpointing=0):
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.layers = layers
        self.feed_forward_dim = feed_forward_dim
        self.feed_forward_dropout = feed_forward_dropout
        self.attn_dropout = attn_dropout
        self.residual_dropout = residual_dropout
        self.embedding_dropout = embedding_dropout
        self.weight_norm = weight_norm
        self.masked_layers = masked_layers
        self.future_masking = future_masking
        self.gated_residuals = gated_residuals
        self.feed_forward_type = feed_forward_type
        self.ignore_context = ignore_context
        self.share_encoder_decoder = share_encoder_decoder
        self.checkpointing = checkpointing
        self.batch_first = batch_first

        encoder = self.get_encoder(positional_encoding)
        decoder = self.get_decoder(positional_encoding, encoder)
        super().__init__(encoder, decoder)

    def get_encoder(self, positional_encoding):
        encoder = TransformerEncoder(
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            layers=self.layers,
            feed_forward_dim=self.feed_forward_dim,
            feed_forward_dropout=self.feed_forward_dropout,
            attn_dropout=self.attn_dropout,
            residual_dropout=self.residual_dropout,
            embedding_dropout=self.embedding_dropout,
            weight_norm=self.weight_norm,
            gated_residuals=self.gated_residuals,
            masked_layers=self.masked_layers,
            batch_first=self.batch_first,
            feed_forward_type=self.feed_forward_type,
            positional_encoding=positional_encoding,
            checkpointing=self.checkpointing,
        )
        return encoder

    def get_decoder(self, positional_encoding, encoder):
        decoder = TransformerDecoder(
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            layers=self.layers,
            feed_forward_dim=self.feed_forward_dim,
            feed_forward_dropout=self.feed_forward_dropout,
            attn_dropout=self.attn_dropout,
            residual_dropout=self.residual_dropout,
            embedding_dropout=self.embedding_dropout,
            weight_norm=self.weight_norm,
            gated_residuals=self.gated_residuals,
            future_masking=self.future_masking,
            masked_layers=self.masked_layers,
            batch_first=self.batch_first,
            feed_forward_type=self.feed_forward_type,
            positional_encoding=positional_encoding,
            checkpointing=self.checkpointing,
            ignore_context=self.ignore_context,
            encoder_to_share=encoder if self.share_encoder_decoder else None
        )
        return decoder

    @staticmethod
    def add_options(parser):
        EncoderDecoderModel.add_options(parser)
        parser.add_argument('-layers', type=int, default=6,
                            help='Number of layers in the encoder/decoder')
        parser.add_argument('-model_size', type=int, default=512,
                            help='Size of embedding / transformer hidden state')
        parser.add_argument('-inner_size', type=int, default=2048,
                            help='Size of inner feed forward layer')
        parser.add_argument('-n_heads', type=int, default=8,
                            help='Number of heads for multi-head attention')
        parser.add_argument('-checkpointing', type=int, default=0,
                            help='Create a gradient checkpoint every N layers. '
                                 'Experimental')
        parser.add_argument('-mask_layers', action='store_true',
                            help='Use Masked Functions to reduce operations on certain layers. '
                                 'Only effective if you expect to have a lot of padding in your input. '
                                 'Experimental')
        parser.add_argument('-dropout', type=float, default=0.1,
                            help='Feed-forward dropout probability.')
        parser.add_argument('-attn_dropout', type=float, default=0.1,
                            help='Dropout probability; applied on multi-head attention.')
        parser.add_argument('-emb_dropout', type=float, default=0.1,
                            help='Dropout probability; applied on top of embedding.')
        parser.add_argument('-residual_dropout', type=float, default=0.2,
                            help='Dropout probability; applied on residual connection.')
        parser.add_argument('-weight_norm', action='store_true',
                            help='Apply weight normalization on linear modules')
        parser.add_argument('-activation_layer', default='linear_relu_linear', type=str,
                            help='The activation layer in each transformer block')
        parser.add_argument('-time', default='positional_encoding', type=str,
                            choices=['positional_encoding', 'learned', 'gru', 'lstm', 'none'],
                            help='Type of time representation. ')
        parser.add_argument('-max_position_length', type=int, default=1024,
                            help='Maximum length for positional embedding')
        parser.add_argument('-residual_type', default='default', choices=['default', 'gated'],
                            help='Type of residual type')
        parser.add_argument('-batch_first', action='store_true',
                            help='Use a batch first model')
        parser.add_argument('-no_future_masking', action='store_true',
                            help='Do not perform future masking on the decoder attention. '
                                 'This will prevent incremental decoding (i.e. beam search)!')
        parser.add_argument('-ignore_context', action='store_true',
                            help='Ignore the output of the encoder when decoding. Experimental')

        parser.set_defaults(optimizer='adam', update_method='noam')

    @staticmethod
    def map_options(args):
        opts = EncoderDecoderModel.map_options(args)

        if args.time == 'positional_encoding':
            positional_encoding = SinusoidalPositionalEncoding(
                args.model_size, args.batch_first, args.max_position_length)
        elif args.time == 'learned':
            positional_encoding = LearnedPositionalEncoding(
                args.model_dim, args.max_position_length, args.batch_first)
        elif args.time == 'gru':
            positional_encoding = RNNPositionalEncoding(
                nn.GRU(args.model_size, args.model_size, 1, batch_first=args.batch_first))
        elif args.time == 'lstm':
            positional_encoding = RNNPositionalEncoding(
                nn.LSTM(args.model_size, args.model_size, 1, batch_first=args.batch_first))
        elif args.time == 'none':
            positional_encoding = None
        else:
            raise ValueError('Unrecognized positional encoding type "{}"'.format(args.time))

        opts.model_dim = args.model_size
        opts.num_heads = args.n_heads
        opts.layers = args.layers
        opts.feed_forward_dim = args.inner_size
        opts.feed_forward_dropout = args.dropout
        opts.attn_dropout = args.attn_dropout
        opts.residual_dropout = args.residual_dropout
        opts.embedding_dropout = args.emb_dropout
        opts.weight_norm = args.weight_norm
        opts.gated_residuals = args.residual_type == 'gated'
        opts.masked_layers = args.mask_layers
        opts.future_masking = not args.no_future_masking
        opts.batch_first = args.batch_first
        opts.feed_forward_type = args.activation_layer
        opts.positional_encoding = positional_encoding
        opts.ignore_context = args.ignore_context
        opts.checkpointing = args.checkpointing
        opts.share_encoder_decoder = args.share_enc_dec_weights

        return opts

    @staticmethod
    def convert_state_dict(opt, state_dict):
        res = EncoderDecoderModel.convert_state_dict(opt, state_dict)
        res['encoder'] = {}
        if opt.time in ('lstm', 'gru'):
            res['encoder']['positional_encoding'] = {'rnn': state_dict['encoder']['time_transformer']}
            res['decoder']['positional_encoding'] = {'rnn': state_dict['decoder']['time_transformer']}
        else:
            res['encoder']['positional_encoding'] = state_dict['encoder']['time_transformer']
            res['decoder']['positional_encoding'] = state_dict['decoder']['time_transformer']
        res['encoder']['postprocess'] = state_dict['encoder']['postprocess_layer']
        res['decoder']['postprocess'] = state_dict['decoder']['postprocess_layer']

        def convert_linear_relu_linear(ffn_dict):
            return {'layer_1': ffn_dict['fc_1']['linear'], 'layer_2': ffn_dict['fc_2']['linear']}

        def convert_maxout(ffn_dict):
            return {'linear': ffn_dict['lin']}

        convert_ffn = convert_linear_relu_linear if opt.activation_layer == 'linear_relu_linear' else convert_maxout

        res['encoder']['layers'] = {}
        res['decoder']['layers'] = {}
        for i in range(opt.layers):
            layer_in = state_dict['encoder']['layer_modules'][str(i)]
            layer_dict = {
                'preprocess_attn': layer_in['preprocess_attn'],
                'preprocess_ffn': layer_in['preprocess_ffn'],
                'attention': {
                    'query_projection': {'function': layer_in['multihead']['fc_query']['function']['linear']},
                    'key_projection': {'function': layer_in['multihead']['fc_key']['function']['linear']},
                    'value_projection': {'function': layer_in['multihead']['fc_value']['function']['linear']},
                    'out_projection': {'function': layer_in['multihead']['fc_concat']['function']['linear']}
                },
                'feed_forward': {'function': convert_ffn(layer_in['feedforward']['function'])}
            }
            res['encoder']['layers'][str(i)] = layer_dict

            layer_in = state_dict['decoder']['layer_modules'][str(i)]
            layer_dict = {
                'preprocess_self_attn': layer_in['preprocess_attn'],
                'preprocess_ffn': layer_in['preprocess_ffn'],
                'attention_tgt': {
                    'query_projection': {'function': layer_in['multihead_tgt']['fc_query']['function']['linear']},
                    'key_projection': {'function': layer_in['multihead_tgt']['fc_key']['function']['linear']},
                    'value_projection': {'function': layer_in['multihead_tgt']['fc_value']['function']['linear']},
                    'out_projection': {'function': layer_in['multihead_tgt']['fc_concat']['function']['linear']}
                },
                'feed_forward': {'function': convert_ffn(layer_in['feedforward']['function'])},
                'preprocess_enc_attn': layer_in['preprocess_src_attn'],
                'attention_src': {
                    'query_projection': {'function': layer_in['multihead_src']['fc_query']['function']['linear']},
                    'key_projection': {'function': layer_in['multihead_src']['fc_key']['function']['linear']},
                    'value_projection': {'function': layer_in['multihead_src']['fc_value']['function']['linear']},
                    'out_projection': {'function': layer_in['multihead_src']['fc_concat']['function']['linear']}
                }
            }
            res['decoder']['layers'][str(i)] = layer_dict

        opt.batch_first = False
        opt.ignore_context = False
        opt.freeze_embeddings = False

        return res


class TransformerEncoder(Encoder):
    """
    A stack of TransformerEncoderLayers

    Layers:
        Scale up by sqrt(model_size)
        Positional encoding
        Dropout
        TransformerEncoderLayer x layers
        Layer norm

    Args:
        All from TransformerEncoderLayer
        layers:              Number of layers
        positional_encoding: Instance of PositionalEncoding, optional

    Input Shapes:
        inputs:     batch_size x len_query x model_dim  or  len_query x batch_size x model_dim
        input_mask:  batch_size x len_query  or  len_query x batch_size (or broadcastable)

    Output Shapes:
        out: batch_size x len_query x model_dim  or  len_query x batch_size x model_dim

    """

    def __init__(self, *, model_dim=512, num_heads=8, layers=6, feed_forward_dim=2048,
                 feed_forward_dropout=0.1, attn_dropout=0.1, residual_dropout=0.1, embedding_dropout=0.1,
                 weight_norm=False, masked_layers=False, gated_residuals=False, batch_first=False,
                 feed_forward_type='linear_relu_linear', positional_encoding=None,
                 checkpointing=0):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = layers
        self.feed_forward_dim = feed_forward_dim
        self.feed_forward_dropout = feed_forward_dropout
        self.feed_forward_type = feed_forward_type
        self.attn_dropout = attn_dropout
        self.residual_dropout = residual_dropout
        self.embedding_dropout = embedding_dropout
        self.weight_norm = weight_norm
        self.masked_layers = masked_layers
        self.gated_residuals = gated_residuals
        self.positional_encoding = positional_encoding
        self.batch_first = batch_first
        self.checkpointing = checkpointing

        self.preprocess = PrePostProcessing(model_dim, 'd', embedding_dropout)
        self.postprocess = PrePostProcessing(model_dim, 'n', masking=self.masked_layers)

        self.build_layers()

    # noinspection PyAttributeOutsideInit
    def build_layers(self):
        self.layers = nn.ModuleList([TransformerEncoderLayer(
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            feed_forward_dim=self.feed_forward_dim,
            feed_forward_dropout=self.feed_forward_dropout,
            attention_dropout=self.attn_dropout,
            residual_dropout=self.residual_dropout,
            weight_norm=self.weight_norm,
            masked_layers=self.masked_layers,
            gated_residuals=self.gated_residuals,
            batch_first=self.batch_first,
            feed_forward_type=self.feed_forward_type
        ) for _ in range(self.num_layers)])

    def forward(self, inputs, input_mask=None):
        positions = self.positional_encoding(inputs) if self.positional_encoding is not None else None

        inputs *= math.sqrt(self.model_dim)

        if positions is not None:
            inputs += positions

        inputs = self.preprocess(inputs)
        self_attention_bias = self.get_self_attention_bias(inputs, self.batch_first, input_mask)

        for i, layer in enumerate(self.layers):
            if self.checkpointing > 0 and self.training and (i + 1) % self.checkpointing == 0:
                inputs = torch.utils.checkpoint.checkpoint(layer, inputs, input_mask, self_attention_bias)
            else:
                inputs = layer(inputs, input_mask, self_attention_bias)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        outputs = self.postprocess(inputs, mask=input_mask)

        return outputs


class TransformerDecoder(IncrementalDecoder):
    """
    A stack of TransformerDecoderLayers

    Layers:
        Scale up by sqrt(model_size)
        Positional encoding
        Dropout
        TransformerDecoderLayer x layers
        Layer norm

    Args:
        All from TransformerDecoderLayer
        layers:            Number of layers
        encoder_to_share:  Share each layer with this TransformerEncoder
        positional_encoding: Instance of PositionalEncoding, optional

    Input Shapes:
        inputs:       len_query x batch_size x model_dim  or  batch_size x len_query x model_dim
        input_mask:   batch_size x len_query  or  len_query x batch_size
        context:      len_context x batch_size x model_dim  or  batch_size x len_context x model_dim
        context_mask: batch_size x len_context  or  len_context x batch_size
    """

    def __init__(self, *, model_dim=512, num_heads=8, layers=6, feed_forward_dim=2048,
                 feed_forward_dropout=0.1, attn_dropout=0.1, residual_dropout=0.1, embedding_dropout=0.1,
                 weight_norm=False, gated_residuals=False, masked_layers=False, future_masking=True, batch_first=False,
                 feed_forward_type='linear_relu_linear', positional_encoding=None,
                 ignore_context=False, checkpointing=0, encoder_to_share=None):
        super().__init__(future_masking, encoder_to_share)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = layers
        self.feed_forward_dim = feed_forward_dim
        self.feed_forward_dropout = feed_forward_dropout
        self.feed_forward_type = feed_forward_type
        self.attn_dropout = attn_dropout
        self.residual_dropout = residual_dropout
        self.embedding_dropout = embedding_dropout
        self.weight_norm = weight_norm
        self.masked_layers = masked_layers
        self.gated_residuals = gated_residuals
        self.positional_encoding = positional_encoding
        self.batch_first = batch_first
        self.checkpointing = checkpointing
        self.ignore_context = ignore_context

        self.preprocess = PrePostProcessing(model_dim, 'd', embedding_dropout,
                                            gated_residuals=gated_residuals)
        self.postprocess = PrePostProcessing(model_dim, 'n', masking=self.masked_layers)

        if encoder_to_share is not None:
            self.positional_encoding = encoder_to_share.positional_encoding

        self.build_layers(encoder_to_share)

    # noinspection PyAttributeOutsideInit
    def build_layers(self, encoder=None):
        self.layers = nn.ModuleList([TransformerDecoderLayer(
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            feed_forward_dim=self.feed_forward_dim,
            feed_forward_dropout=self.feed_forward_dropout,
            attention_dropout=self.attn_dropout,
            residual_dropout=self.residual_dropout,
            weight_norm=self.weight_norm,
            masked_layers=self.masked_layers,
            gated_residuals=self.gated_residuals,
            batch_first=self.batch_first,
            feed_forward_type=self.feed_forward_type,
            ignore_context=self.ignore_context,
            encoder_to_share=encoder.layers[i] if encoder is not None else None
        ) for i in range(self.num_layers)])

    def forward(self, decoder_inputs, encoder_outputs, input_mask=None, encoder_mask=None):
        if self.positional_encoding is not None:
            positions = self.positional_encoding(decoder_inputs)
        else:
            positions = None

        decoder_inputs *= math.sqrt(self.model_dim)

        if positions is not None:
            decoder_inputs += positions

        decoder_inputs = self.preprocess(decoder_inputs)
        self_attention_bias = self.get_self_attention_bias(decoder_inputs, self.batch_first, input_mask)
        encoder_attention_bias = self.get_encoder_attention_bias(encoder_outputs, self.batch_first, encoder_mask)

        for i, layer in enumerate(self.layers):
            if self.checkpointing > 0 and self.training and (i + 1) % self.checkpointing == 0:
                decoder_inputs = torch.utils.checkpoint.checkpoint(
                    layer, decoder_inputs, encoder_outputs, input_mask,
                    encoder_mask, self_attention_bias, encoder_attention_bias)
            else:
                decoder_inputs = layer(decoder_inputs, encoder_outputs, input_mask,
                                       encoder_mask, self_attention_bias, encoder_attention_bias)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        outputs = self.postprocess(decoder_inputs, mask=input_mask)

        return outputs

    def _step(self, decoder_inputs, encoder_outputs, incremental_state, input_mask=None, encoder_mask=None):
        if self.positional_encoding is not None:
            positions = self.positional_encoding.step(decoder_inputs, incremental_state)
        else:
            positions = None

        decoder_inputs *= math.sqrt(self.model_dim)

        if positions is not None:
            decoder_inputs += positions

        decoder_inputs = self.preprocess(decoder_inputs)
        self_attention_bias = self.get_self_attention_bias(decoder_inputs, self.batch_first, input_mask,
                                                           future_masking=False)
        encoder_attention_bias = self.get_encoder_attention_bias(encoder_outputs, self.batch_first, encoder_mask)

        for i, layer in enumerate(self.layers):
            decoder_inputs = layer.step(decoder_inputs, encoder_outputs, incremental_state,
                                        input_mask, encoder_mask,
                                        self_attention_bias, encoder_attention_bias)

        outputs = self.postprocess(decoder_inputs, mask=input_mask)

        return outputs


class TransformerEncoderLayer(nn.Module):
    """
    Wraps multi-head attentions and position-wise feed forward into one encoder layer.

    Layers:
        (1)
         Layer norm
         Multi-head self-attention
         Dropout
         Residual with (1)
         (2)
         Layer norm
         Feed-forward
         Dropout
         Residual with (2)

    Feed-Forward:
        Configurable between linear -> ReLU -> linear and Maxout

    Args:
        model_dim:            dimension of model
        num_heads:            number of heads
        feed_forward_dim:     dimension of feed forward
        feed_forward_dropout: dropout probability in the feed forward
        attention_dropout:    dropout probability in attention
        residual_dropout:     dropout probability for the residual layers
        weight_norm:          whether to use weight normalization on the feed forward layers
        masked_layers:        whether to use masking for layer norm and feed forward. Useful for sparse masks
        gated_residuals:      whether to use gated residuals
        batch_first:          whether input (and output) should be batch dimension first or sequence
                              length dimension first
        feed_forward_type:    Which type of feed forward to use. Currently supports 'linear_relu_linear'
                              and 'maxout'

    Params:
        attention:    multi-head self-attentions layer
        feed_forward:  feed forward layer

    Input Shapes:
        inputs:         batch_size x len_query x model_dim  or  len_query x batch_size x model_dim
        input_mask:     batch_size x len_query  or  len_query x batch_size (or broadcastable)
        attention_bias: batch_size x len_query x len_query or broadcastable, regardless of batch_first

    Output Shapes:
        out: batch_size x len_query x model_dim  or  len_query x batch_size x model_dim
    """

    def __init__(self, *, model_dim=512, num_heads=8, feed_forward_dim=2048,
                 feed_forward_dropout=0.1, attention_dropout=0.1, residual_dropout=0.1,
                 weight_norm=False, masked_layers=False, gated_residuals=False, batch_first=False,
                 feed_forward_type='linear_relu_linear'):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.feed_forward_dropout = feed_forward_dropout
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.weight_norm = weight_norm
        self.masked_layers = masked_layers
        self.gated_residuals = gated_residuals
        self.batch_first = batch_first
        self.feed_forward_type = feed_forward_type

        self.preprocess_attn = PrePostProcessing(model_dim, 'n', masking=masked_layers)
        self.postprocess_attn = PrePostProcessing(model_dim, 'da', residual_dropout,
                                                  gated_residuals=gated_residuals)
        self.preprocess_ffn = PrePostProcessing(model_dim, 'n', masking=masked_layers)
        self.postprocess_ffn = PrePostProcessing(model_dim, 'da', residual_dropout,
                                                 gated_residuals=gated_residuals)
        self.build_self_attention()

        self.build_feed_forward()

    def get_preprocessing_module(self):
        return PrePostProcessing(self.model_dim, 'n', masking=self.masked_layers)

    def get_postprocessing_module(self):
        return PrePostProcessing(self.model_dim, 'da', self.residual_dropout, gated_residuals=self.gated_residuals)

    # noinspection PyAttributeOutsideInit
    def build_self_attention(self):
        self.preprocess_attn = self.get_preprocessing_module()
        self.attention = MultiHeadAttention(
            self.model_dim, self.num_heads, self.attention_dropout,
            masked_layers=self.masked_layers, batch_first=self.batch_first)
        self.postprocess_attn = self.get_postprocessing_module()

    def self_attention_layer(self, inputs, input_mask=None, self_attention_bias=None):
        query = self.preprocess_attn(inputs, mask=input_mask)
        self_attention_out = self.attention(query, query, query, self_attention_bias, input_mask)
        self_attention_out = self.postprocess_attn(self_attention_out, inputs)
        return self_attention_out

    # noinspection PyAttributeOutsideInit
    def build_feed_forward(self):
        self.preprocess_ffn = self.get_preprocessing_module()
        self.feed_forward = MaskedFunction(
            get_feed_forward(
                self.feed_forward_type,
                self.model_dim,
                self.feed_forward_dim,
                self.feed_forward_dropout,
                self.weight_norm))
        self.postprocess_ffn = self.get_postprocessing_module()

    def feed_forward_layer(self, attention_out, input_mask=None):
        out = self.preprocess_ffn(attention_out, mask=input_mask)
        out = self.feed_forward(out, mask=input_mask if self.masked_layers else None)
        out = self.postprocess_ffn(out, attention_out)
        return out

    def forward(self, inputs, input_mask=None, attention_bias=None):
        attention_out = self.self_attention_layer(inputs, input_mask, attention_bias)

        out = self.feed_forward_layer(attention_out, input_mask)
        return out


class TransformerDecoderLayer(IncrementalModule):
    """
    Wraps multi-head self-attention, encoder-decoder attention and position-wise
    feed forward into one layer of decoder

    Layers:
        (1)
         Layer norm
         Multi-head self-attention
         Dropout
         Residual with (1)
         (2)
         Layer norm
         Multi-head query-context attention
         Dropout
         Residual with (2)
         (3)
         Layer norm
         Feed-forward
         Dropout
         Residual with (3)

    Feed-Forward:
        Configurable between linear -> ReLU -> linear and Maxout

    Args:
        model_dim:            dimension of model
        num_heads:            number of heads
        feed_forward_dim:     dimension of feed forward
        feed_forward_dropout: dropout probability in the feed forward
        attention_dropout:    dropout probability in attention
        residual_dropout:     dropout probability for the residual layers
        weight_norm:          whether to use weight normalization on the feed forward layers
        masked_layers:        whether to use masking for layer norm and feed forward. Useful for sparse masks
        gated_residuals:      whether to use gated residuals
        batch_first:          whether input (and output) should be batch dimension first or sequence
                              length dimension first
        feed_forward_type:    Which type of feed forward to use. Currently supports 'linear_relu_linear'
                              and 'maxout'
        ignore_context:       If True, do not use the context input at all
        encoder_to_share:     Instance of TransformerEncoderLayer to share parameters with

    Input Shapes:
        inputs:              len_query x batch_size x model_dim  or  batch_size x len_query x model_dim
        context:             len_context x batch_size x model_dim  or  batch_size x len_context x model_dim
        input_mask:          batch_size x len_query  or  len_query x batch_size
        context_mask:        batch_size x len_context  or  len_context x batch_size
        self_attention_mask: batch_size x len_query x len_query or broadcastable, regardless of batch_first

    Output Shapes:
        out:      len_query x batch_size x model_dim  or  len_query x batch_size x model_dim
    """

    def __init__(self, *, model_dim=512, num_heads=8, feed_forward_dim=2048,
                 feed_forward_dropout=0.1, attention_dropout=0.1, residual_dropout=0.1,
                 weight_norm=False, masked_layers=False, gated_residuals=False, batch_first=False,
                 feed_forward_type='linear_relu_linear',
                 ignore_context=False, encoder_to_share=None):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.feed_forward_dropout = feed_forward_dropout
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.weight_norm = weight_norm
        self.masked_layers = masked_layers
        self.gated_residuals = gated_residuals
        self.batch_first = batch_first
        self.feed_forward_type = feed_forward_type
        self.ignore_context = ignore_context

        if encoder_to_share is None:
            self.build_self_attention()
            self.build_feed_forward()
        else:
            # share the self-attention layers between encoder and decoder
            self.share_feed_forward(encoder_to_share)
            self.share_self_attention(encoder_to_share)

        if not ignore_context:
            self.build_encoder_attention()

    def get_preprocessing_module(self):
        return PrePostProcessing(self.model_dim, 'n', masking=self.masked_layers)

    def get_postprocessing_module(self):
        return PrePostProcessing(self.model_dim, 'da', self.residual_dropout, gated_residuals=self.gated_residuals)

    # noinspection PyAttributeOutsideInit
    def build_self_attention(self):
        self.preprocess_attn = self.get_preprocessing_module()
        self.attention_tgt = MultiHeadAttention(
            self.model_dim, self.num_heads, self.attention_dropout,
            masked_layers=self.masked_layers, batch_first=self.batch_first)
        self.postprocess_attn = self.get_postprocessing_module()

    # noinspection PyAttributeOutsideInit
    def share_self_attention(self, encoder):
        self.preprocess_attn = encoder.preprocess_attn
        self.postprocess_attn = encoder.postprocess_attn
        self.attention_tgt = encoder.attention

    def self_attention_layer(self, inputs, input_mask=None, self_attention_bias=None):
        query = self.preprocess_attn(inputs, mask=input_mask)
        self_attention_out = self.attention_tgt(query, query, query, self_attention_bias, input_mask)
        self_attention_out = self.postprocess_attn(self_attention_out, inputs)
        return self_attention_out

    def self_attention_step(self, inputs, incremental_state, input_mask=None, self_attention_bias=None):
        query = self.preprocess_attn(inputs, mask=input_mask)
        self_attention_out = self.attention_tgt.step(query, query, query, incremental_state,
                                                     self_attention_bias, input_mask)
        self_attention_out = self.postprocess_attn(self_attention_out, inputs)
        return self_attention_out

    # noinspection PyAttributeOutsideInit
    def build_encoder_attention(self):
        self.preprocess_src_attn = self.get_preprocessing_module()
        self.attention_src = MultiHeadAttention(
            self.model_dim, self.num_heads, self.attention_dropout,
            masked_layers=self.masked_layers, batch_first=self.batch_first)
        self.postprocess_src_attn = self.get_postprocessing_module()

    def encoder_attention_layer(self, inputs, encoder_outputs, input_mask=None,
                                context_mask=None, encoder_attention_bias=None):
        query = self.preprocess_src_attn(inputs, mask=input_mask)
        enc_attention_out = self.attention_src(query, encoder_outputs, encoder_outputs,
                                               encoder_attention_bias, input_mask, context_mask)
        enc_attention_out = self.postprocess_src_attn(enc_attention_out, inputs)
        return enc_attention_out

    def encoder_attention_step(self, inputs, encoder_outputs, incremental_state, input_mask=None,
                               context_mask=None, encoder_attention_bias=None):
        query = self.preprocess_src_attn(inputs, mask=input_mask)
        enc_attention_out = self.attention_src.step(query, encoder_outputs, encoder_outputs, incremental_state,
                                                    encoder_attention_bias, input_mask, context_mask,
                                                    static_kv=True)
        enc_attention_out = self.postprocess_src_attn(enc_attention_out, inputs)
        return enc_attention_out

    # noinspection PyAttributeOutsideInit
    def build_feed_forward(self):
        self.preprocess_ffn = self.get_preprocessing_module()
        self.feed_forward = MaskedFunction(
            get_feed_forward(
                self.feed_forward_type,
                self.model_dim,
                self.feed_forward_dim,
                self.feed_forward_dropout,
                self.weight_norm))
        self.postprocess_ffn = self.get_postprocessing_module()

    # noinspection PyAttributeOutsideInit
    def share_feed_forward(self, encoder):
        self.preprocess_ffn = encoder.preprocess_ffn
        self.postprocess_ffn = encoder.postprocess_ffn
        self.feed_forward = encoder.feed_forward

    def feed_forward_layer(self, inputs, input_mask=None):
        out = self.preprocess_ffn(inputs, mask=input_mask)
        out = self.feed_forward(out, mask=input_mask if self.masked_layers else None)
        out = self.postprocess_ffn(out, inputs)
        return out

    def feed_forward_step(self, inputs, input_mask):
        return self.feed_forward_layer(inputs, input_mask)

    def forward(self, inputs, context, input_mask=None, context_mask=None, self_attention_bias=None,
                encoder_attention_bias=None):
        self_attention_out = self.self_attention_layer(inputs, input_mask, self_attention_bias)

        if not self.ignore_context:
            context_attention_out = self.encoder_attention_layer(
                self_attention_out, context, input_mask, context_mask, encoder_attention_bias)
        else:
            context_attention_out = self_attention_out

        out = self.feed_forward_layer(context_attention_out, input_mask)
        return out

    def _step(self, inputs, encoder_outputs, incremental_state, input_mask=None, context_mask=None,
              self_attention_bias=None,
              encoder_attention_bias=None):
        self_attention_out = self.self_attention_step(inputs, incremental_state, input_mask, self_attention_bias)

        if not self.ignore_context:
            enc_attention_out = self.encoder_attention_step(self_attention_out, encoder_outputs, incremental_state,
                                                            input_mask, context_mask, encoder_attention_bias)
        else:
            enc_attention_out = self_attention_out

        out = self.feed_forward_step(enc_attention_out, input_mask)
        return out
