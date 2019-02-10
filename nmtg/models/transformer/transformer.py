import math

from torch import nn
import torch.utils.checkpoint

from nmtg.models import register_model
from nmtg.models.encoder_decoder import Encoder, IncrementalDecoder, EncoderDecoderModel
from nmtg.modules.positional_encoding import SinusoidalPositionalEncoding, LearnedPositionalEncoding, \
    RNNPositionalEncoding
from nmtg.modules.transformer_layers import PrePostProcessing, TransformerEncoderLayer, TransformerDecoderLayer


@register_model('transformer')
class Transformer(EncoderDecoderModel):
    def __init__(self, *, model_dim=512, num_heads=8, layers=6, feed_forward_dim=2048,
                 feed_forward_dropout=0.1, attn_dropout=0.1, residual_dropout=0.1, embedding_dropout=0.1,
                 weight_norm=False, masked_layers=False, future_masking=True, gated_residuals=False, batch_first=False,
                 feed_forward_type='linear_relu_linear', positional_encoding=None,
                 ignore_context=False, share_encoder_decoder=False, checkpointing=0):
        self.batch_first = batch_first

        encoder = TransformerEncoder(
            model_dim=model_dim,
            num_heads=num_heads,
            layers=layers,
            feed_forward_dim=feed_forward_dim,
            feed_forward_dropout=feed_forward_dropout,
            attn_dropout=attn_dropout,
            residual_dropout=residual_dropout,
            embedding_dropout=embedding_dropout,
            weight_norm=weight_norm,
            gated_residuals=gated_residuals,
            masked_layers=masked_layers,
            batch_first=batch_first,
            feed_forward_type=feed_forward_type,
            positional_encoding=positional_encoding,
            checkpointing=checkpointing,
        )

        decoder = TransformerDecoder(
            model_dim=model_dim,
            num_heads=num_heads,
            layers=layers,
            feed_forward_dim=feed_forward_dim,
            feed_forward_dropout=feed_forward_dropout,
            attn_dropout=attn_dropout,
            residual_dropout=residual_dropout,
            embedding_dropout=embedding_dropout,
            weight_norm=weight_norm,
            gated_residuals=gated_residuals,
            future_masking=future_masking,
            masked_layers=masked_layers,
            batch_first=batch_first,
            feed_forward_type=feed_forward_type,
            positional_encoding=positional_encoding,
            checkpointing=checkpointing,
            ignore_context=ignore_context,
            encoder_to_share=encoder if share_encoder_decoder else None
        )

        super().__init__(encoder, decoder)

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
        parser.add_argument('-residual_type', default='regular', choices=['default', 'gated'],
                            help='Type of residual type')
        parser.add_argument('-batch_first', action='store_true',
                            help='Use a batch first model')
        parser.add_argument('-no_future_masking', action='store_true',
                            help='Do not perform future masking on the decoder attention. '
                                 'This will prevent incremental decoding (i.e. beam search)!')
        parser.add_argument('-ignore_context', action='store_true',
                            help='Ignore the output of the encoder when decoding. Experimental')

        parser.add_argument('-death_rate', type=float, default=0.5,
                            help='Stochastic layer death rate')
        parser.add_argument('-death_type', type=str, default='linear_decay',
                            help='Stochastic layer death type: linear decay or uniform')

        parser.set_defaults(optimizer='adam', update_method='noam')

    @classmethod
    def build_model(cls, args):
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

        return cls(
            model_dim=args.model_size,
            num_heads=args.n_heads,
            layers=args.layers,
            feed_forward_dim=args.inner_size,
            feed_forward_dropout=args.dropout,
            attn_dropout=args.attn_dropout,
            residual_dropout=args.residual_dropout,
            embedding_dropout=args.emb_dropout,
            weight_norm=args.weight_norm,
            gated_residuals=args.residual_type == 'gated',
            future_masking=not args.no_future_masking,
            batch_first=args.batch_first,
            feed_forward_type=args.activation_layer,
            positional_encoding=positional_encoding,
            ignore_context=args.ignore_context,
            checkpointing=args.checkpointing,
            share_encoder_decoder=args.share_enc_dec_weights
        )

    @staticmethod
    def convert_state_dict(opt, state_dict):
        res = super().convert_state_dict(opt, state_dict)
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
            return {'layer_1': ffn_dict['fc_1'], 'layer_2': ffn_dict['fc_2']}

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
                'preprocess_attn': layer_in['preprocess_attn'],
                'preprocess_ffn': layer_in['preprocess_ffn'],
                'attention_tgt': {
                    'query_projection': {'function': layer_in['multihead_tgt']['fc_query']['function']['linear']},
                    'key_projection': {'function': layer_in['multihead_tgt']['fc_key']['function']['linear']},
                    'value_projection': {'function': layer_in['multihead_tgt']['fc_value']['function']['linear']},
                    'out_projection': {'function': layer_in['multihead_tgt']['fc_concat']['function']['linear']}
                },
                'feed_forward': {'function': convert_ffn(layer_in['feedforward']['function'])}
            }
            if not opt.ignore_context:
                layer_dict['preprocess_src_attn'] = layer_in['preprocess_src_attn']
                layer_dict['attention_src'] = {
                    'query_projection': {'function': layer_in['multihead_src']['fc_query']['function']['linear']},
                    'key_projection': {'function': layer_in['multihead_src']['fc_key']['function']['linear']},
                    'value_projection': {'function': layer_in['multihead_src']['fc_value']['function']['linear']},
                    'out_projection': {'function': layer_in['multihead_src']['fc_concat']['function']['linear']}
                }
            res['decoder']['layers'][str(i)] = layer_dict
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
        self.batch_first = batch_first
        self.checkpointing = checkpointing
        self.masked_layers = masked_layers

        self.positional_encoding = positional_encoding

        self.preprocess = PrePostProcessing(model_dim, 'd', embedding_dropout)
        self.postprocess = PrePostProcessing(model_dim, 'n')

        self.layers = nn.ModuleList([TransformerEncoderLayer(
            model_dim=model_dim,
            num_heads=num_heads,
            feed_forward_dim=feed_forward_dim,
            feed_forward_dropout=feed_forward_dropout,
            attention_dropout=attn_dropout,
            residual_dropout=residual_dropout,
            weight_norm=weight_norm,
            masked_layers=masked_layers,
            gated_residuals=gated_residuals,
            batch_first=batch_first,
            feed_forward_type=feed_forward_type
        ) for _ in range(layers)])

    def forward(self, inputs, input_mask=None):
        inputs *= math.sqrt(self.model_dim)

        if self.positional_encoding is not None:
            inputs = self.positional_encoding(inputs)

        inputs = self.preprocess(inputs)

        if input_mask is not None:
            self_attention_mask = input_mask if self.batch_first else input_mask.transpose(0, 1)
            self_attention_bias = inputs.new_full(self_attention_mask.size(), float('-inf'))\
                .masked_fill_(self_attention_mask, 0).unsqueeze(1)
        else:
            self_attention_bias = None

        for i, layer in enumerate(self.layers):
            if self.checkpointing > 0 and self.training and (i + 1) % self.checkpointing == 0:
                inputs = torch.utils.checkpoint.checkpoint(layer, inputs, input_mask, self_attention_bias)
            else:
                inputs = layer(inputs, input_mask, self_attention_bias)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        outputs = self.postprocess(inputs, mask=input_mask if self.masked_layers else None)

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
        self.batch_first = batch_first
        self.checkpointing = checkpointing
        self.masked_layers = masked_layers
        self.ignore_context = ignore_context
        self.positional_encoding = encoder_to_share.positional_encoding \
            if encoder_to_share is not None else positional_encoding

        self.preprocess = PrePostProcessing(model_dim, 'd', embedding_dropout,
                                            gated_residuals=gated_residuals)
        self.postprocess = PrePostProcessing(model_dim, 'n')

        self.layers = nn.ModuleList([TransformerDecoderLayer(
            model_dim=model_dim,
            num_heads=num_heads,
            feed_forward_dim=feed_forward_dim,
            feed_forward_dropout=feed_forward_dropout,
            attention_dropout=attn_dropout,
            residual_dropout=residual_dropout,
            weight_norm=weight_norm,
            masked_layers=masked_layers,
            gated_residuals=gated_residuals,
            batch_first=batch_first,
            feed_forward_type=feed_forward_type,
            ignore_context=ignore_context,
            encoder_to_share=encoder_to_share.layers[i] if encoder_to_share is not None else None
        ) for i in range(layers)])

    def forward(self, decoder_inputs, encoder_outputs, input_mask=None, context_mask=None,
                incremental_state=None):
        decoder_inputs *= math.sqrt(self.model_dim)

        if self.positional_encoding is not None:
            decoder_inputs = self.positional_encoding(decoder_inputs, incremental_state=incremental_state)

        decoder_inputs = self.preprocess(decoder_inputs)

        if incremental_state is None and self.future_masking:
            len_tgt = decoder_inputs.size(1 if self.batch_first else 0)
            # future_mask: (len_tgt x len_tgt)
            future_mask = self.get_future_mask(len_tgt, decoder_inputs.device)
            if input_mask is None:
                self_attention_mask = future_mask.unsqueeze(0)
            else:
                # padding_mask: (batch_size x len_tgt)
                padding_mask = input_mask if self.batch_first else input_mask.transpose(0, 1)
                self_attention_mask = (padding_mask.unsqueeze(1) + future_mask).gt_(1)
        elif input_mask is not None:
            self_attention_mask = input_mask if self.batch_first else input_mask.transpose(0, 1)
            self_attention_mask = self_attention_mask.unsqueeze(1)
        else:
            self_attention_mask = None

        if self_attention_mask is not None:
            # self_attention_mask: (batch_size x len_tgt x len_tgt) or broadcastable
            self_attention_bias = decoder_inputs.new_full(self_attention_mask.size(), float('-inf'))\
                .masked_fill_(self_attention_mask, 0)
        else:
            self_attention_bias = None

        if context_mask is not None:
            # enc_padding_mask: (batch_size x len_src) or broadcastable
            enc_padding_mask = context_mask if self.batch_first else context_mask.transpose(0, 1)
            encoder_attention_bias = encoder_outputs.new_full(enc_padding_mask.size(), float('-inf'))\
                .masked_fill(enc_padding_mask, 0).unsqueeze(1)
        else:
            encoder_attention_bias = None

        for i, layer in enumerate(self.layers):
            if self.checkpointing > 0 and self.training and (i + 1) % self.checkpointing == 0:
                decoder_inputs = torch.utils.checkpoint.checkpoint(layer, decoder_inputs, encoder_outputs, input_mask, context_mask,
                                                                   self_attention_bias, encoder_attention_bias,
                                                                   incremental_state)
            else:
                decoder_inputs = layer(decoder_inputs, encoder_outputs, input_mask, context_mask,
                                       self_attention_bias, encoder_attention_bias,
                                       incremental_state)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        outputs = self.postprocess(decoder_inputs, mask=input_mask if self.masked_layers else None)

        return outputs
