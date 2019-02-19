import math

from torch import nn

from nmtg.models import register_model
from nmtg.models.encoder_decoder import EncoderDecoderModel, IncrementalDecoder
from nmtg.models.transformer import Transformer
from nmtg.models.transformer.transformer import TransformerEncoder, TransformerDecoder
from nmtg.modules.transformer_layers import PrePostProcessing, AverageTransformerDecoderLayer


@register_model('average_transformer')
class AverageTransformer(Transformer):

    def __init__(self, *, model_dim=512, num_heads=8, layers=6, feed_forward_dim=2048, feed_forward_dropout=0.1,
                 attn_dropout=0.1, residual_dropout=0.1, embedding_dropout=0.1, weight_norm=False, masked_layers=False,
                 future_masking=True, gated_residuals=False, batch_first=False, feed_forward_type='linear_relu_linear',
                 positional_encoding=None, ignore_context=False, share_encoder_decoder=False, checkpointing=0):
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

        decoder = AverageTransformerDecoder(
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

        EncoderDecoderModel.__init__(self, encoder, decoder)


class AverageTransformerDecoder(TransformerDecoder):

    def __init__(self, *, model_dim=512, num_heads=8, layers=6, feed_forward_dim=2048, feed_forward_dropout=0.1,
                 attn_dropout=0.1, residual_dropout=0.1, embedding_dropout=0.1, weight_norm=False,
                 gated_residuals=False, masked_layers=False, future_masking=True, batch_first=False,
                 feed_forward_type='linear_relu_linear', positional_encoding=None, ignore_context=False,
                 checkpointing=0, encoder_to_share=None):
        IncrementalDecoder.__init__(self, future_masking, encoder_to_share)
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

        self.layers = nn.ModuleList([AverageTransformerDecoderLayer(
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

    def get_self_attention_bias(self, decoder_inputs, batch_first, input_mask=None, future_masking=True):
        # ignore future_masking parameter
        self_attention_mask = self.get_self_attention_mask(decoder_inputs, batch_first, input_mask, True)
        self_attention_bias = self_attention_mask.type_as(decoder_inputs)
        return self_attention_bias / self_attention_bias.sum(-1, keepdim=True)
