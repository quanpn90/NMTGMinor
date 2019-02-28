from torch import nn

from torch import nn

from nmtg.models import register_model
from nmtg.models.transformer import Transformer
from nmtg.models.transformer.transformer import TransformerDecoder, TransformerDecoderLayer
from nmtg.modules.attention import AverageAttention
from nmtg.modules.transformer_layers import get_feed_forward


@register_model('average_transformer')
class AverageTransformer(Transformer):
    def get_decoder(self, positional_encoding, encoder):
        decoder = AverageTransformerDecoder(
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


class AverageTransformerDecoder(TransformerDecoder):

    # noinspection PyAttributeOutsideInit
    def build_layers(self, encoder=None):
        self.layers = nn.ModuleList([AverageTransformerDecoderLayer(
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
        ) for i in range(encoder)])

    def get_self_attention_bias(self, decoder_inputs, batch_first, input_mask=None, future_masking=True):
        # ignore future_masking parameter
        self_attention_mask = self.get_self_attention_mask(decoder_inputs, batch_first, input_mask, True)
        self_attention_bias = self_attention_mask.type_as(decoder_inputs)
        return self_attention_bias / self_attention_bias.sum(-1, keepdim=True)


class AverageTransformerDecoderLayer(TransformerDecoderLayer):

    # noinspection PyAttributeOutsideInit
    def build_self_attention(self):
        self.preprocess_self_attn = self.get_preprocessing_module()
        self.self_attention = AverageAttention(
            self.model_dim, self.attention_dropout,
            get_feed_forward(self.feed_forward_type,
                             self.model_dim,
                             self.feed_forward_dim,
                             self.feed_forward_dropout,
                             self.weight_norm),
            batch_first=self.batch_first,
            masked_layers=self.masked_layers)
        self.postprocess_self_attn = self.get_postprocessing_module()

    def self_attention_layer(self, inputs, input_mask=None, self_attention_bias=None):
        query = self.preprocess_self_attn(inputs, mask=input_mask)
        self_attention_out, _ = self.self_attention(query, self_attention_bias)
        self_attention_out = self.postprocess_self_attn(self_attention_out, inputs)
        return self_attention_out

    def self_attention_step(self, inputs, incremental_state, input_mask=None, self_attention_bias=None):
        query = self.preprocess_self_attn(inputs, mask=input_mask)
        self_attention_out, _ = self.self_attention.step(query, incremental_state, input_mask)
        self_attention_out = self.postprocess_self_attn(self_attention_out, inputs)
        return self_attention_out
