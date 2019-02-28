import torch

from torch import nn

from nmtg.models import register_model
from nmtg.models.transformer import Transformer
from nmtg.models.transformer.transformer import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, \
    TransformerDecoderLayer


@register_model('stochastic_transformer')
class StochasticTransformer(Transformer):
    def __init__(self, *, death_rate=0.5, **kwargs):
        self.death_rate = death_rate
        super().__init__(**kwargs)

    @staticmethod
    def add_options(parser):
        Transformer.add_options(parser)
        parser.add_argument('-death_rate', type=float, default=0.5,
                            help='Stochastic layer death rate')

    def get_encoder(self, positional_encoding):
        encoder = StochasticTransformerEncoder(
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
            death_rate=self.death_rate
        )
        return encoder

    def get_decoder(self, positional_encoding, encoder):
        decoder = StochasticTransformerDecoder(
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
            death_rate=self.death_rate,
            ignore_context=self.ignore_context,
            encoder_to_share=encoder if self.share_encoder_decoder else None
        )
        return decoder


class StochasticTransformerEncoder(TransformerEncoder):
    def __init__(self, *, death_rate=0.5, **kwargs):
        self.death_rate = death_rate
        super().__init__(**kwargs)

    def build_layers(self):
        layers = nn.ModuleList([StochasticTransformerEncoderLayer(
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
            death_rate=(i + 1) / self.num_layers * self.death_rate  # linearly decay death rate
        ) for i in range(self.num_layers)])
        return layers


class StochasticTransformerDecoder(TransformerDecoder):
    def __init__(self, *, death_rate=0.5, **kwargs):
        self.death_rate = death_rate
        super().__init__(**kwargs)

    def build_layers(self, encoder=None):
        layers = nn.ModuleList([StochasticTransformerDecoderLayer(
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
            death_rate=(i + 1) / self.num_layers * self.death_rate,  # linearly decay death rate
            ignore_context=self.ignore_context,
            encoder_to_share=encoder.layers[i] if encoder is not None else None
        ) for i in range(self.num_layers)])
        return layers


class StochasticTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, *, death_rate=0.5, **kwargs):
        self.death_rate = death_rate
        super().__init__(**kwargs)

    def forward(self, inputs, input_mask=None, attention_bias=None):
        if self.training and torch.rand(1).item() >= self.death_rate:
            return inputs
        else:
            return super().forward(inputs, input_mask, attention_bias)

    def self_attention_layer(self, inputs, input_mask=None, self_attention_bias=None):
        query = self.preprocess_attn(inputs, mask=input_mask)
        attention_out, _ = self.attention(query, query, query, self_attention_bias, input_mask)

        if self.training:
            attention_out = attention_out / (1 - self.death_rate)

        attention_out = self.postprocess_attn(attention_out, inputs)
        return attention_out

    def feed_forward_layer(self, inputs, input_mask=None):
        out = self.preprocess_ffn(inputs, mask=input_mask)
        out = self.apply_feed_forward(out, input_mask)

        if self.training:
            out = out / (1 - self.death_rate)

        out = self.postprocess_ffn(out, inputs)
        return out


class StochasticTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, *, death_rate=0.5, **kwargs):
        self.death_rate = death_rate
        super().__init__(**kwargs)

    def forward(self, inputs, context, input_mask=None, context_mask=None, self_attention_bias=None,
                encoder_attention_bias=None):
        if self.training and torch.rand(1).item() >= self.death_rate:
            return inputs
        else:
            return super().forward(inputs, context, input_mask, context_mask,
                                   self_attention_bias, encoder_attention_bias)

    def self_attention_layer(self, inputs, input_mask=None, self_attention_bias=None):
        query = self.preprocess_self_attn(inputs, mask=input_mask)
        self_attention_out, _ = self.self_attention(query, query, query, self_attention_bias, input_mask)

        if self.training:
            self_attention_out = self_attention_out / (1 - self.death_rate)

        self_attention_out = self.postprocess_self_attn(self_attention_out, inputs)
        return self_attention_out

    def encoder_attention_layer(self, inputs, encoder_outputs, input_mask=None,
                                context_mask=None, encoder_attention_bias=None):
        query = self.preprocess_enc_attn(inputs, mask=input_mask)
        enc_attention_out, attention_weights = self.enc_attention(
            query, encoder_outputs, encoder_outputs,
            encoder_attention_bias, input_mask, context_mask)

        if self.training:
            enc_attention_out = enc_attention_out / (1 - self.death_rate)

        enc_attention_out = self.postprocess_enc_attn(enc_attention_out, inputs)
        return enc_attention_out, attention_weights

    def feed_forward_layer(self, inputs, input_mask=None):
        out = self.preprocess_ffn(inputs, mask=input_mask)
        out = self.feed_forward(out, mask=input_mask if self.masked_layers else None)

        if self.training:
            out = out / (1 - self.death_rate)

        out = self.postprocess_ffn(out, inputs)
        return out

    # We don't apply scaling or layer dropout during incremental decoding, so no step functions here
