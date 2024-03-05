from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

from torch.nn import GELU, SiLU
from onmt.models.registry_utils import ArchitectureRegistry
from .norm_order import TransformerNormOrder


@dataclass
class Wav2Vec2EncoderConfig:
    """Holds the configuration of a wav2vec 2.0 encoder."""

    model_dim: int
    """The dimensionality of the model."""

    max_seq_len: int
    """The expected maximum sequence length after feature extraction."""

    # Features
    feature_dim: int
    """The dimensionality of extracted features."""

    use_fbank: bool
    """If ``True``, uses log-mel filterbanks instead of waveforms as input."""

    first_pass_dropout_p: float
    """The dropout probability on extracted features before masking and
    positional encoding."""

    layer_norm_features: bool
    """If ``True``, applies Layer Normalization to extracted features."""

    # Waveform Feature Extractor
    feature_extractor_layer_descs: List[Tuple[int, int, int]]
    """A tuple of output dimension, kernel size, and stride for each feature
    extraction layer."""

    feature_extractor_bias: bool
    """If ``True``, convolutions in feature extraction layers learn an additive
    bias."""

    feature_extractor_layer_norm_convs: bool
    """If ``True``, applies Layer Normalization to outputs of convolutions in
    feature extraction layers."""

    feature_grad_scale: float
    """The scale factor for gradients of extracted features. Setting to a value
    less than 1.0 allows the feature extractor to learn at a lower rate than the
    rest of the model."""

    # Filterbank Feature Extractor
    num_fbank_channels: int
    """The number of source log-mel filterbank channels."""

    fbank_stride: int

    sample_fbank_every_k: int

    # Position Encoder
    pos_encoder_type: str
    """The type of position encoder ('conv', 'relative', 'rotary')."""

    # Convolutional Position Encoder
    pos_encoder_depth: int
    """The number of stacked position encoder layers."""

    pos_conv_kernel_size: int
    """The total kernel size of 1D convolutions in position encoder layers."""

    num_pos_conv_groups: int
    """The number of convolution groups in position encoder layers."""

    # Encoder (i.e. Context Network)
    use_conformer: bool
    """If ``True``, uses Conformer blocks instead of Transformer encoder layers."""

    num_encoder_layers: int
    """The number of Transformer encoder layers."""

    num_encoder_attn_heads: int
    """The number of attention heads in Transformer encoder layers."""

    ffn_inner_dim: int
    """The inner dimensionality of Transformer feed-forward networks."""

    dropout_p: float
    """The dropout probability in Transformer layers."""

    attn_dropout_p: float
    """The dropout probability on Transformer attention weights."""

    layer_drop_p: float
    """If greater than zero, applies LayerDrop to Transformer encoder layers
    as described in :cite:t:`https://doi.org/10.48550/arxiv.1909.11556`."""

    norm_order: TransformerNormOrder
    """The Layer Normalization order."""

    depthwise_conv_kernel_size: int
    """The kernel size of depthwise convolutions in Conformer blocks."""


@dataclass
class Wav2Vec2Config:
    """Holds the configuration of a wav2vec 2.0 model."""

    encoder_config: Wav2Vec2EncoderConfig
    """The configuration of the wav2vec 2.0 encoder."""

    final_dim: int
    """The dimensionality of the final projection that is applied to context
    network outputs and quantized targets."""

    final_proj_bias: bool
    """If ``True``, the final projection learns an additive bias."""

    # Mask
    temporal_mask_span_len: int
    """The length of each temporal mask span that is applied over time steps."""

    max_temporal_mask_prob: float
    """The maximum probability of masking a time step. Note that, due to mask
    span overlap, the effective probability might be smaller."""

    spatial_mask_span_len: int
    """The length of each spatial mask span that is applied over features."""

    max_spatial_mask_prob: float
    """The maximum probability of masking a feature. Note that, due to mask span
    overlap, the effective probability might be smaller."""

    # Quantization
    quantized_dim: int
    """The output dimensionality of vector quantizer."""

    num_codebooks: int
    """The number of codebooks."""

    num_codebook_entries: int
    """The number of entries per codebook."""

    codebook_sampling_temperature: Tuple[float, float, float]
    """A tuple of start temperature, end temperature, and decay factor for
    codebook entry sampling."""

    # Loss
    num_distractors: int
    """The number of distractors to use in contrastive prediction."""

    logit_temp: float
    """The temperature to divide logits by."""

    diversity_loss_weight: float
    """The weight of diversity in loss computation."""


wav2vec2_archs = ArchitectureRegistry[Wav2Vec2Config]("wav2vec2")

wav2vec2_arch = wav2vec2_archs.decorator


def _encoder_600m() -> Wav2Vec2EncoderConfig:
    return Wav2Vec2EncoderConfig(
        model_dim=1024,
        max_seq_len=4096,
        feature_dim=160,
        use_fbank=True,
        first_pass_dropout_p=0.0,
        layer_norm_features=False,
        feature_extractor_layer_descs=[],
        feature_extractor_bias=False,
        feature_extractor_layer_norm_convs=False,
        feature_grad_scale=0.0,
        num_fbank_channels=80,
        fbank_stride=2,
        sample_fbank_every_k=1,
        pos_encoder_type="relative",
        pos_encoder_depth=0,
        pos_conv_kernel_size=0,
        num_pos_conv_groups=0,
        use_conformer=True,
        num_encoder_layers=24,
        num_encoder_attn_heads=16,
        ffn_inner_dim=4096,
        dropout_p=0.0,
        attn_dropout_p=0.0,
        layer_drop_p=0.0,
        norm_order=TransformerNormOrder.POST,
        depthwise_conv_kernel_size=31,
    )


def _encoder_300m() -> Wav2Vec2EncoderConfig:
    return Wav2Vec2EncoderConfig(
        model_dim=1024,
        max_seq_len=4096,
        feature_dim=160,
        use_fbank=True,
        first_pass_dropout_p=0.0,
        layer_norm_features=False,
        feature_extractor_layer_descs=[],
        feature_extractor_bias=False,
        feature_extractor_layer_norm_convs=False,
        feature_grad_scale=0.0,
        num_fbank_channels=80,
        fbank_stride=2,
        sample_fbank_every_k=1,
        pos_encoder_type="relative",
        pos_encoder_depth=0,
        pos_conv_kernel_size=0,
        num_pos_conv_groups=0,
        use_conformer=True,
        num_encoder_layers=12,
        num_encoder_attn_heads=16,
        ffn_inner_dim=4096,
        dropout_p=0.0,
        attn_dropout_p=0.0,
        layer_drop_p=0.0,
        norm_order=TransformerNormOrder.POST,
        depthwise_conv_kernel_size=31,
    )


def _test_config() -> Wav2Vec2EncoderConfig:
    return Wav2Vec2EncoderConfig(
        model_dim=32,
        max_seq_len=4096,
        feature_dim=160,
        use_fbank=True,
        first_pass_dropout_p=0.0,
        layer_norm_features=False,
        feature_extractor_layer_descs=[],
        feature_extractor_bias=False,
        feature_extractor_layer_norm_convs=False,
        feature_grad_scale=0.0,
        num_fbank_channels=80,
        fbank_stride=2,
        sample_fbank_every_k=1,
        pos_encoder_type="relative",
        pos_encoder_depth=0,
        pos_conv_kernel_size=0,
        num_pos_conv_groups=0,
        use_conformer=True,
        num_encoder_layers=12,
        num_encoder_attn_heads=16,
        ffn_inner_dim=4096,
        dropout_p=0.0,
        attn_dropout_p=0.0,
        layer_drop_p=0.0,
        norm_order=TransformerNormOrder.POST,
        depthwise_conv_kernel_size=31,
    )


@dataclass
class W2VBertConfig:
    """Holds the configuration of a w2v-BERT model."""

    w2v2_config: Wav2Vec2Config
    """The configuration of the wav2vec 2.0 model."""

    num_bert_encoder_layers: int
    """The number of Transformer encoder layers to use for masked prediction."""

    num_target_codebooks: int
    """The number of consecutive codebooks to use as masked prediction targets."""

    w2v2_loss_weight: float
    """The weight of wav2vec 2.0 loss in loss computation."""

    bert_loss_weight: float
    """The weight of masked prediction loss in loss computation."""

    bert_label_smoothing: float
    """The amount of label smoothing when computing masked prediction loss."""


w2vbert_archs = ArchitectureRegistry[W2VBertConfig]("w2v-bert")

w2vbert_arch = w2vbert_archs.decorator


@w2vbert_arch("600m")
def _600m() -> W2VBertConfig:
    w2v2_encoder_config = _encoder_600m()

    w2v2_config = Wav2Vec2Config(
        w2v2_encoder_config,
        final_dim=768,
        final_proj_bias=True,
        temporal_mask_span_len=10,
        max_temporal_mask_prob=0.65,
        spatial_mask_span_len=10,
        max_spatial_mask_prob=0.0,
        quantized_dim=1024,
        num_codebooks=1,
        num_codebook_entries=1024,
        codebook_sampling_temperature=(2.0, 0.1, 0.999995),
        num_distractors=100,
        logit_temp=0.1,
        diversity_loss_weight=0.2,
    )

    return W2VBertConfig(
        w2v2_config,
        num_bert_encoder_layers=16,
        num_target_codebooks=1,
        w2v2_loss_weight=1.0,
        bert_loss_weight=1.0,
        bert_label_smoothing=0.0,
    )


@w2vbert_arch("test")
def _w2vbert_test_config() -> W2VBertConfig:
    w2v2_encoder_config = _test_config()

    w2v2_config = Wav2Vec2Config(
        w2v2_encoder_config,
        final_dim=32,
        final_proj_bias=True,
        temporal_mask_span_len=10,
        max_temporal_mask_prob=0.65,
        spatial_mask_span_len=10,
        max_spatial_mask_prob=0.0,
        quantized_dim=32,
        num_codebooks=1,
        num_codebook_entries=16,
        codebook_sampling_temperature=(2.0, 0.1, 0.999995),
        num_distractors=100,
        logit_temp=0.1,
        diversity_loss_weight=0.2,
    )

    return W2VBertConfig(
        w2v2_config,
        num_bert_encoder_layers=16,
        num_target_codebooks=1,
        w2v2_loss_weight=1.0,
        bert_loss_weight=1.0,
        bert_label_smoothing=0.0,
    )


w2vbert_test_config = _w2vbert_test_config


@dataclass
class ShawRelativePositionSDPAConfig:
    """Holds the configuration of the :class:ShawRelativePositionSDPA module."""

    max_left_rel_pos: int
    """The left clipping value for relative positions."""

    max_right_rel_pos: Optional[int]
    """The right clipping value for relative positions."""

    use_rel_pos_values: bool = False
    """If True, also uses relative position values to compute relative attention."""


@dataclass
class ConformerShawEncoderConfig(Wav2Vec2EncoderConfig):
    """Holds the configuration of a conformer shaw encoder."""

    shaw_rel_pos_sdpa_config: Optional[ShawRelativePositionSDPAConfig]
    """The parameters for ShawRelativePositionSDPA."""


conformer_shaw_archs = ArchitectureRegistry[ConformerShawEncoderConfig](
    "conformer_shaw"
)

conformer_shaw_arch = conformer_shaw_archs.decorator


@conformer_shaw_arch("600m")
def _conformer_shaw_600m_encoder() -> ConformerShawEncoderConfig:
    w2vbert_config = w2vbert_archs.get_config("600m")
    w2v2_encoder_config = w2vbert_config.w2v2_config.encoder_config
    sdpa_config = ShawRelativePositionSDPAConfig(
        max_left_rel_pos=64,
        max_right_rel_pos=8,
        use_rel_pos_values=False,
    )
    conformer_shaw_encoder_config = ConformerShawEncoderConfig(
        **asdict(w2v2_encoder_config),
        shaw_rel_pos_sdpa_config=sdpa_config,
    )
    conformer_shaw_encoder_config.pos_encoder_type = "shaw_relative"
    return conformer_shaw_encoder_config


@wav2vec2_arch("conformer_shaw_600m")
def _conformer_shaw_600m() -> Wav2Vec2Config:
    encoder_config = _conformer_shaw_600m_encoder()

    return Wav2Vec2Config(
        encoder_config,
        final_dim=768,
        final_proj_bias=True,
        temporal_mask_span_len=10,
        max_temporal_mask_prob=0.65,
        spatial_mask_span_len=10,
        max_spatial_mask_prob=0.0,
        quantized_dim=768,
        num_codebooks=2,
        num_codebook_entries=320,
        codebook_sampling_temperature=(2.0, 0.1, 0.999995),
        num_distractors=100,
        logit_temp=0.1,
        diversity_loss_weight=0.2,
    )

conformer_shaw_600m = _conformer_shaw_600m


@conformer_shaw_arch("test")
def _conformer_shaw_test_encoder() -> ConformerShawEncoderConfig:
    w2vbert_config = w2vbert_archs.get_config("test")
    w2v2_encoder_config = w2vbert_config.w2v2_config.encoder_config
    sdpa_config = ShawRelativePositionSDPAConfig(
        max_left_rel_pos=64,
        max_right_rel_pos=8,
        use_rel_pos_values=False,
    )
    conformer_shaw_encoder_config = ConformerShawEncoderConfig(
        **asdict(w2v2_encoder_config),
        shaw_rel_pos_sdpa_config=sdpa_config,
    )
    conformer_shaw_encoder_config.pos_encoder_type = "shaw_relative"
    return conformer_shaw_encoder_config


@wav2vec2_arch("conformer_shaw_test")
def _conformer_shaw_test() -> Wav2Vec2Config:
    encoder_config = _conformer_shaw_test_encoder()

    return Wav2Vec2Config(
        encoder_config,
        final_dim=32,
        final_proj_bias=True,
        temporal_mask_span_len=10,
        max_temporal_mask_prob=0.65,
        spatial_mask_span_len=10,
        max_spatial_mask_prob=0.0,
        quantized_dim=32,
        num_codebooks=2,
        num_codebook_entries=320,
        codebook_sampling_temperature=(2.0, 0.1, 0.999995),
        num_distractors=100,
        logit_temp=0.1,
        diversity_loss_weight=0.2,
    )


conformer_shaw_test = _conformer_shaw_test
