import torch
import torch.nn as nn
import onmt
from onmt.models.transformers import TransformerEncoder, TransformerDecoder, Transformer, MixedEncoder
from onmt.models.relative_transformer import RelativeTransformer
from onmt.models.bayes_by_backprop.relative_transformer import \
    RelativeTransformerEncoder, RelativeTransformerDecoder, BayesianTransformer
from onmt.models.transformer_layers import PositionalEncoding
from onmt.models.relative_transformer import SinusoidalPositionalEmbedding, RelativeTransformer
from onmt.modules.copy_generator import CopyGenerator
from onmt.options import backward_compatible
import math

init = torch.nn.init

MAX_LEN = onmt.constants.max_position_length  # This should be the longest sentence from the dataset


def build_model(opt, dicts):
    opt = backward_compatible(opt)

    onmt.constants.layer_norm = opt.layer_norm
    onmt.constants.weight_norm = opt.weight_norm
    onmt.constants.activation_layer = opt.activation_layer
    onmt.constants.version = 1.0
    onmt.constants.attention_out = opt.attention_out
    onmt.constants.residual_type = opt.residual_type

    if not opt.fusion:
        model = build_tm_model(opt, dicts)
    else:
        raise NotImplementedError
        model = build_fusion(opt, dicts)

    return model


def build_tm_model(opt, dicts):
    onmt.constants.neg_log_sigma1 = opt.neg_log_sigma1
    onmt.constants.neg_log_sigma2 = opt.neg_log_sigma2
    onmt.constants.prior_pi = opt.prior_pi

    # BUILD POSITIONAL ENCODING
    if opt.time == 'positional_encoding':
        positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN)
    else:
        raise NotImplementedError

    # BUILD GENERATOR
    if opt.copy_generator:
        generators = [CopyGenerator(opt.model_size, dicts['tgt'].size(),
                                    fix_norm=opt.fix_norm_output_embedding)]
    else:
        generators = [onmt.modules.base_seq2seq.Generator(opt.model_size, dicts['tgt'].size(),
                                                          fix_norm=opt.fix_norm_output_embedding)]

    # BUILD EMBEDDINGS
    if 'src' in dicts:
        embedding_src = nn.Embedding(dicts['src'].size(),
                                     opt.model_size,
                                     padding_idx=onmt.constants.PAD)
    else:
        embedding_src = None

    if opt.join_embedding and embedding_src is not None:
        embedding_tgt = embedding_src
        print("* Joining the weights of encoder and decoder word embeddings")
    else:
        embedding_tgt = nn.Embedding(dicts['tgt'].size(),
                                     opt.model_size,
                                     padding_idx=onmt.constants.PAD)

    if opt.use_language_embedding:
        print("* Create language embeddings with %d languages" % len(dicts['langs']))
        language_embeddings = nn.Embedding(len(dicts['langs']), opt.model_size)
    else:
        language_embeddings = None

    if opt.encoder_type == "text":
        encoder = RelativeTransformerEncoder(opt, embedding_src, None,
                                             opt.encoder_type, language_embeddings=language_embeddings)
    if opt.encoder_type == "audio":
        # raise NotImplementedError
        encoder = RelativeTransformerEncoder(opt, None, None, encoder_type=opt.encoder_type,
                                             language_embeddings=language_embeddings)

    generator = nn.ModuleList(generators)
    decoder = RelativeTransformerDecoder(opt, embedding_tgt, None, language_embeddings=language_embeddings)

    if opt.reconstruct:
        rev_decoder = RelativeTransformerDecoder(opt, embedding_src, None, language_embeddings=language_embeddings)
        rev_generator = [onmt.modules.base_seq2seq.Generator(opt.model_size, dicts['src'].size(),
                                                             fix_norm=opt.fix_norm_output_embedding)]
        rev_generator = nn.ModuleList(rev_generator)
    else:
        rev_decoder = None
        rev_generator = None

    model = BayesianTransformer(encoder, decoder, generator, rev_decoder, rev_generator, mirror=opt.mirror_loss)

    if opt.tie_weights:
        print("* Joining the weights of decoder input and output embeddings")
        model.tie_weights()

    return model

