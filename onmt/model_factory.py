import torch
import torch.nn as nn
import onmt
from onmt.models.tranformers import TransformerEncoder, TransformerDecoder, Transformer, MixedEncoder
from onmt.models.transformer_layers import PositionalEncoding
from onmt.models.relative_transformer import SinusoidalPositionalEmbedding
from onmt.modules.copy_generator import CopyGenerator
from options import backward_compatible

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

    # BUILD POSITIONAL ENCODING
    if opt.time == 'positional_encoding':
        positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN)
    else:
        raise NotImplementedError

    # BUILD GENERATOR
    if opt.copy_generator:
        generators = [CopyGenerator(opt.model_size, dicts['tgt'].size())]
    else:
        generators = [onmt.modules.base_seq2seq.Generator(opt.model_size, dicts['tgt'].size())]

    # BUILD EMBEDDING
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

    if 'atb' in dicts and dicts['atb'] is not None:
        from onmt.modules.utilities import AttributeEmbeddings
        #
        attribute_embeddings = AttributeEmbeddings(dicts['atb'], opt.model_size)
        # attribute_embeddings = nn.Embedding(dicts['atb'].size(), opt.model_size)

    else:
        attribute_embeddings = None

    if opt.ctc_loss != 0:
        generators.append(onmt.modules.base_seq2seq.Generator(opt.model_size, dicts['tgt'].size() + 1))

    if opt.model == 'transformer':
        # raise NotImplementedError

        onmt.constants.init_value = opt.param_init

        if opt.encoder_type == "text":
            encoder = TransformerEncoder(opt, embedding_src, positional_encoder, opt.encoder_type)
        elif opt.encoder_type == "audio":
            encoder = TransformerEncoder(opt, None, positional_encoder, opt.encoder_type)
        elif opt.encoder_type == "mix":
            text_encoder = TransformerEncoder(opt, embedding_src, positional_encoder, "text")
            audio_encoder = TransformerEncoder(opt, None, positional_encoder, "audio")
            encoder = MixedEncoder(text_encoder, audio_encoder)
        else:
            print ("Unknown encoder type:", opt.encoder_type)
            exit(-1)

        decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, attribute_embeddings=attribute_embeddings)

        model = Transformer(encoder, decoder, nn.ModuleList(generators))

    elif opt.model == 'stochastic_transformer':
        
        from onmt.models.stochastic_transformers import StochasticTransformerEncoder, StochasticTransformerDecoder

        onmt.constants.weight_norm = opt.weight_norm
        onmt.constants.init_value = opt.param_init
        
        if opt.encoder_type == "text":
            encoder = StochasticTransformerEncoder(opt, embedding_src, positional_encoder, opt.encoder_type)
        elif opt.encoder_type == "audio":
            encoder = StochasticTransformerEncoder(opt, 0, positional_encoder, opt.encoder_type)
        elif opt.encoder_type == "mix":
            text_encoder = StochasticTransformerEncoder(opt, embedding_src, positional_encoder, "text")
            audio_encoder = StochasticTransformerEncoder(opt, None, positional_encoder, "audio")
            encoder = MixedEncoder(text_encoder, audio_encoder)
        else:
            print ("Unknown encoder type:", opt.encoder_type)
            exit(-1)

        decoder = StochasticTransformerDecoder(opt, embedding_tgt, positional_encoder, attribute_embeddings=attribute_embeddings)

        model = Transformer(encoder, decoder, nn.ModuleList(generators))

    elif opt.model == 'relative_transformer':

        from onmt.models.relative_transformer import RelativeTransformerEncoder, RelativeTransformerDecoder

        if opt.encoder_type == "text":
            encoder = RelativeTransformerEncoder(opt, embedding_src, None, opt.encoder_type)
        if opt.encoder_type == "audio":
            # raise NotImplementedError
            encoder = RelativeTransformerEncoder(opt, None, None, encoder_type=opt.encoder_type)

        generator = nn.ModuleList(generators)
        decoder = RelativeTransformerDecoder(opt, embedding_tgt, None, attribute_embeddings=attribute_embeddings)
        model = Transformer(encoder, decoder, generator)

    elif opt.model == 'unified_transformer':
        from onmt.models.relative_transformer import RelativeTransformer

        if opt.encoder_type == "audio":
            raise NotImplementedError

        generator = nn.ModuleList(generators)
        model = RelativeTransformer(opt, embedding_src, embedding_tgt,
                                    generator, None, attribute_embeddings=attribute_embeddings)

    else:
        raise NotImplementedError

    # TODO: adding the "united" model (one encoder and one decoder)

    if opt.tie_weights:  
        print("Joining the weights of decoder input and output embeddings")
        model.tie_weights()

    # if opt.encoder_type == "audio":
    #
    #     if opt.init_embedding == 'xavier':
    #         init.xavier_uniform_(model.decoder.word_lut.weight)
    #     elif opt.init_embedding == 'normal':
    #         init.normal_(model.decoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)
    # else:
    #     if opt.init_embedding == 'xavier':
    #         init.xavier_uniform_(model.encoder.word_lut.weight)
    #         init.xavier_uniform_(model.decoder.word_lut.weight)
    #     elif opt.init_embedding == 'normal':
    #         # init.normal_(model.encoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)
    #         # init.normal_(model.decoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)
    #         init.normal_(model.encoder.word_lut.weight, mean=0, std=0.01)
    #         init.normal_(model.decoder.word_lut.weight, mean=0, std=0.01)

    return model


def init_model_parameters(model, opt):
    """
    Initializing model parameters. Mostly using normal distribution (0, std)
    """
    init_std = 0.02  # magic number

    def init_weight(weight):
        nn.init.normal_(weight, 0.0, init_std)

    def init_embed(weight):
        nn.init.normal_(weight, mean=0, std=init_std)

    def init_bias(bias):
        nn.init.constant_(bias, 0.0)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                init_bias(m.bias)
        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                init_weight(m.weight)
        elif classname.find('LayerNorm') != -1 or classname.find('FusedLayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 1.0, init_std)
            if hasattr(m, 'bias') and m.bias is not None:
                init_bias(m.bias)
        elif classname.find('RelativeTransformerEncoder') != -1:
            if hasattr(m, 'r_emb'):
                init_weight(m.r_emb)
            if hasattr(m, 'r_w_bias'):
                init_weight(m.r_w_bias)
            if hasattr(m, 'r_r_bias'):
                init_weight(m.r_r_bias)
            if hasattr(m, 'r_bias'):
                init_bias(m.r_bias)
        elif classname.find('RelativeTransformerDecoder') != -1:
            if hasattr(m, 'r_emb'):
                init_weight(m.r_emb)
            if hasattr(m, 'r_w_bias'):
                init_weight(m.r_w_bias)
            if hasattr(m, 'r_r_bias'):
                init_weight(m.r_r_bias)
            if hasattr(m, 'r_bias'):
                init_bias(m.r_bias)
        elif classname.find('RelPartialLearnableMultiHeadAttn') != -1:
            if hasattr(m, 'r_w_bias'):
                init_weight(m.r_w_bias)
            if hasattr(m, 'r_r_bias'):
                init_weight(m.r_r_bias)

    model.apply(weights_init)

    if hasattr(model, 'decoder'):
        model.decoder.word_lut.apply(weights_init)
    else:
        model.tgt_embedding.apply(weights_init)

    return


def build_language_model(opt, dicts):

    onmt.constants.layer_norm = opt.layer_norm
    onmt.constants.weight_norm = opt.weight_norm
    onmt.constants.activation_layer = opt.activation_layer
    onmt.constants.version = 1.0
    onmt.constants.attention_out = opt.attention_out
    onmt.constants.residual_type = opt.residual_type

    from onmt.legacy.LSTMLM.Models import LSTMLMDecoder, LSTMLM

    decoder = LSTMLMDecoder(opt, dicts['tgt'])

    generators = [onmt.modules.base_seq2seq.Generator(opt.model_size, dicts['tgt'].size())]

    model = LSTMLM(None, decoder, nn.ModuleList(generators))

    if opt.tie_weights:
        print("* Joining the weights of decoder input and output embeddings")
        model.tie_weights()

    for g in model.generator:
        init.xavier_uniform_(g.linear.weight)

    init.normal_(model.decoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)

    return model


def build_fusion(opt, dicts):

    # the fusion model requires a pretrained language model
    print("Loading pre-trained language model from %s" % opt.lm_checkpoint)
    lm_checkpoint = torch.load(opt.lm_checkpoint, map_location=lambda storage, loc: storage)

    # first we build the lm model and lm checkpoint
    lm_opt = lm_checkpoint['opt']

    lm_model = build_language_model(lm_opt, dicts)

    # load parameter for pretrained model
    lm_model.load_state_dict(lm_checkpoint['model'])

    # main model for seq2seq (translation, asr)
    tm_model = build_tm_model(opt, dicts)

    from onmt.legacy.FusionNetwork.Models import FusionNetwork
    model = FusionNetwork(tm_model, lm_model)

    return model


def optimize_model(model):
    """
    Used to potentially upgrade the components with more optimized counterparts in the future
    """

    def replace_layer_norm(m, name):

        replacable = True
        try:
            # from apex.normalization.fused_layer_norm import FusedLayerNorm
            import importlib
            from apex.normalization.fused_layer_norm import FusedLayerNorm
            fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")

        except ModuleNotFoundError:
            replacable = False

        if replacable:
            for attr_str in dir(m):
                target_attr = getattr(m, attr_str)
                if type(target_attr) == torch.nn.LayerNorm:
                    setattr(m, attr_str, FusedLayerNorm(target_attr.normalized_shape,
                                                        eps=target_attr.eps,
                                                        elementwise_affine=target_attr.elementwise_affine))
            for n, ch in m.named_children():
                replace_layer_norm(ch, n)

    replace_layer_norm(model, "Transformer")
