import torch
import torch.nn as nn
import onmt
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder, Transformer, MixedEncoder
from onmt.modules.Transformer.Layers import PositionalEncoding
from onmt.modules.RelativeTransformer.Layers import SinusoidalPositionalEmbedding

init = torch.nn.init

MAX_LEN = onmt.Constants.max_position_length  # This should be the longest sentence from the dataset


def build_model(opt, dicts):

    model = None
    
    if not hasattr(opt, 'model'):
        opt.model = 'recurrent'
        
    if not hasattr(opt, 'layer_norm'):
        opt.layer_norm = 'slow'
        
    if not hasattr(opt, 'attention_out'):
        opt.attention_out = 'default'
    
    if not hasattr(opt, 'residual_type'):
        opt.residual_type = 'regular'

    if not hasattr(opt, 'input_size'):
        opt.input_size = 40

    if not hasattr(opt, 'init_embedding'):
        opt.init_embedding = 'xavier'

    if not hasattr(opt, 'ctc_loss'):
        opt.ctc_loss = 0

    if not hasattr(opt, 'encoder_layers'):
        opt.encoder_layers = -1

    if not hasattr(opt, 'fusion'):
        opt.fusion = False

    if not hasattr(opt, 'cnn_downsampling'):
        opt.cnn_downsampling = False

    if not hasattr(opt, 'switchout'):
        opt.switchout = 0.0

    if not hasattr(opt, 'variational_dropout'):
        opt.variational_dropout = False

    onmt.Constants.layer_norm = opt.layer_norm
    onmt.Constants.weight_norm = opt.weight_norm
    onmt.Constants.activation_layer = opt.activation_layer
    onmt.Constants.version = 1.0
    onmt.Constants.attention_out = opt.attention_out
    onmt.Constants.residual_type = opt.residual_type

    if not opt.fusion:
        model = build_tm_model(opt, dicts)
    else:
        model = build_fusion(opt, dicts)

    return model


def build_tm_model(opt, dicts):

    # BUILD POSITIONAL ENCODING
    if opt.time == 'positional_encoding':
        positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN)
    else:
        raise NotImplementedError

    # BUILD GENERATOR
    generators = [onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())]

    # BUILD EMBEDDING
    if 'src' in dicts:
        embedding_src = nn.Embedding(dicts['src'].size(),
                                     opt.model_size,
                                     padding_idx=onmt.Constants.PAD)
    else:
        embedding_src = None

    if opt.join_embedding and embedding_src is not None:
        embedding_tgt = embedding_src
        print("* Joining the weights of encoder and decoder word embeddings")
    else:
        embedding_tgt = nn.Embedding(dicts['tgt'].size(),
                                     opt.model_size,
                                     padding_idx=onmt.Constants.PAD)

    if 'atb' in dicts and dicts['atb'] is not None:
        from onmt.modules.Utilities import AttributeEmbeddings
        #
        attribute_embeddings = AttributeEmbeddings(dicts['atb'], opt.model_size)
        # attribute_embeddings = nn.Embedding(dicts['atb'].size(), opt.model_size)

    else:
        attribute_embeddings = None

    if opt.ctc_loss != 0:
        generators.append(onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size() + 1))

    if opt.model == 'transformer':
        # raise NotImplementedError

        onmt.Constants.init_value = opt.param_init

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
        
        from onmt.modules.StochasticTransformer.Models import StochasticTransformerEncoder, StochasticTransformerDecoder

        onmt.Constants.weight_norm = opt.weight_norm
        onmt.Constants.init_value = opt.param_init
        
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

    elif opt.model == 'dlcltransformer' :

        from onmt.modules.DynamicTransformer.Models import DlclTransformerDecoder, DlclTransformerEncoder

        if opt.encoder_type == "text":
            encoder = DlclTransformerEncoder(opt, embedding_src, positional_encoder, opt.encoder_type)
        elif opt.encoder_type == "audio":
            encoder = DlclTransformerEncoder(opt, None, positional_encoder, opt.encoder_type)

        decoder = DlclTransformerDecoder(opt, embedding_tgt, positional_encoder)

        model = Transformer(encoder, decoder, nn.ModuleList(generators))
    elif opt.model == 'relative_transformer':
        from onmt.modules.RelativeTransformer.Models import RelativeTransformer
        positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)
        # if opt.encoder_type == "text":
        # encoder = TransformerEncoder(opt, embedding_src, positional_encoder, opt.encoder_ty   pe)
        # encoder = RelativeTransformerEncoder(opt, embedding_src, relative_positional_encoder, opt.encoder_type)
        if opt.encoder_type == "audio":
            raise NotImplementedError
            # encoder = TransformerEncoder(opt, None, positional_encoder, opt.encoder_type)
        generator = nn.ModuleList(generators)
        model = RelativeTransformer(opt, [embedding_src, embedding_tgt], positional_encoder, generator=generator)
        # model = Transformer(encoder, decoder, )

    else:
        raise NotImplementedError

    if opt.tie_weights:  
        print("Joining the weights of decoder input and output embeddings")
        model.tie_weights()

    for g in model.generator:
        init.xavier_uniform_(g.linear.weight)

    if opt.encoder_type == "audio":

        if opt.init_embedding == 'xavier':
            init.xavier_uniform_(model.decoder.word_lut.weight)
        elif opt.init_embedding == 'normal':
            init.normal_(model.decoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)
    else:
        if opt.init_embedding == 'xavier':
            init.xavier_uniform_(model.encoder.word_lut.weight)
            init.xavier_uniform_(model.decoder.word_lut.weight)
        elif opt.init_embedding == 'normal':
            init.normal_(model.encoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)
            init.normal_(model.decoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)

    return model


def init_model_parameters(model, opt):

    # currently this function does not do anything
    # because the parameters are locally initialized
    pass


def build_language_model(opt, dicts):

    onmt.Constants.layer_norm = opt.layer_norm
    onmt.Constants.weight_norm = opt.weight_norm
    onmt.Constants.activation_layer = opt.activation_layer
    onmt.Constants.version = 1.0
    onmt.Constants.attention_out = opt.attention_out
    onmt.Constants.residual_type = opt.residual_type

    from onmt.modules.LSTMLM.Models import LSTMLMDecoder, LSTMLM

    decoder = LSTMLMDecoder(opt, dicts['tgt'])

    generators = [onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())]

    model = LSTMLM(None, decoder, nn.ModuleList(generators))

    if opt.tie_weights:
        print("Joining the weights of decoder input and output embeddings")
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

    from onmt.modules.FusionNetwork.Models import FusionNetwork
    model = FusionNetwork(tm_model, lm_model)

    return model
