import torch, copy
import torch.nn as nn
from torch.autograd import Variable
import onmt
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder, Transformer
from onmt.modules.Transformer.Layers import PositionalEncoding

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


    onmt.Constants.layer_norm = opt.layer_norm
    onmt.Constants.weight_norm = opt.weight_norm
    onmt.Constants.activation_layer = opt.activation_layer
    onmt.Constants.version = 1.0
    onmt.Constants.attention_out = opt.attention_out
    onmt.Constants.residual_type = opt.residual_type

    # BUILD POSITIONAL ENCODING
    if opt.time == 'positional_encoding':
        positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN)
    else:
        positional_encoder = None
        raise NotImplementedError

    # BUILD GENERATOR
    generators = [onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())]

    if opt.ctc_loss != 0:
        generators.append(onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size() + 1))
    
    if opt.model == 'transformer':
        # raise NotImplementedError

        onmt.Constants.init_value = opt.param_init

        if opt.encoder_type == "text":
            encoder = TransformerEncoder(opt, dicts['src'], positional_encoder)
        else:
            encoder = TransformerEncoder(opt, opt.input_size, positional_encoder)

        decoder = TransformerDecoder(opt, dicts['tgt'], positional_encoder)

        model = Transformer(encoder, decoder, nn.ModuleList(generators))

    elif opt.model == 'stochastic_transformer':
        
        from onmt.modules.StochasticTransformer.Models import StochasticTransformerEncoder, StochasticTransformerDecoder

        onmt.Constants.weight_norm = opt.weight_norm
        onmt.Constants.init_value = opt.param_init
        
        if opt.encoder_type == "text":
            encoder = StochasticTransformerEncoder(opt, dicts['src'], positional_encoder)
        else:
            encoder = StochasticTransformerEncoder(opt, opt.input_size, positional_encoder)

        decoder = StochasticTransformerDecoder(opt, dicts['tgt'], positional_encoder)

        model = Transformer(encoder, decoder, nn.ModuleList(generators))


    elif opt.model in ['universal_transformer', 'utransformer'] :

        from onmt.modules.UniversalTransformer.Models import UniversalTransformerDecoder, UniversalTransformerEncoder
        from onmt.modules.UniversalTransformer.Layers import TimeEncoding

        onmt.Constants.weight_norm = opt.weight_norm
        onmt.Constants.init_value = opt.param_init

        time_encoder = TimeEncoding(opt.model_size, len_max=32)

        encoder = UniversalTransformerEncoder(opt, dicts['src'], positional_encoder, time_encoder)
        decoder = UniversalTransformerDecoder(opt, dicts['tgt'], positional_encoder, time_encoder)

        model = Transformer(encoder, decoder, nn.ModuleList(generators))

    else:
        raise NotImplementedError

    if opt.tie_weights:  
        print("Joining the weights of decoder input and output embeddings")
        model.tie_weights()
       
    if opt.join_embedding:
        print("Joining the weights of encoder and decoder word embeddings")
        model.share_enc_dec_embedding()

    for g in model.generator:
        init.xavier_uniform_(g.linear.weight)

    if opt.encoder_type == "audio":
        init.xavier_uniform_(model.encoder.audio_trans.weight.data)
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

    from onmt.modules.TransformerLM.Models import TransformerLM

    positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN)

    decoder = TransformerDecoder(opt, dicts['tgt'], positional_encoder, ignore_source=True)

    generators = [onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())]

    model = TransformerLM(None, decoder, nn.ModuleList(generators))

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
    lm_checkpoint = torch.load(opt.lm_checkpoint)

    # first we build the lm model and lm checkpoint
    lm_opt = lm_checkpoint['opt']

    lm_model = build_language_model(lm_opt, dicts)

    # load parameter for pretrained model
    lm_model.load_state_dict(lm_checkpoint['model'])

    # main model for seq2seq (translation, asr)
    tm_model = build_model(opt, dicts)

    from onmt.modules.FusionNetwork.Models import FusionNetwork
    model = FusionNetwork(tm_model, lm_model)

    return model