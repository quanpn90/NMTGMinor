import torch, copy
import torch.nn as nn
from torch.autograd import Variable
import onmt
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder, Transformer
from onmt.modules.Transformer.Layers import PositionalEncoding
from onmt.modules.Loss import NMTLossFunc


def update_backward_compatibility(opt):
    
    if not hasattr(opt, 'model'):
        opt.model = 'recurrent'
        
    if not hasattr(opt, 'layer_norm'):
        opt.layer_norm = 'slow'
        
    if not hasattr(opt, 'attention_out'):
        opt.attention_out = 'default'
    
    if not hasattr(opt, 'residual_type'):
        opt.residual_type = 'regular'
        
    if not hasattr(opt, 'init_embedding'):
        opt.init_embedding = 'xavier'
        
    if not hasattr(opt, 'death_type'):
        opt.death_type = 'linear_decay'
    
    if not hasattr(opt, 'residual_dropout'):
        opt.residual_dropout = opt.dropout

    if not hasattr(opt, 'share_enc_dec_weights'):
        opt.share_enc_dec_weights = False

    if not hasattr(opt, 'var_posterior_share_weight'):
        opt.var_posterior_share_weight = False

    if not hasattr(opt, 'var_posterior_combine'):
        opt.var_posterior_combine = 'concat'

    if not hasattr(opt, 'var_ignore_first_source_token'):
        opt.var_ignore_first_source_token = False

    if not hasattr(opt, 'var_ignore_first_target_token'):
        opt.var_ignore_first_target_token = False

    if not hasattr(opt, 'var_ignore_source'):
        opt.var_ignore_source = False

    if not hasattr(opt, 'var_pooling'):
        opt.var_pooling = 'mean'

    if not hasattr(opt, 'var_combine_z'):
        opt.var_combine_z = 'once'
        
    return opt


def build_model(opt, dicts):

    opt = update_backward_compatibility(opt)
    
    onmt.Constants.layer_norm = opt.layer_norm
    onmt.Constants.weight_norm = opt.weight_norm
    onmt.Constants.activation_layer = opt.activation_layer
    onmt.Constants.version = 1.0
    onmt.Constants.attention_out = opt.attention_out
    onmt.Constants.residual_type = opt.residual_type
    onmt.Constants.init_value = opt.param_init
    
    MAX_LEN = onmt.Constants.max_position_length  # This should be the longest sentence from the dataset

    embedding_src = nn.Embedding(dicts['src'].size(),
                                     opt.model_size,
                                     padding_idx=onmt.Constants.PAD)

    if opt.join_embedding:
        embedding_tgt = embedding_src
    else:
        embedding_tgt = nn.Embedding(dicts['tgt'].size(),
                                     opt.model_size,
                                     padding_idx=onmt.Constants.PAD)

    feat_embedding = nn.Embedding(dicts['atb'].size(), opt.model_size)

    positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN)
    # print("* Joining the weights of encoder and decoder word embeddings")
    # model.share_enc_dec_embedding()

    if opt.model == 'recurrent' or opt.model == 'rnn':
    
        from onmt.modules.rnn.Models import RecurrentEncoder, RecurrentDecoder, RecurrentModel 

        encoder = RecurrentEncoder(opt, embedding_src)

        decoder = RecurrentDecoder(opt, embedding_tgt)
        
        generator = onmt.modules.BaseModel.Generator(opt.rnn_size, dicts['tgt'].size())
        
        model = RecurrentModel(encoder, decoder, generator)    

        loss_function = NMTLossFunc(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)
        
    elif opt.model == 'transformer':

        encoder = TransformerEncoder(opt, embedding_src, positional_encoder)
        decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding,
                                     encoder_to_share=encoder if opt.share_enc_dec_weights else None)
        
        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
        
        model = Transformer(encoder, decoder, generator)    

        loss_function = NMTLossFunc(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)

    elif opt.model == 'stochastic_transformer':
        
        from onmt.modules.StochasticTransformer.Models import StochasticTransformerEncoder, StochasticTransformerDecoder
        
        onmt.Constants.weight_norm = opt.weight_norm
        onmt.Constants.init_value = opt.param_init

        encoder = StochasticTransformerEncoder(opt, embedding_src, positional_encoder)
        
        decoder = StochasticTransformerDecoder(opt, embedding_tgt, positional_encoder)
        
        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
        
        model = Transformer(encoder, decoder, generator)       
        
        loss_function = NMTLossFunc(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)
        
    elif opt.model == 'fctransformer':
    
        from onmt.modules.FCTransformer.Models import FCTransformerEncoder, FCTransformerDecoder
        
        onmt.Constants.weight_norm = opt.weight_norm
        onmt.Constants.init_value = opt.param_init
        
        positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN )
        
        encoder = FCTransformerEncoder(opt, embedding_src, positional_encoder)
        decoder = FCTransformerDecoder(opt, embedding_tgt, positional_encoder)
        
        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
        
        model = Transformer(encoder, decoder, generator)

    elif opt.model == 'simplified_transformer' or opt.model == 'l2_transformer':

        from onmt.modules.SimplifiedTransformer.Models import SimplifiedTransformerEncoder, SimplifiedTransformer

        encoder = SimplifiedTransformerEncoder(opt, embedding_src, positional_encoder)

        decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding)

        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())

        tgt_encoder = SimplifiedTransformerEncoder(opt, embedding_tgt, positional_encoder, share=encoder) \
                            if opt.model == 'l2_transformer' else None

        model = SimplifiedTransformer(encoder, decoder, generator, tgt_encoder=tgt_encoder)

        if opt.model == 'simplified_transformer':

            loss_function = NMTLossFunc(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)
        elif opt.model == 'l2_transformer':

            from onmt.modules.SimplifiedTransformer.NMTL2Loss import NMTL2Loss
            loss_function = NMTL2Loss(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)

    elif opt.model == 'simplified_transformer_v2' or opt.model == 'l2_simplified_transformer_v2' :

        from onmt.modules.CompressedTransformer.Models import CompressedTransformerEncoder
        from onmt.modules.SimplifiedTransformer.Models import SimplifiedTransformerEncoder, SimplifiedTransformer

        encoder = CompressedTransformerEncoder(opt, embedding_src, positional_encoder)

        decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding)

        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())

        tgt_encoder = None

        model = SimplifiedTransformer(encoder, decoder, generator, tgt_encoder=tgt_encoder)

        loss_function = NMTLossFunc(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)

    # elif opt.model == 'l2_transformer':
    #
    #     from onmt.modules.SimplifiedTransformer.Models import SimplifiedTransformerEncoder, SimplifiedTransformer
    #     from onmt.modules.SimplifiedTransformer.NMTL2Loss import NMTL2Loss
    #
    #     encoder = SimplifiedTransformerEncoder(opt, embedding_src, positional_encoder)
    #     decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding)
    #
    #     tgt_encoder = SimplifiedTransformerEncoder(opt, embedding_tgt, positional_encoder, share=encoder)
    #
    #     generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
    #
    #     model = SimplifiedTransformer(encoder, decoder, generator, tgt_encoder=tgt_encoder)
    #
    #     loss_function = NMTL2Loss(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)

    elif opt.model == 'l2_full_transformer':

        from onmt.modules.SimplifiedTransformer.NMTL2Loss import NMTL2Loss

        encoder = TransformerEncoder(opt, embedding_src, positional_encoder)
        decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding)

        tgt_encoder = TransformerEncoder(opt, embedding_tgt, positional_encoder, share=encoder)

        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())

        model = Transformer(encoder, decoder, generator, tgt_encoder=tgt_encoder)

        loss_function = NMTL2Loss(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)

    elif opt.model == 'parallel_transformer':

        from onmt.modules.ParallelTransformer.Models import ParallelTransformer
        from onmt.modules.ParallelTransformer.NMTL2Loss import NMTL2Loss

        encoder = TransformerEncoder(opt, embedding_src, positional_encoder)
        decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding)

        tgt_encoder = TransformerEncoder(opt, embedding_tgt, positional_encoder, share=encoder)

        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())

        model = ParallelTransformer(encoder, decoder, generator, tgt_encoder=tgt_encoder)

        loss_function = NMTL2Loss(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)

    elif opt.model == 'parallel_transformer_v2':

        from onmt.modules.ParallelTransformerv2.Models import ParallelTransformer
        from onmt.modules.ParallelTransformer.NMTL2Loss import NMTL2Loss

        encoder = TransformerEncoder(opt, embedding_src, positional_encoder)
        decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding)

        tgt_encoder = TransformerEncoder(opt, embedding_tgt, positional_encoder, share=encoder)

        tgt_decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding)

        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())

        model = ParallelTransformer(encoder, decoder, generator, tgt_encoder=tgt_encoder, tgt_decoder=tgt_decoder)

        loss_function = NMTL2Loss(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)
    
    elif opt.model in ['universal_transformer', 'utransformer'] :
    
        from onmt.modules.UniversalTransformer.Models import UniversalTransformerDecoder, UniversalTransformerEncoder
        from onmt.modules.UniversalTransformer.Layers import TimeEncoding
        
        onmt.Constants.weight_norm = opt.weight_norm
        onmt.Constants.init_value = opt.param_init
        
        positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN )
        time_encoder = TimeEncoding(opt.model_size, len_max=32)

        encoder = UniversalTransformerEncoder(opt, embedding_src, positional_encoder, time_encoder)
        decoder = UniversalTransformerDecoder(opt, embedding_tgt, positional_encoder, time_encoder)
        
        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
        
        model = Transformer(encoder, decoder, generator)    

        loss_function = NMTLossFunc(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)

    elif opt.model in ['vtransformer', 'variational_transformer']:

        from onmt.modules.VariationalTransformer.Models import VariationalDecoder, VariationalTransformer
        from onmt.modules.VariationalTransformer.Inference import NeuralPrior, NeuralPosterior

        positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN)

        if opt.var_ignore_source:
            encoder = None
        else:
            encoder = TransformerEncoder(opt, embedding_src, positional_encoder)

        decoder = VariationalDecoder(opt, embedding_tgt, positional_encoder,
                                     encoder_to_share=encoder if opt.share_enc_dec_weights else None)

        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
        prior = NeuralPrior(opt, embedding_src, positional_encoder)
        posterior = NeuralPosterior(opt, embedding_tgt, positional_encoder, prior=prior)

        model = VariationalTransformer(encoder, decoder, prior, posterior, generator,
                                       use_prior_training=opt.var_use_prior_training)

        from onmt.modules.VariationalTransformer.VariationalLoss import VariationalLoss

        loss_function = VariationalLoss(dicts['tgt'].size(), opt)

    elif opt.model in ['recurrent_variational']:

        from onmt.modules.RecurrentVariational.Models import RecurrentVariationalTransformer
        from onmt.modules.RecurrentVariational.Inference import NeuralPrior, NeuralPosterior
        
        positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN)

        # encoder = TransformerEncoder(opt, embedding_src, positional_encoder)

        # note: this model doesn't have encoder
        encoder = None

        decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder)

        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())

        prior = NeuralPrior(opt, embedding_src, positional_encoder)

        if opt.var_sample_from == 'prior':
            posterior = None
        else:
            posterior = NeuralPosterior(opt, embedding_src, embedding_tgt, positional_encoder, prior=prior)

        model = RecurrentVariationalTransformer(encoder, decoder, prior, posterior, generator)

        from onmt.modules.RecurrentVariational.VariationalLoss import VariationalLoss

        loss_function = VariationalLoss(dicts['tgt'].size(), opt)

    elif opt.model in ['deep_vtransformer']:

        from onmt.modules.DeepVariationalTransformer.Models import VariationalDecoder, VariationalTransformer
        from onmt.modules.DeepVariationalTransformer.Inference import NeuralPrior, NeuralPosterior
        from onmt.modules.DeepVariationalTransformer.VariationalLoss import VariationalLoss
        
        positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN)
        
        if opt.var_ignore_source:
            encoder = None
            print("* Model has No Encoder")
        else:
            encoder = TransformerEncoder(opt, embedding_src, positional_encoder)

        decoder = VariationalDecoder(opt, embedding_tgt, positional_encoder, 
                                          encoder_to_share=encoder if opt.share_enc_dec_weights else None)

        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
        prior = NeuralPrior(opt, embedding_src, positional_encoder)

        if opt.var_sample_from == 'prior':
            posterior = None
        else:
            posterior = NeuralPosterior(opt, embedding_tgt, positional_encoder, prior=prior)

        model = VariationalTransformer(encoder, decoder, prior, posterior, generator)

        loss_function = VariationalLoss(dicts['tgt'].size(), opt)

    elif opt.model in ['moe_transformer']:

        from onmt.modules.MixtureModel.Models import MixtureEncoder, MixtureDecoder
        positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN)
        
        encoder = MixtureEncoder(opt, embedding_src, positional_encoder)
        decoder = MixtureDecoder(opt, embedding_tgt, positional_encoder)

        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())

        model = Transformer(encoder, decoder, generator)

        loss_function = NMTLossFunc(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)
    else:
        raise NotImplementedError
        
    if opt.tie_weights:  
        print("* Joining the weights of decoder input and output embeddings")
        model.tie_weights()

    init = torch.nn.init
        
    init.xavier_uniform_(model.generator.linear.weight)
    
    if opt.init_embedding == 'xavier':
        init.xavier_uniform_(model.encoder.word_lut.weight)
        init.xavier_uniform_(model.decoder.word_lut.weight)
    elif opt.init_embedding == 'normal':
        if model.encoder is not None:
            init.normal_(model.encoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)
        init.normal_(model.decoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)

    return model, loss_function


def init_model_parameters(model, opt):
    
    pass

