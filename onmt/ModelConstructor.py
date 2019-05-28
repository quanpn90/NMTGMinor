import torch, copy
import torch.nn as nn
from torch.autograd import Variable
import onmt
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder, Transformer
from onmt.modules.Transformer.Layers import PositionalEncoding
from onmt.modules.Loss import NMTLossFunc
from onmt.modules.MultilingualLoss import MSEAttnLoss, MSEDecoderLoss, MSEEncoderLoss, KLSoftmaxLoss


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

    if not hasattr(opt, 'copy_generator'):
        opt.copy_generator = False

    if not hasattr(opt, 'fixed_target_length'):
        opt.fixed_target_length = 'no'

    if not hasattr(opt, 'loss_function'):

        if opt.model in ['transformer', 'simplified_transformer', 'simplified_transformer_v2']:
            opt.loss_function = 0
        elif opt.model in ['l2_transformer', 'l2_simplified_transformer_v2']:
            opt.loss_function = 1
        elif opt.model in ['parallel_transformer', 'parallel_transformer_v2']:
            opt.loss_function = 2
        elif opt.model in ['parallel_attention_transformer']:
            opt.loss_function = 3
        elif opt.model in ['parallel_softmax_transformer']:
            opt.loss_function = 4

    return opt


def build_model(opt, dicts):

    # for loading models
    # the old models might not have the options
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
        print("* Joining the weights of encoder and decoder word embeddings")
    else:
        embedding_tgt = nn.Embedding(dicts['tgt'].size(),
                                     opt.model_size,
                                     padding_idx=onmt.Constants.PAD)
    #
    feat_embedding = nn.Embedding(dicts['atb'].size(), opt.model_size)
    #
    positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN)

    if opt.copy_generator:

        from onmt.modules.CopyGenerator import  CopyGenerator
        generator = CopyGenerator(opt.model_size, dicts['tgt'].size())

    else:
        from onmt.modules.BaseModel import Generator
        generator = Generator(opt.model_size, dicts['tgt'].size())

    if opt.model == 'transformer':

        encoder = TransformerEncoder(opt, embedding_src, positional_encoder)
        decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding,
                                     encoder_to_share=encoder if opt.share_enc_dec_weights else None)


        model = Transformer(encoder, decoder, generator)

    elif opt.model == 'stochastic_transformer':
        """
        The stochastic implementation of the Transformer as in 
        "Very Deep Self-Attention Networks for End-to-End Speech Recognition"
        """
        encoder = TransformerEncoder(opt, embedding_src, positional_encoder,
                                     stochastic=True)
        decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding,
                                     encoder_to_share=encoder if opt.share_enc_dec_weights else None,
                                     stochastic=True)

        model = Transformer(encoder, decoder, generator)

        model = Transformer(encoder, decoder, generator)

    elif opt.model == 'simplified_transformer' or opt.model == 'l2_transformer':

        from onmt.modules.SimplifiedTransformer.Models import SimplifiedTransformerEncoder, SimplifiedTransformer

        encoder = SimplifiedTransformerEncoder(opt, embedding_src, positional_encoder)

        decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding)

        tgt_encoder = SimplifiedTransformerEncoder(opt, embedding_tgt, positional_encoder, share=encoder) \
                            if opt.model == 'l2_transformer' else None

        model = SimplifiedTransformer(encoder, decoder, generator, tgt_encoder=tgt_encoder)

    elif opt.model == 'simplified_transformer_v2' or opt.model == 'l2_simplified_transformer_v2' :

        from onmt.modules.CompressedTransformer.Models import CompressedTransformerEncoder
        from onmt.modules.SimplifiedTransformer.Models import SimplifiedTransformer, ParallelSimplifiedTransformer

        encoder = CompressedTransformerEncoder(opt, embedding_src, positional_encoder)

        decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding)

        # tgt_encoder = CompressedTransformerEncoder(opt, embedding_tgt, positional_encoder, share=encoder) \
        #     if opt.model == 'l2_simplified_transformer_v2' else None
        if opt.loss_function == 0:
            tgt_encoder = None

            model = SimplifiedTransformer(encoder, decoder, generator, tgt_encoder=tgt_encoder)

        else:
            tgt_encoder = CompressedTransformerEncoder(opt, embedding_tgt, positional_encoder, share=encoder)

            if opt.loss_function==2:
                tgt_decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding)

                model = ParallelSimplifiedTransformer(encoder, decoder, generator, tgt_encoder=tgt_encoder,
                                                      tgt_decoder=tgt_decoder)
            elif opt.loss_function==1:
                model = SimplifiedTransformer(encoder, decoder, generator, tgt_encoder=tgt_encoder)

            else:
                raise NotImplementedError


    elif opt.model == 'parallel_attention_transformer':

        from onmt.modules.ParallelAttentionTransformer.Models import ParallelAttentionTransformer
        from onmt.modules.ParallelAttentionTransformer.NMTL2Loss import NMTL2Loss

        encoder = TransformerEncoder(opt, embedding_src, positional_encoder)
        decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding)

        tgt_encoder = TransformerEncoder(opt, embedding_tgt, positional_encoder, share=encoder)

        tgt_decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding)

        model = ParallelAttentionTransformer(encoder, decoder, generator, tgt_encoder=tgt_encoder, tgt_decoder=tgt_decoder)

    # this is 'probably' the model that uses normalization like Google
    elif opt.model == 'l2_full_transformer':

        from onmt.modules.SimplifiedTransformer.NMTL2Loss import NMTL2Loss

        encoder = TransformerEncoder(opt, embedding_src, positional_encoder)
        decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding)

        tgt_encoder = TransformerEncoder(opt, embedding_tgt, positional_encoder, share=encoder)

        model = Transformer(encoder, decoder, generator, tgt_encoder=tgt_encoder)

    elif opt.model == 'parallel_transformer':

        # from onmt.modules.ParallelTransformer.Models import ParallelTransformer
        # from onmt.modules.ParallelTransformer.NMTL2Loss import NMTL2Loss
        #
        # encoder = TransformerEncoder(opt, embedding_src, positional_encoder)
        # decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding)
        #
        # tgt_encoder = TransformerEncoder(opt, embedding_tgt, positional_encoder, share=encoder)
        #
        # generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
        #
        # model = ParallelTransformer(encoder, decoder, generator, tgt_encoder=tgt_encoder)
        raise NotImplementedError

    elif opt.model == 'parallel_transformer_v2':

        from onmt.modules.ParallelTransformerv2.Models import ParallelTransformer
        from onmt.modules.ParallelTransformer.NMTL2Loss import NMTL2Loss

        encoder = TransformerEncoder(opt, embedding_src, positional_encoder)
        decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding)

        tgt_encoder = TransformerEncoder(opt, embedding_tgt, positional_encoder, share=encoder)

        tgt_decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding)

        model = ParallelTransformer(encoder, decoder, generator, tgt_encoder=tgt_encoder, tgt_decoder=tgt_decoder)

    elif opt.model == 'parallel_softmax_transformer':

        from onmt.modules.ParallelSoftmax.Models import ParallelTransformer
        from onmt.modules.ParallelSoftmax.NMTL2Loss import NMTL2Loss

        encoder = TransformerEncoder(opt, embedding_src, positional_encoder)
        decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding)

        tgt_encoder = TransformerEncoder(opt, embedding_tgt, positional_encoder, share=encoder)

        tgt_decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding)

        model = ParallelTransformer(encoder, decoder, generator, tgt_encoder=tgt_encoder, tgt_decoder=tgt_decoder)

    elif opt.model == 'parallel_simplified_transformer':

        from onmt.modules.SimplifiedTransformer.Models import SimplifiedTransformerEncoder, ParallelSimplifiedTransformer

        encoder = SimplifiedTransformerEncoder(opt, embedding_src, positional_encoder)

        decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding)

        tgt_encoder = SimplifiedTransformerEncoder(opt, embedding_tgt, positional_encoder, share=encoder)

        tgt_decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, feat_embedding)

        model = ParallelSimplifiedTransformer(encoder, decoder, generator, tgt_encoder=tgt_encoder, tgt_decoder=tgt_decoder)

    else:
        print(opt.model)
        raise NotImplementedError

    # now we have to build the loss function

    if opt.loss_function == 0:
        loss_function = NMTLossFunc(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)
    elif opt.loss_function == 1:  # L2 encoder
        loss_function = MSEEncoderLoss(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)
    elif opt.loss_function == 2:  # L2 decoder
        loss_function = MSEDecoderLoss(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)
    elif opt.loss_function == 3:  # L2 attn
        loss_function = MSEAttnLoss(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)
    elif opt.loss_function == 4:  # L2 Softmax
        loss_function = KLSoftmaxLoss(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)
    else:
        loss_function = None
        
    if opt.tie_weights:  
        print("* Joining the weights of decoder input and output embeddings")
        model.tie_weights()

    init = torch.nn.init
        
    # init.xavier_uniform_(model.generator.linear.weight)
    
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

