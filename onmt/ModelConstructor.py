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
        
    return opt

def build_model(opt, dicts):

    model = None
    
    opt = update_backward_compatibility(opt)
    
    onmt.Constants.layer_norm = opt.layer_norm
    onmt.Constants.weight_norm = opt.weight_norm
    onmt.Constants.activation_layer = opt.activation_layer
    onmt.Constants.version = 1.0
    onmt.Constants.attention_out = opt.attention_out
    onmt.Constants.residual_type = opt.residual_type
    
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
        # raise NotImplementedError
        
        onmt.Constants.init_value = opt.param_init
        
        if opt.time == 'positional_encoding':
            positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN)
        else:
            positional_encoder = None
        
        encoder = TransformerEncoder(opt, embedding_src, positional_encoder)
        decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, encoder_to_share=encoder if opt.share_enc_dec_weights else None)
        
        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
        
        model = Transformer(encoder, decoder, generator)    

        loss_function = NMTLossFunc(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)

                
    elif opt.model == 'stochastic_transformer':
        
        from onmt.modules.StochasticTransformer.Models import StochasticTransformerEncoder, StochasticTransformerDecoder
        
        onmt.Constants.weight_norm = opt.weight_norm
        onmt.Constants.init_value = opt.param_init
        
        
        
        positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN)
        
        encoder = StochasticTransformerEncoder(opt, embedding_src, positional_encoder)
        
        decoder = StochasticTransformerDecoder(opt, dicts['tgt'], positional_encoder)
        
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

        loss_function = NMTLossFunc(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)
    elif opt.model == 'ptransformer':
    
        from onmt.modules.ParallelTransformer.Models import ParallelTransformerEncoder, ParallelTransformerDecoder
        
        onmt.Constants.weight_norm = opt.weight_norm
        onmt.Constants.init_value = opt.param_init
        
        positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN )
        
        encoder = ParallelTransformerEncoder(opt, embedding_src, positional_encoder)
        decoder = ParallelTransformerDecoder(opt, embedding_tgt, positional_encoder)
        
        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
        
        model = Transformer(encoder, decoder, generator)   

        loss_function = NMTLossFunc(dicts['tgt'].size(), label_smoothing=opt.label_smoothing) 
    
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
        
        encoder = TransformerEncoder(opt, embedding_src, positional_encoder)
        decoder = VariationalDecoder(opt, embedding_tgt, positional_encoder)

        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
        prior = NeuralPrior(opt, embedding_src, positional_encoder)
        posterior = NeuralPosterior(opt, embedding_tgt, positional_encoder)

        model = VariationalTransformer(encoder, decoder, prior, posterior, generator)

        from onmt.modules.VariationalTransformer.VariationalLoss import VariationalLoss

        loss_function = VariationalLoss(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)

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
        init.normal_(model.encoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)
        init.normal_(model.decoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)
    
   
    return model, loss_function
    
def init_model_parameters(model, opt):
    
    if opt.model == 'recurrent':
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

