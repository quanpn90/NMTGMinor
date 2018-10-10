import torch, copy
import torch.nn as nn
from torch.autograd import Variable
import onmt
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder, Transformer
from onmt.modules.Transformer.Layers import PositionalEncoding

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

    
    if opt.model == 'recurrent' or opt.model == 'rnn':
    
        from onmt.modules.rnn.Models import RecurrentEncoder, RecurrentDecoder, RecurrentModel 

        encoder = RecurrentEncoder(opt, dicts['src'])

        decoder = RecurrentDecoder(opt, dicts['tgt'])
        
        generator = onmt.modules.BaseModel.Generator(opt.rnn_size, dicts['tgt'].size())
        
        model = RecurrentModel(encoder, decoder, generator)    
        
    elif opt.model == 'transformer':
        # raise NotImplementedError
        
        onmt.Constants.init_value = opt.param_init
        
        if opt.time == 'positional_encoding':
            positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN)
        else:
            positional_encoder = None
        
        encoder = TransformerEncoder(opt, dicts['src'], positional_encoder)
        decoder = TransformerDecoder(opt, dicts['tgt'], positional_encoder)
        
        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
        
        model = Transformer(encoder, decoder, generator)    
        
        #~ print(encoder)
        
    elif opt.model == 'stochastic_transformer':
        
        from onmt.modules.StochasticTransformer.Models import StochasticTransformerEncoder, StochasticTransformerDecoder
        
        onmt.Constants.weight_norm = opt.weight_norm
        onmt.Constants.init_value = opt.param_init
        
        
        
        positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN)
        #~ positional_encoder = None
        
        encoder = StochasticTransformerEncoder(opt, dicts['src'], positional_encoder)
        
        decoder = StochasticTransformerDecoder(opt, dicts['tgt'], positional_encoder)
        
        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
        
        model = Transformer(encoder, decoder, generator)       
        
        
        
    elif opt.model == 'fctransformer':
    
        from onmt.modules.FCTransformer.Models import FCTransformerEncoder, FCTransformerDecoder
        
        onmt.Constants.weight_norm = opt.weight_norm
        onmt.Constants.init_value = opt.param_init
        
        positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN )
        
        encoder = FCTransformerEncoder(opt, dicts['src'], positional_encoder)
        decoder = FCTransformerDecoder(opt, dicts['tgt'], positional_encoder)
        
        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
        
        model = Transformer(encoder, decoder, generator)    
    elif opt.model == 'ptransformer':
    
        from onmt.modules.ParallelTransformer.Models import ParallelTransformerEncoder, ParallelTransformerDecoder
        
        onmt.Constants.weight_norm = opt.weight_norm
        onmt.Constants.init_value = opt.param_init
        
        positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN )
        
        encoder = ParallelTransformerEncoder(opt, dicts['src'], positional_encoder)
        decoder = ParallelTransformerDecoder(opt, dicts['tgt'], positional_encoder)
        
        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
        
        model = Transformer(encoder, decoder, generator)    
    
    elif opt.model in ['universal_transformer', 'utransformer'] :
    
        from onmt.modules.UniversalTransformer.Models import UniversalTransformerDecoder, UniversalTransformerEncoder
        from onmt.modules.UniversalTransformer.Layers import TimeEncoding
        
        onmt.Constants.weight_norm = opt.weight_norm
        onmt.Constants.init_value = opt.param_init
        
        positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN )
        time_encoder = TimeEncoding(opt.model_size, len_max=32)

        
        encoder = UniversalTransformerEncoder(opt, dicts['src'], positional_encoder, time_encoder)
        decoder = UniversalTransformerDecoder(opt, dicts['tgt'], positional_encoder, time_encoder)
        
        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
        
        model = Transformer(encoder, decoder, generator)    

    elif opt.model in ['iid_stochastic_transformer'] :
    
        from onmt.modules.IIDStochasticTransformer.Models import IIDStochasticTransformerEncoder, IIDStochasticTransformerDecoder
                
        onmt.Constants.weight_norm = opt.weight_norm
        onmt.Constants.init_value = opt.param_init
        
        positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN)
        #~ positional_encoder = None
        
        encoder = IIDStochasticTransformerEncoder(opt, dicts['src'], positional_encoder)
        
        decoder = IIDStochasticTransformerDecoder(opt, dicts['tgt'], positional_encoder)
        
        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
        
        model = Transformer(encoder, decoder, generator)       
        
    elif opt.model in ['reinforce_transformer'] :
    
        
        from onmt.modules.StochasticTransformer.Models import StochasticTransformerEncoder, StochasticTransformerDecoder
        from onmt.modules.ReinforceTransformer.Models import ReinforcedStochasticDecoder, ReinforceTransformer
                
        onmt.Constants.weight_norm = opt.weight_norm
        onmt.Constants.init_value = opt.param_init
        
        positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN)
        
        encoder = StochasticTransformerEncoder(opt, dicts['src'], positional_encoder)
        
        decoder = ReinforceStochasticTransformerDecoder(opt, dicts['tgt'], positional_encoder)
        
        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
        
        model = ReinforceTransformer(encoder, decoder, generator)       


    else:
        raise NotImplementedError
        
    if opt.tie_weights:  
        print("Joining the weights of decoder input and output embeddings")
        model.tie_weights()
       
    if opt.join_embedding:
        print("Joining the weights of encoder and decoder word embeddings")
        model.share_enc_dec_embedding()
        
    init = torch.nn.init
        
    init.xavier_uniform_(model.generator.linear.weight)
    
    if opt.init_embedding == 'xavier':
        init.xavier_uniform_(model.encoder.word_lut.weight)
        init.xavier_uniform_(model.decoder.word_lut.weight)
    elif opt.init_embedding == 'normal':
        init.normal_(model.encoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)
        init.normal_(model.decoder.word_lut.weight, mean=0, std=opt.model_size ** -0.5)
            
    
    return model
    
def init_model_parameters(model, opt):
    
    if opt.model == 'recurrent':
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

