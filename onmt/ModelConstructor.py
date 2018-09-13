import torch, copy
import torch.nn as nn
from torch.autograd import Variable
import onmt
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder, Transformer
from onmt.modules.Transformer.Layers import PositionalEncoding



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
    
    onmt.Constants.layer_norm = opt.layer_norm
    onmt.Constants.weight_norm = opt.weight_norm
    onmt.Constants.activation_layer = opt.activation_layer
    onmt.Constants.version = 1.0
    onmt.Constants.attention_out = opt.attention_out
    onmt.Constants.residual_type = opt.residual_type
    

    
    if opt.model == 'recurrent' or opt.model == 'rnn':
    
        from onmt.modules.rnn.Models import RecurrentEncoder, RecurrentDecoder, RecurrentModel 

        encoder = RecurrentEncoder(opt, dicts['src'])

        decoder = RecurrentDecoder(opt, dicts['tgt'])
        
        generator = onmt.modules.BaseModel.Generator(opt.rnn_size, dicts['tgt'].size())
        
        model = RecurrentModel(encoder, decoder, generator)    
        
    elif opt.model == 'transformer':
        # raise NotImplementedError
        
        max_size = 262 # This should be the longest sentence from the dataset
        onmt.Constants.init_value = opt.param_init
        
        if opt.time == 'positional_encoding':
            positional_encoder = PositionalEncoding(opt.model_size, len_max=max_size)
        else:
            positional_encoder = None
        #~ elif opt.time == 'gru':
            #~ positional_encoder = nn.GRU(opt.model_size, opt.model_size, 1, batch_first=True)
        #~ elif opt.time == 'lstm':
            #~ positional_encoder = nn.LSTM(opt.model_size, opt.model_size, 1, batch_first=True)
        
        encoder = TransformerEncoder(opt, dicts['src'], positional_encoder)
        decoder = TransformerDecoder(opt, dicts['tgt'], positional_encoder)
        
        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
        
        model = Transformer(encoder, decoder, generator)    
        
        #~ print(encoder)
        
    elif opt.model == 'stochastic_transformer':
        
        from onmt.modules.StochasticTransformer.Models import StochasticTransformerEncoder, StochasticTransformerDecoder

        
        max_size = 256 # This should be the longest sentence from the dataset
        onmt.Constants.weight_norm = opt.weight_norm
        onmt.Constants.init_value = opt.param_init
        
        
        
        positional_encoder = PositionalEncoding(opt.model_size, len_max=max_size)
        #~ positional_encoder = None
        
        encoder = StochasticTransformerEncoder(opt, dicts['src'], positional_encoder)
        
        decoder = StochasticTransformerDecoder(opt, dicts['tgt'], positional_encoder)
        
        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
        
        model = Transformer(encoder, decoder, generator)       
        
        
        
    elif opt.model == 'fctransformer':
    
        from onmt.modules.FCTransformer.Models import FCTransformerEncoder, FCTransformerDecoder
        
        max_size = 256 # This should be the longest sentence from the dataset
        onmt.Constants.weight_norm = opt.weight_norm
        onmt.Constants.init_value = opt.param_init
        
        positional_encoder = PositionalEncoding(opt.model_size, len_max=max_size)
        
        encoder = FCTransformerEncoder(opt, dicts['src'], positional_encoder)
        decoder = FCTransformerDecoder(opt, dicts['tgt'], positional_encoder)
        
        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
        
        model = Transformer(encoder, decoder, generator)    
    elif opt.model == 'ptransformer':
    
        from onmt.modules.ParallelTransformer.Models import ParallelTransformerEncoder, ParallelTransformerDecoder
        
        max_size = 256 # This should be the longest sentence from the dataset
        onmt.Constants.weight_norm = opt.weight_norm
        onmt.Constants.init_value = opt.param_init
        
        positional_encoder = PositionalEncoding(opt.model_size, len_max=max_size)
        
        encoder = ParallelTransformerEncoder(opt, dicts['src'], positional_encoder)
        decoder = ParallelTransformerDecoder(opt, dicts['tgt'], positional_encoder)
        
        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
        
        model = Transformer(encoder, decoder, generator)    
    
    elif opt.model in ['universal_transformer', 'utransformer'] :
    
        #~ from onmt.modules.ParallelTransformer.Models import ParallelTransformerEncoder, ParallelTransformerDecoder
        from onmt.modules.UniversalTransformer.Models import UniversalTransformerDecoder, UniversalTransformerEncoder
        from onmt.modules.UniversalTransformer.Layers import TimeEncoding
        
        max_size = 256 # This should be the longest sentence from the dataset
        onmt.Constants.weight_norm = opt.weight_norm
        onmt.Constants.init_value = opt.param_init
        
        positional_encoder = PositionalEncoding(opt.model_size, len_max=max_size)
        time_encoder = TimeEncoding(opt.model_size, len_max=32)

        
        encoder = UniversalTransformerEncoder(opt, dicts['src'], positional_encoder, time_encoder)
        decoder = UniversalTransformerDecoder(opt, dicts['tgt'], positional_encoder, time_encoder)
        
        generator = onmt.modules.BaseModel.Generator(opt.model_size, dicts['tgt'].size())
        
        model = Transformer(encoder, decoder, generator)    
        
        
        #~ print(encoder)

    else:
        raise NotImplementedError
        
        #~ 
    #~ init = torch.nn.init
        #~ 
    #~ init.xavier_uniform(model.encoder.word_lut.weight)
    #~ init.xavier_uniform(model.decoder.word_lut.weight)
        
     # Weight tying between decoder input and output embedding:
    if opt.tie_weights:  
        print("Joining the weights of decoder input and output embeddings")
        model.tie_weights()
       
    if opt.join_embedding:
        print("Joining the weights of encoder and decoder word embeddings")
        model.share_enc_dec_embedding()
    
    return model
    
def init_model_parameters(model, opt):
    
    if opt.model == 'recurrent':
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)
    else:
        from onmt.modules.Transformer.Layers import uniform_unit_scaling

        # We initialize the model parameters with Xavier init
        #~ init = torch.nn.init
        
        #~ init.xavier_uniform_(model.generator.linear.weight)
        #~ init.xavier_uniform_(model.encoder.word_lut.weight.data)
        #~ init.xavier_uniform_(model.decoder.word_lut.weight.data)
        
        #~ init = torch.nn.init.uniform
        #~ 
        #~ init(model.generator.linear.weight, -onmt.Constants.init_value, 
                                             #~ onmt.Constants.init_value)
        #~ init(model.encoder.word_lut.weight, -onmt.Constants.init_value, 
                                             #~ onmt.Constants.init_value)
        #~ init(model.decoder.word_lut.weight, -onmt.Constants.init_value, 
                                             #~ onmt.Constants.init_value)
#~ def build_reconstructor(opt, dicts, positional_encoder):
    #~ decoder = TransformerDecoder(opt, dicts['src'], positional_encoder)
    
    #~ reconstructor = 
