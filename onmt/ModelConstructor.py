import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder, Transformer
from onmt.modules.Transformer.Layers import PositionalEncoding

def build_model(opt, dicts):

    model = None
    
    if not hasattr(opt, 'model'):
        opt.model = 'recurrent'

    
    if opt.model == 'recurrent' or opt.model == 'rnn':
    
        from onmt.modules.rnn.Models import RecurrentEncoder, RecurrentDecoder, RecurrentModel 

        encoder = RecurrentEncoder(opt, dicts['src'])

        decoder = RecurrentDecoder(opt, dicts['tgt'])
        
        generator = onmt.modules.BaseModel.Generator(opt.rnn_size, dicts['tgt'].size())
        
        model = RecurrentModel(encoder, decoder, generator)    
        
    elif opt.model == 'transformer':
        # raise NotImplementedError
        
        max_size = 256 # This should be the longest sentence from the dataset
        onmt.Constants.weight_norm = opt.weight_norm
        onmt.Constants.init_value = opt.param_init
        
        
        positional_encoder = PositionalEncoding(opt.model_size, len_max=max_size)
        
        encoder = TransformerEncoder(opt, dicts['src'], positional_encoder)
        
        decoder = TransformerDecoder(opt, dicts['tgt'], positional_encoder)
        
        generator = onmt.modules.BaseModel.Generator(opt.rnn_size, dicts['tgt'].size())
        
        model = Transformer(encoder, decoder, generator)    
        
    elif opt.model == 'stochastic_transformer':
        
        from onmt.modules.StochasticTransformer.Models import StochasticTransformerEncoder, StochasticTransformerDecoder

        
        max_size = 256 # This should be the longest sentence from the dataset
        onmt.Constants.weight_norm = opt.weight_norm
        onmt.Constants.init_value = opt.param_init
        
        
        
        positional_encoder = PositionalEncoding(opt.model_size, len_max=max_size)
        
        encoder = StochasticTransformerEncoder(opt, dicts['src'], positional_encoder)
        
        decoder = StochasticTransformerDecoder(opt, dicts['tgt'], positional_encoder)
        
        generator = onmt.modules.BaseModel.Generator(opt.rnn_size, dicts['tgt'].size())
        
        model = Transformer(encoder, decoder, generator)        
        
    else:
        raise NotImplementedError
        
    #~ model.generator = generator 
        
     # Weight tying between decoder input and output embedding:
    if opt.tie_weights:  
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
        init = torch.nn.init.xavier_uniform
        #~ init(model.encoder.word_lut.weight)
        #~ init(model.decoder.word_lut.weight)
        uniform_unit_scaling(model.encoder.word_lut.weight.data)
        uniform_unit_scaling(model.decoder.word_lut.weight.data)
        #~ 
        #~ init = torch.nn.init.uniform
        #~ 
        #~ init(model.generator.linear.weight, -onmt.Constants.init_value, 
                                             #~ onmt.Constants.init_value)
        #~ init(model.encoder.word_lut.weight, -onmt.Constants.init_value, 
                                             #~ onmt.Constants.init_value)
        #~ init(model.decoder.word_lut.weight, -onmt.Constants.init_value, 
                                             #~ onmt.Constants.init_value)
def build_reconstructor(opt, dicts, positional_encoder):
    decoder = TransformerDecoder(opt, dicts['src'], positional_encoder)
    
    #~ reconstructor = 
