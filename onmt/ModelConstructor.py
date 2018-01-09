import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt


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
        
        max_size = 256
        
        from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder, Transformer
        from onmt.modules.Transformer.Layers import PositionalEncoding
        
        positional_encoder = PositionalEncoding(opt.model_size, len_max=max_size)
        
        encoder = TransformerEncoder(opt, dicts['src'], positional_encoder)
        
        decoder = TransformerDecoder(opt, dicts['tgt'], positional_encoder)
        
        generator = onmt.modules.BaseModel.Generator(opt.rnn_size, dicts['tgt'].size())
        
        model = Transformer(encoder, decoder, generator)    
        
        
    else:
        raise NotImplementedError
        
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
        # We initialize the model parameters with Xavier init
        xavier = torch.nn.init.xavier_uniform
        xavier(model.generator.linear.weight)
        xavier(model.encoder.word_lut.weight)
        xavier(model.decoder.word_lut.weight)
