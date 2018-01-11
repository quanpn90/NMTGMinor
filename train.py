from __future__ import division

import onmt
import onmt.Markdown
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
from onmt.train_utils.trainer import XETrainer
from onmt.modules.Loss import NMTLossFunc
from onmt.ModelConstructor import build_model, init_model_parameters

parser = argparse.ArgumentParser(description='train.py')
onmt.Markdown.add_md_help_argument(parser)

# Data options

parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-load_from', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")
parser.add_argument('-model', default='recurrent',
                    help="Optimization method. [recurrent|transformer]")
parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')                   
# Recurrent Model options
parser.add_argument('-rnn_size', type=int, default=512,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=512,
                    help='Word embedding sizes')
parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
parser.add_argument('-brnn', action='store_true',
                    help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")

# Transforer Model options
parser.add_argument('-model_size', type=int, default=512,
    help='Size of embedding / transformer hidden')      
parser.add_argument('-inner_size', type=int, default=2048,
    help='Size of inner feed forward layer')  
parser.add_argument('-n_heads', type=int, default=8,
    help='Number of heads for multi-head attention') 
parser.add_argument('-attn_dropout', type=float, default=0.1,
                    help='Dropout probability; applied between LSTM stacks.')    
# Optimization options
parser.add_argument('-encoder_type', default='text',
                    help="Type of encoder to use. Options are [text|img].")
parser.add_argument('-batch_size', type=int, default=1024,
                    help='Maximum batch size')
parser.add_argument('-max_seq_num', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=32,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=13,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-optim', default='adam',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-max_grad_norm', type=float, default=0,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-word_dropout', type=float, default=0.0,
                    help='Dropout probability; applied on embedding indices.')
parser.add_argument('-label_smoothing', type=float, default=0.0,
                    help='Label smoothing value for loss functions.')
parser.add_argument('-scheduled_sampling_rate', type=float, default=0.0,
                    help='Scheduled sampling rate.')
parser.add_argument('-curriculum', type=int, default=-1,
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-extra_shuffle', action="store_true",
                    help="""By default only shuffle mini-batch order; when true,
                    shuffle and re-assign mini-batches""")
parser.add_argument('-virtual_gpu', type=int, default=1,
                    help='Number of virtual gpus. The trainer will try to mimic asynchronous multi-gpu training')
# learning rate
parser.add_argument('-learning_rate', type=float, default=1.0,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1,
                    adadelta = 1, adam = 0.001""")
parser.add_argument('-learning_rate_decay', type=float, default=1,
                    help="""If update_learning_rate, decay learning rate by
                    this much if (i) perplexity does not decrease on the
                    validation set or (ii) epoch has gone past
                    start_decay_at""")
parser.add_argument('-start_decay_at', type=int, default=99999,
                    help="""Start decaying every epoch after and including this
                    epoch""")
parser.add_argument('-warmup_steps', type=int, default=4096,
                    help="""Start decaying every epoch after and including this
                    epoch""")
parser.add_argument('-beta1', type=float, default=0.9,
                    help="""beta_1 value for adam""")
parser.add_argument('-beta2', type=float, default=0.98,
                    help="""beta_2 value for adam""")
                    
# pretrained word vectors
parser.add_argument('-tie_weights', action='store_true',
                    help='Tie the weights of the encoder and decoder layer')
parser.add_argument('-join_embedding', action='store_true',
                    help='Jointly train the embedding of encoder and decoder in one weight')
parser.add_argument('-pre_word_vecs_enc',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")

# GPU
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-seed', default=9999, nargs='+', type=int,
                    help="Seed for deterministic runs.")

parser.add_argument('-log_interval', type=int, default=100,
                    help="Print stats at this interval.")
parser.add_argument('-save_every', type=int, default=-1,
                    help="Save every this interval.")

opt = parser.parse_args()

print(opt)

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, should run with -gpus 0")

if opt.gpus:
    cuda.set_device(opt.gpus[0])

torch.manual_seed(opt.seed)


def main():
    print("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)
    print("Done")
    dict_checkpoint = opt.load_from 
    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        dataset['dicts'] = checkpoint['dicts']

    trainData = onmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, opt.gpus,
                             data_type=dataset.get("type", "text"), max_seq_num=opt.max_seq_num)
    validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.batch_size, opt.gpus,
                             volatile=True,
                             data_type=dataset.get("type", "text"), max_seq_num=128)

    dicts = dataset['dicts']
    print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    print(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    print(' * maximum batch size (words per batch). %d' % opt.batch_size)

    print('Building model...')
    model = build_model(opt, dicts)

    if opt.load_from:
        print('Loading model from checkpoint at %s'
              % opt.load_from)
        model.load_state_dict(checkpoint['model'])
        opt.start_epoch = checkpoint['epoch'] + 1

    if len(opt.gpus) >= 1:
        model.cuda()
    else:
        model.cpu()
    
    

    if not opt.load_from:
        
        init_model_parameters(model, opt)

        optim = onmt.NoamOptim(opt)
    else:
        print('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
        print(optim)
        
    loss_function = NMTLossFunc(dataset['dicts']['tgt'].size(), 
                                        label_smoothing=opt.label_smoothing,
                                        shard_size=opt.max_generator_batches)
    
    if len(opt.gpus) >= 1:
        loss_function = loss_function.cuda()
    
    optim.set_parameters(model.parameters())
    
    print(model)
    print(loss_function)

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)
    
    trainer = XETrainer(model, loss_function, trainData, validData, dataset, optim, opt)
    
    trainer.run()
        
    #~ trainModel(model, trainData, validData, dataset, optim)


if __name__ == "__main__":
    main()
