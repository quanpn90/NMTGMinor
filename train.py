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
import time, datetime
from onmt.train_utils.trainer import XETrainer
from onmt.train_utils.fp16_trainer import FP16XETrainer
from onmt.train_utils.multiGPUtrainer import MultiGPUXETrainer
from onmt.train_utils.fp16_vi_trainer import VariationalTrainer
from onmt.modules.VariationalLoss import VariationalLoss
from onmt.ModelConstructor import build_model, init_model_parameters

parser = argparse.ArgumentParser(description='train.py')
onmt.Markdown.add_md_help_argument(parser)

from options import make_parser
# Please look at the options file to see the options regarding models and data
parser = make_parser(parser)

opt = parser.parse_args()

print(opt)

# An ugly hack to have weight norm on / off
onmt.Constants.weight_norm = opt.weight_norm
onmt.Constants.checkpointing = opt.checkpointing
onmt.Constants.max_position_length = opt.max_position_length

# Use static dropout if checkpointing > 0
if opt.checkpointing > 0:
    onmt.Constants.static = True

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, should run with -gpus 0")
    




torch.manual_seed(opt.seed)


def main():
    
    
    
    start = time.time()
    print("Loading data from '%s'" % opt.data)
    
    if opt.data_format == 'raw':
        dataset = torch.load(opt.data)
        elapse = str(datetime.timedelta(seconds=int(time.time() - start)))
        print("Done after %s" % elapse )
      

        trainData = onmt.Dataset(dataset['train']['src'],
                                 dataset['train']['tgt'], opt.batch_size_words, opt.gpus,
                                 max_seq_num=opt.batch_size_sents,
                                 pad_count = opt.pad_count,
                                 multiplier = opt.batch_size_multiplier,
                                 sort_by_target=opt.sort_by_target)
        validData = onmt.Dataset(dataset['valid']['src'],
                                 dataset['valid']['tgt'], opt.batch_size_words, opt.gpus,
                                 max_seq_num=opt.batch_size_sents)

        dicts = dataset['dicts']
        print(' * vocabulary size. source = %d; target = %d' %
              (dicts['src'].size(), dicts['tgt'].size()))
        print(' * number of training sentences. %d' %
              len(dataset['train']['src']))
        print(' * maximum batch size (words per batch). %d' % opt.batch_size_words)
    elif opt.data_format == 'bin':
        from onmt.data_utils.IndexedDataset import IndexedInMemoryDataset

        dicts = torch.load(opt.data + ".dict.pt")
        
        #~ train = {}
        train_path = opt.data + '.train'
        train_src = IndexedInMemoryDataset(train_path + '.src')
        train_tgt = IndexedInMemoryDataset(train_path + '.tgt')
        
        trainData = onmt.Dataset(train_src,
                                 train_tgt, opt.batch_size_words, opt.gpus,
                                 max_seq_num=opt.batch_size_sents,
                                 pad_count = opt.pad_count,
                                 multiplier = opt.batch_size_multiplier,
                                 sort_by_target=opt.sort_by_target)
                                 
        valid_path = opt.data + '.valid'
        valid_src = IndexedInMemoryDataset(valid_path + '.src')
        valid_tgt = IndexedInMemoryDataset(valid_path + '.tgt')
        
        validData = onmt.Dataset(valid_src,
                                 valid_tgt, opt.batch_size_words, opt.gpus,
                                 max_seq_num=opt.batch_size_sents)
        
        print(' * vocabulary size. source = %d; target = %d' %
              (dicts['src'].size(), dicts['tgt'].size()))
        print(' * number of training sentences. %d' %
              len(train_src))
        print(' * maximum batch size (words per batch). %d' % opt.batch_size_words)
    
    else:
        raise NotImplementedError
    
    print('Building model...')

    # Loss function is built according to model 
    # Go to ModelConstructor for more details
    model, loss_function = build_model(opt, dicts)
    
    
    
                                
    
    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)
    
    optim = None
    
    if len(opt.gpus) > 1 or opt.virtual_gpu > 1:
        #~ trainer = MultiGPUXETrainer(model, loss_function, trainData, validData, dataset, opt)
        raise NotImplementedError("Multi-GPU training is not supported atm")
    else:
        if opt.fp16:
            trainer = FP16XETrainer(model, loss_function, trainData, validData, dicts, opt)
        else:
            trainer = XETrainer(model, loss_function, trainData, validData, dicts, opt)

    
    trainer.run(save_file=opt.load_from)

if __name__ == "__main__":
    main()
