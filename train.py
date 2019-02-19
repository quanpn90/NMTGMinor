# noinspection PyInterpreter
from __future__ import division

import onmt
import onmt.Markdown
# noinspection PyInterpreter
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
from onmt.ModelConstructor import build_model, init_model_parameters

parser = argparse.ArgumentParser(description='train.py')
onmt.Markdown.add_md_help_argument(parser)

from options import make_parser
# Please look at the options file to see the options regarding models and data
parser = make_parser(parser)

opt = parser.parse_args()

print(torch.__version__)
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
    print("Locally in debugging mode")
    print("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)
    elapse = str(datetime.timedelta(seconds=int(time.time() - start)))
    print("Done after %s" % elapse)

    train_data = onmt.Dataset(dataset['train']['src'],
                              dataset['train']['tgt'], opt.batch_size_words, opt.gpus,
                              batch_size_sents=opt.batch_size_sents,
                              multiplier=opt.batch_size_multiplier)

    valid_data = onmt.Dataset(dataset['valid']['src'],
                              dataset['valid']['tgt'], opt.batch_size_words, opt.gpus,
                              batch_size_sents=opt.batch_size_sents)

    dicts = dataset['dicts']
    print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    print(' * number of training sentences. %d' %
          len(dataset['train']['src']['words']))
    print(' * maximum batch size (words per batch). %d' % opt.batch_size_words)

    # ------------------------------------------------------------------------------------------------#
    
    print('Building model...')

    # Loss function is built according to model 
    # Go to ModelConstructor for more details
    model, loss_function = build_model(opt, dicts)
        
    num_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % num_params)
    
    optim = None
    
    if len(opt.gpus) > 1 or opt.virtual_gpu > 1:
        # trainer = MultiGPUXETrainer(model, loss_function, train_data, valid_data, dataset, opt)
        raise NotImplementedError("Multi-GPU training is not supported atm")
    else:
        if opt.fp16:
            trainer = FP16XETrainer(model, loss_function, train_data, valid_data, dicts, opt)
        else:
            trainer = XETrainer(model, loss_function, train_data, valid_data, dicts, opt)

    trainer.run(save_file=opt.load_from)


if __name__ == "__main__":
    main()
