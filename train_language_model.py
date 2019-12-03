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
from onmt.modules.Loss import NMTLossFunc, NMTAndCTCLossFunc
from onmt.ModelConstructor import build_language_model
from onmt.Dataset import LanguageModelDataset

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


        train_data = LanguageModelDataset(
                                 dataset['train']['tgt'],
                                 batch_size_sents=opt.batch_size_sents,
                                 seq_length=opt.lm_seq_length)
        valid_data = LanguageModelDataset(
                                 dataset['valid']['tgt'],
                                 batch_size_sents=opt.batch_size_sents,
                                 seq_length=opt.lm_seq_length)

        dicts = dataset['dicts']
        if "src" in dicts:
            print(' * vocabulary size. source = %d; target = %d' %
            (dicts['src'].size(), dicts['tgt'].size()))
        else:
            print(' * vocabulary size. target = %d' %
            (dicts['tgt'].size()))

        print(' * number of training sentences. %d' %
          train_data.size())
        print(' * maximum batch size (words per batch). %d' % opt.batch_size_words)

    else:
        raise NotImplementedError

    print('Building model...')
    model = build_language_model(opt, dicts)

    print(model)
    
    """ Building the loss function """

    loss_function = NMTLossFunc(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)

    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    
    if len(opt.gpus) > 1 or opt.virtual_gpu > 1:
        raise NotImplementedError("Multi-GPU training is not supported ATM.")
    else:
        # if opt.fp16:
        #     trainer = FP16XETrainer(model, loss_function, train_data, valid_data, dicts, opt)
        # else:
        trainer = XETrainer(model, loss_function, train_data, valid_data, dicts, opt)

    
    trainer.run(save_file=opt.load_from)

if __name__ == "__main__":
    main()
