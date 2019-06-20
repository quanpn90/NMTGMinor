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
from onmt.modules.Loss import NMTLossFunc
from onmt.ModelConstructor import build_model, init_model_parameters
from ae.Autoencoder import Autoencoder
from ae.Trainer import AETrainer

parser = argparse.ArgumentParser(description='train.py')
onmt.Markdown.add_md_help_argument(parser)

from options import make_parser

# Please look at the options file to see the options regarding models and data
parser = make_parser(parser)


parser.add_argument('-representation', type=str, default="EncoderHiddenState",
                    help="Representation for Autoencoder")
parser.add_argument('-auto_encoder_hidden_size', type=int, default=100,
                    help="Hidden size of autoencoder")
parser.add_argument('-auto_encoder_drop_out', type=float, default=0,
                    help="Use drop_out in autoencoder")
parser.add_argument('-auto_encoder_type', type=str, default="Baseline",
                    help="Use drop_out in autoencoder")

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

    if opt.data_format == 'raw':
        start = time.time()
        print("Loading data from '%s'" % opt.data)

        if opt.data.endswith(".train.pt"):
            print("Loading data from '%s'" % opt.data)
            dataset = torch.load(opt.data)
        else:
            print("Loading data from %s" % opt.data + ".train.pt")
            dataset = torch.load(opt.data + ".train.pt")

        elapse = str(datetime.timedelta(seconds=int(time.time() - start)))
        print("Done after %s" % elapse)

        trainData = onmt.Dataset(dataset['train']['src'],
                                 dataset['train']['tgt'], opt.batch_size_words,
                                 data_type=dataset.get("type", "text"),
                                 batch_size_sents=opt.batch_size_sents,
                                 multiplier=opt.batch_size_multiplier)
        validData = onmt.Dataset(dataset['valid']['src'],
                                 dataset['valid']['tgt'], opt.batch_size_words,
                                 data_type=dataset.get("type", "text"),
                                 batch_size_sents=opt.batch_size_sents)

        dicts = dataset['dicts']
        if ("src" in dicts):
            print(' * vocabulary size. source = %d; target = %d' %
                  (dicts['src'].size(), dicts['tgt'].size()))
        else:
            print(' * vocabulary size. target = %d' %
                  (dicts['tgt'].size()))

        print(' * number of training sentences. %d' %
              len(dataset['train']['src']))
        print(' * maximum batch size (words per batch). %d' % opt.batch_size_words)
    elif opt.data_format == 'bin':
        from onmt.data_utils.IndexedDataset import IndexedInMemoryDataset

        dicts = torch.load(opt.data + ".dict.pt")

        # ~ train = {}
        train_path = opt.data + '.train'
        train_src = IndexedInMemoryDataset(train_path + '.src')
        train_tgt = IndexedInMemoryDataset(train_path + '.tgt')

        trainData = onmt.Dataset(train_src,
                                 train_tgt, opt.batch_size_words,
                                 batch_size_sents=opt.batch_size_sents,
                                 multiplier=opt.batch_size_multiplier)

        valid_path = opt.data + '.valid'
        valid_src = IndexedInMemoryDataset(valid_path + '.src')
        valid_tgt = IndexedInMemoryDataset(valid_path + '.tgt')

        validData = onmt.Dataset(valid_src,
                                 valid_tgt, opt.batch_size_words,
                                 batch_size_sents=opt.batch_size_sents)

    else:
        raise NotImplementedError

    print('Building model...')
    model = build_model(opt, dicts)
    autoencoder = Autoencoder(model,opt)

    """ Building the loss function """
    loss_function = nn.MSELoss(size_average=False)

    nParams = sum([p.nelement() for p in autoencoder.parameters()])
    print('* number of parameters: %d' % nParams)

    # load nmt model
    checkpoint = None
    if opt.load_from:
        checkpoint = torch.load(opt.load_from, map_location=lambda storage, loc: storage)
    else:
        raise NotImplementedError

    if checkpoint is not None:
        print('Loading model from checkpoint at %s' % opt.load_from)
        model.load_state_dict(checkpoint['model'])

        del checkpoint['model']
        del checkpoint['optim']
        del checkpoint

    if len(opt.gpus) > 1 or opt.virtual_gpu > 1:
        # ~ trainer = MultiGPUXETrainer(model, loss_function, trainData, validData, dataset, opt)
        raise NotImplementedError("Warning! Multi-GPU training is not fully tested and potential bugs can happen.")
    else:
            trainer = AETrainer(autoencoder,model, loss_function, trainData, validData, dicts, opt)

    trainer.run(save_file=False)


if __name__ == "__main__":
    main()
