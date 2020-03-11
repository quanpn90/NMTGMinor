from __future__ import division

import onmt
import onmt.markdown
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time, datetime
from onmt.train_utils.trainer import XETrainer
from onmt.modules.loss import NMTLossFunc, NMTAndCTCLossFunc
from onmt.model_factory import build_language_model, optimize_model
from onmt.data.lm_dataset import LanguageModelDataset
from collections import defaultdict


parser = argparse.ArgumentParser(description='train.py')
onmt.markdown.add_md_help_argument(parser)

from options import make_parser
# Please look at the options file to see the options regarding models and data
parser = make_parser(parser)

opt = parser.parse_args()

print(opt)

# An ugly hack to have weight norm on / off
onmt.constants.weight_norm = opt.weight_norm
onmt.constants.checkpointing = opt.checkpointing
onmt.constants.max_position_length = opt.max_position_length

# Use static dropout if checkpointing > 0
if opt.checkpointing > 0:
    onmt.constants.static = True

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, should run with -gpus 0")


torch.manual_seed(opt.seed)


def main():

    start = time.time()
    print("Loading data from '%s'" % opt.data)

    if opt.data_format == 'raw':
        dataset = torch.load(opt.data)
        elapse = str(datetime.timedelta(seconds=int(time.time() - start)))
        print("Done after %s" % elapse)

        dicts = dataset['dicts']

        # For backward compatibility
        train_dict = defaultdict(lambda: None, dataset['train'])
        valid_dict = defaultdict(lambda: None, dataset['valid'])

        if train_dict['src_lang'] is not None:
            assert 'langs' in dicts
            train_src_langs = train_dict['src_lang']
            train_tgt_langs = train_dict['tgt_lang']
        else:
            # allocate new languages
            dicts['langs'] = {'src': 0, 'tgt': 1}
            train_src_langs = list()
            train_tgt_langs = list()
            # Allocation one for the bilingual case
            train_src_langs.append(torch.Tensor([dicts['langs']['src']]))
            train_tgt_langs.append(torch.Tensor([dicts['langs']['tgt']]))

        train_data = LanguageModelDataset(
                                 dataset['train']['tgt'], train_tgt_langs,
                                 batch_size_sents=opt.batch_size_sents,
                                 seq_length=opt.lm_seq_length)

        if valid_dict['src_lang'] is not None:
            assert 'langs' in dicts
            valid_src_langs = valid_dict['src_lang']
            valid_tgt_langs = valid_dict['tgt_lang']
        else:
            # allocate new languages
            valid_src_langs = list()
            valid_tgt_langs = list()

            # Allocation one for the bilingual case
            valid_src_langs.append(torch.Tensor([dicts['langs']['src']]))
            valid_tgt_langs.append(torch.Tensor([dicts['langs']['tgt']]))

        valid_data = LanguageModelDataset(
                                 dataset['valid']['tgt'], valid_tgt_langs,
                                 batch_size_sents=opt.batch_size_sents,
                                 seq_length=opt.lm_seq_length)



        if opt.load_from:
            checkpoint = torch.load(opt.load_from, map_location=lambda storage, loc: storage)
            print("* Loading dictionaries from the checkpoint")
            dicts = checkpoint['dicts']
        else:
            dicts['tgt'].patch(opt.patch_vocab_multiplier)
            checkpoint = None

        if "src" in dicts:
            print(' * vocabulary size. source = %d; target = %d' %
            (dicts['src'].size(), dicts['tgt'].size()))
        else:
            print(' * vocabulary size. target = %d' %
            (dicts['tgt'].size()))

        print(' * number of training sentences. %d' %
          train_data.size())
        print(' * maximum batch size (words per batch). %d' % (opt.batch_size_sents * opt.lm_seq_length))

    else:
        raise NotImplementedError

    print('Building model...')
    model = build_language_model(opt, dicts)
    optimize_model(model)

    """ Building the loss function """
    loss_function = NMTLossFunc(opt.model_size, dicts['tgt'].size(), label_smoothing=opt.label_smoothing)

    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    
    if len(opt.gpus) > 1 or opt.virtual_gpu > 1:
        raise NotImplementedError("Multi-GPU training is not supported ATM.")
    else:
        # if opt.fp16:
        #     trainer = FP16XETrainer(model, loss_function, train_data, valid_data, dicts, opt)
        # else:
        trainer = XETrainer(model, loss_function, train_data, valid_data, dicts, opt)

    trainer.run(checkpoint=checkpoint)


if __name__ == "__main__":
    main()
