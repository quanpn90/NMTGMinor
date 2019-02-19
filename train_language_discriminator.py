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
from onmt.train_utils.lang_discriminator_trainer import LanguageDiscriminatorTrainer
from onmt.ModelConstructor import build_model, init_model_parameters
from onmt.data_utils.IndexedDataset import IndexedInMemoryDataset

parser = argparse.ArgumentParser(description='train_language_discriminator.py')
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
    print("Debugging a language discriminator")
    print("Loading data from '%s'" % opt.data)

    dicts = torch.load(opt.data + ".dict.pt")

    train_path = opt.data + '.train'
    train_src, train_tgt = dict(), dict()
    train_src['words'] = IndexedInMemoryDataset(train_path + '.words.src')
    train_tgt['words'] = IndexedInMemoryDataset(train_path + '.words.tgt')
    train_src['attbs'] = IndexedInMemoryDataset(train_path + '.langs.src')
    train_tgt['attbs'] = IndexedInMemoryDataset(train_path + '.langs.tgt')

    train_data = onmt.Dataset(train_src,
                              train_tgt, opt.batch_size_words, opt.gpus,
                              batch_size_sents=opt.batch_size_sents,
                              pad_count=opt.pad_count,
                              multiplier=opt.batch_size_multiplier,
                              sort_by_target=opt.sort_by_target)

    valid_path = opt.data + '.valid'
    valid_src, valid_tgt = dict(), dict()
    valid_src['words'] = IndexedInMemoryDataset(valid_path + '.words.src')
    valid_tgt['words'] = IndexedInMemoryDataset(valid_path + '.words.tgt')
    valid_src['attbs'] = IndexedInMemoryDataset(valid_path + '.langs.src')
    valid_tgt['attbs'] = IndexedInMemoryDataset(valid_path + '.langs.tgt')

    valid_data = onmt.Dataset(valid_src,
                              valid_tgt, opt.batch_size_words, opt.gpus,
                              batch_size_sents=opt.batch_size_sents)

    elapse = str(datetime.timedelta(seconds=int(time.time() - start)))
    print("Done after %s" % elapse)
    print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    print(' * number of training sentences. %d' %
          len(train_src['words']))
    print(' * number of languages. %d' % dicts['atb'].size())
    print(' * maximum batch size (words per batch). %d' % opt.batch_size_words)

    print('Building model...')

    # Loss function is built according to model
    # Go to ModelConstructor for more details
    from onmt.modules.LanguageDiscriminator.Models import LanguageDiscriminator

    n_languages = dicts['atb'].size()

    embedding_src = nn.Embedding(dicts['src'].size(),
                                 opt.model_size,
                                 padding_idx=onmt.Constants.PAD)
    model = LanguageDiscriminator(opt, embedding_src, n_languages)

    loss_function = nn.NLLLoss(reduce=False)

    num_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % num_params)

    trainer = LanguageDiscriminatorTrainer(model, loss_function, train_data, valid_data, dicts, opt)

    trainer.run(save_file=opt.load_from)


if __name__ == "__main__":
    main()
