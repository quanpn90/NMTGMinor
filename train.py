from __future__ import division

import onmt
import onmt.markdown
import onmt.modules
import argparse
import torch
import time, datetime
from onmt.train_utils.trainer import XETrainer
from onmt.modules.loss import NMTLossFunc, NMTAndCTCLossFunc
from onmt.model_factory import build_model, optimize_model
from options import make_parser
from collections import defaultdict
import os
import numpy as np

parser = argparse.ArgumentParser(description='train.py')
onmt.markdown.add_md_help_argument(parser)

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
    if opt.data_format in ['bin', 'raw']:
        start = time.time()

        if opt.data.endswith(".train.pt"):
            print("Loading data from '%s'" % opt.data)
            dataset = torch.load(opt.data)
        else:
            print("Loading data from %s" % opt.data + ".train.pt")
            dataset = torch.load(opt.data + ".train.pt")

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

        if not opt.streaming:
            train_data = onmt.Dataset(train_dict['src'], train_dict['tgt'],
                                      None, None,
                                      train_src_langs, train_tgt_langs,
                                      batch_size_words=opt.batch_size_words,
                                      data_type=dataset.get("type", "text"), sorting=True,
                                      batch_size_sents=opt.batch_size_sents,
                                      multiplier=opt.batch_size_multiplier,
                                      augment=opt.augment_speech,
                                      upsampling=opt.upsampling)
        else:
            train_data = onmt.StreamDataset(train_dict['src'], train_dict['tgt'],
                                            train_src_langs, train_tgt_langs,
                                            batch_size_words=opt.batch_size_words,
                                            data_type=dataset.get("type", "text"), sorting=True,
                                            batch_size_sents=opt.batch_size_sents,
                                            multiplier=opt.batch_size_multiplier,
                                            augment=opt.augment_speech,
                                            upsampling=opt.upsampling)

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

        if not opt.streaming:
            valid_data = onmt.Dataset(valid_dict['src'], valid_dict['tgt'],
                                      None, None,
                                      valid_src_langs, valid_tgt_langs,
                                      batch_size_words=opt.batch_size_words,
                                      data_type=dataset.get("type", "text"), sorting=True,
                                      batch_size_sents=opt.batch_size_sents,
                                      upsampling=opt.upsampling)
        else:
            valid_data = onmt.StreamDataset(valid_dict['src'], valid_dict['tgt'],
                                            valid_src_langs, valid_tgt_langs,
                                            batch_size_words=opt.batch_size_words,
                                            data_type=dataset.get("type", "text"), sorting=True,
                                            batch_size_sents=opt.batch_size_sents,
                                            upsampling=opt.upsampling)

        print(' * number of training sentences. %d' % len(dataset['train']['src']))
        print(' * maximum batch size (words per batch). %d' % opt.batch_size_words)

    elif opt.data_format == 'mmem':
        print("Loading memory mapped data files ....")
        start = time.time()
        from onmt.data.mmap_indexed_dataset import MMapIndexedDataset

        dicts = torch.load(opt.data + ".dict.pt")

        # allocate languages if not
        if 'langs' not in dicts:
            dicts['langs'] = {'src': 0, 'tgt': 1}
        else:
            print(dicts['langs'])

        train_path = opt.data + '.train'
        train_src = MMapIndexedDataset(train_path + '.src')
        train_tgt = MMapIndexedDataset(train_path + '.tgt')

        # check the lang files if they exist (in the case of multi-lingual models)
        if os.path.exists(train_path + '.src_lang.bin'):
            assert 'langs' in dicts
            train_src_langs = MMapIndexedDataset(train_path + '.src_lang')
            train_tgt_langs = MMapIndexedDataset(train_path + '.tgt_lang')
        else:
            train_src_langs = list()
            train_tgt_langs = list()
            # Allocate a Tensor(1) for the bilingual case
            train_src_langs.append(torch.Tensor([dicts['langs']['src']]))
            train_tgt_langs.append(torch.Tensor([dicts['langs']['tgt']]))

        # check the length files if they exist
        if os.path.exists(train_path + '.src_sizes.npy'):
            train_src_sizes = np.load(train_path + '.src_sizes.npy')
            train_tgt_sizes = np.load(train_path + '.tgt_sizes.npy')
        else:
            train_src_sizes, train_tgt_sizes = None, None

        if opt.encoder_type == 'audio':
            data_type = 'audio'
        else:
            data_type = 'text'

        if not opt.streaming:
            train_data = onmt.Dataset(train_src,
                                      train_tgt,
                                      train_src_sizes, train_tgt_sizes,
                                      train_src_langs, train_tgt_langs,
                                      batch_size_words=opt.batch_size_words,
                                      data_type=data_type, sorting=True,
                                      batch_size_sents=opt.batch_size_sents,
                                      multiplier=opt.batch_size_multiplier,
                                      src_align_right=opt.src_align_right,
                                      upsampling=opt.upsampling,
                                      cleaning=True, verbose=True)
        else:
            train_data = onmt.StreamDataset(train_src,
                                            train_tgt,
                                            train_src_langs, train_tgt_langs,
                                            batch_size_words=opt.batch_size_words,
                                            data_type=data_type, sorting=False,
                                            batch_size_sents=opt.batch_size_sents,
                                            multiplier=opt.batch_size_multiplier,
                                            upsampling=opt.upsampling)

        valid_path = opt.data + '.valid'
        valid_src = MMapIndexedDataset(valid_path + '.src')
        valid_tgt = MMapIndexedDataset(valid_path + '.tgt')

        if os.path.exists(valid_path + '.src_lang.bin'):
            assert 'langs' in dicts
            valid_src_langs = MMapIndexedDataset(valid_path + '.src_lang')
            valid_tgt_langs = MMapIndexedDataset(valid_path + '.tgt_lang')
        else:
            valid_src_langs = list()
            valid_tgt_langs = list()

            # Allocation one for the bilingual case
            valid_src_langs.append(torch.Tensor([dicts['langs']['src']]))
            valid_tgt_langs.append(torch.Tensor([dicts['langs']['tgt']]))

        # check the length files if they exist
        if os.path.exists(valid_path + '.src_sizes.npy'):
            valid_src_sizes = np.load(valid_path + '.src_sizes.npy')
            valid_tgt_sizes = np.load(valid_path + '.tgt_sizes.npy')
        else:
            valid_src_sizes, valid_tgt_sizes = None, None

        if not opt.streaming:
            valid_data = onmt.Dataset(valid_src, valid_tgt,
                                      valid_src_sizes, valid_tgt_sizes,
                                      valid_src_langs, valid_tgt_langs,
                                      batch_size_words=opt.batch_size_words,
                                      data_type="text", sorting=True,
                                      batch_size_sents=opt.batch_size_sents,
                                      src_align_right=opt.src_align_right,
                                      cleaning=True, verbose=True, debug=True)
        else:
            # for validation data, we have to go through sentences (very slow but to ensure correctness)
            valid_data = onmt.StreamDataset(valid_src, valid_tgt,
                                            valid_src_langs, valid_tgt_langs,
                                            batch_size_words=opt.batch_size_words,
                                            data_type="text", sorting=True,
                                            batch_size_sents=opt.batch_size_sents)

        elapse = str(datetime.timedelta(seconds=int(time.time() - start)))
        print("Done after %s" % elapse)

    else:
        raise NotImplementedError

    # additional_data = []
    # if opt.additional_data != "none":
    #     add_data = opt.additional_data.split(";")
    #     add_format = opt.additional_data_format.split(";")
    #     assert (len(add_data) == len(add_format))
    #     for i in range(len(add_data)):
    #         if add_format[i] == 'raw':
    #             if add_data[i].endswith(".train.pt"):
    #                 print("Loading data from '%s'" % opt.data)
    #                 add_dataset = torch.load(add_data[i])
    #             else:
    #                 print("Loading data from %s" % opt.data + ".train.pt")
    #                 add_dataset = torch.load(add_data[i] + ".train.pt")
    #
    #             additional_data.append(onmt.Dataset(add_dataset['train']['src'],
    #                                                 dataset['train']['tgt'], batch_size_words=opt.batch_size_words,
    #                                                 data_type=dataset.get("type", "text"), sorting=True,
    #                                                 batch_size_sents=opt.batch_size_sents,
    #                                                 multiplier=opt.batch_size_multiplier,
    #                                                 reshape_speech=opt.reshape_speech,
    #                                                 augment=opt.augment_speech))
    #         elif add_format[i] == 'bin':
    #
    #             from onmt.data.indexed_dataset import IndexedInMemoryDataset
    #
    #             train_path = add_data[i] + '.train'
    #             train_src = IndexedInMemoryDataset(train_path + '.src')
    #             train_tgt = IndexedInMemoryDataset(train_path + '.tgt')
    #
    #             additional_data.append(onmt.Dataset(train_src,
    #                                                 train_tgt,
    #                                                 batch_size_words=opt.batch_size_words,
    #                                                 data_type=opt.encoder_type,
    #                                                 batch_size_sents=opt.batch_size_sents,
    #                                                 multiplier=opt.batch_size_multiplier))

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

    print(' * number of sentences in training data: %d' % train_data.size())
    print(' * number of sentences in validation data: %d' % valid_data.size())

    print('* Building model...')

    if not opt.fusion:
        model = build_model(opt, dicts)

        """ Building the loss function """
        if opt.ctc_loss != 0:
            loss_function = NMTAndCTCLossFunc(dicts['tgt'].size(),
                                              label_smoothing=opt.label_smoothing,
                                              ctc_weight=opt.ctc_loss)
        else:
            loss_function = NMTLossFunc(opt.model_size, dicts['tgt'].size(),
                                        label_smoothing=opt.label_smoothing,
                                        mirror=opt.mirror_loss)

        # This function replaces modules with the more optimized counterparts so that it can run faster
        # Currently exp with LayerNorm
        if not opt.memory_profiling:
            optimize_model(model)

    else:
        from onmt.model_factory import build_fusion
        from onmt.modules.loss import FusionLoss

        model = build_fusion(opt, dicts)

        loss_function = FusionLoss(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)

    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)

    if len(opt.gpus) > 1 or opt.virtual_gpu > 1:
        raise NotImplementedError("Multi-GPU training is not supported at the moment.")
    else:
        trainer = XETrainer(model, loss_function, train_data, valid_data, dicts, opt)

    trainer.run(checkpoint=checkpoint)


if __name__ == "__main__":
    main()
