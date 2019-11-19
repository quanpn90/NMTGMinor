from __future__ import division

import onmt
import onmt.Markdown
import onmt.modules
import argparse
import torch
import time, datetime
from onmt.train_utils.trainer import XETrainer
from onmt.modules.Loss import NMTLossFunc, NMTAndCTCLossFunc
from onmt.ModelConstructor import build_model
from options import make_parser
from collections import defaultdict


parser = argparse.ArgumentParser(description='train.py')
onmt.Markdown.add_md_help_argument(parser)

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

    if opt.data_format == 'raw':
        start = time.time()

        if opt.data.endswith(".train.pt"):
            print("Loading data from '%s'" % opt.data)
            dataset = torch.load(opt.data)
        else:
            print("Loading data from %s" % opt.data + ".train.pt")
            dataset = torch.load(opt.data + ".train.pt")

        elapse = str(datetime.timedelta(seconds=int(time.time() - start)))
        print("Done after %s" % elapse )

        # For backward compatibility
        train_dict = defaultdict(lambda: None, dataset['train'])
        valid_dict = defaultdict(lambda: None, dataset['valid'])

        train_data = onmt.Dataset(train_dict['src'], train_dict['tgt'],
                                  train_dict['src_atbs'], train_dict['tgt_atbs'],
                                  batch_size_words=opt.batch_size_words,
                                  data_type=dataset.get("type", "text"),
                                  batch_size_sents=opt.batch_size_sents,
                                  multiplier=opt.batch_size_multiplier,
                                  reshape_speech=opt.reshape_speech,
                                  augment=opt.augment_speech)
        valid_data = onmt.Dataset(valid_dict['src'], valid_dict['tgt'],
                                  valid_dict['src_atbs'], valid_dict['tgt_atbs'],
                                  batch_size_words=opt.batch_size_words,
                                  data_type=dataset.get("type", "text"),
                                  batch_size_sents=opt.batch_size_sents,
                                  reshape_speech=opt.reshape_speech)

        dicts = dataset['dicts']

        print(' * number of training sentences. %d' % len(dataset['train']['src']))
        print(' * maximum batch size (words per batch). %d' % opt.batch_size_words)

    elif opt.data_format == 'bin':
        print("Loading memory binned data files ....")
        start = time.time()
        from onmt.data_utils.IndexedDataset import IndexedInMemoryDataset

        dicts = torch.load(opt.data + ".dict.pt")

        train_path = opt.data + '.train'
        train_src = IndexedInMemoryDataset(train_path + '.src')
        train_tgt = IndexedInMemoryDataset(train_path + '.tgt')

        train_data = onmt.Dataset(train_src,
                                  train_tgt,
                                  batch_size_words=opt.batch_size_words,
                                  data_type="text",
                                  batch_size_sents=opt.batch_size_sents,
                                  multiplier=opt.batch_size_multiplier)

        valid_path = opt.data + '.valid'
        valid_src = IndexedInMemoryDataset(valid_path + '.src')
        valid_tgt = IndexedInMemoryDataset(valid_path + '.tgt')

        valid_data = onmt.Dataset(valid_src,
                                  valid_tgt,
                                  batch_size_words=opt.batch_size_words,
                                  data_type="text",
                                  batch_size_sents=opt.batch_size_sents)
        elapse = str(datetime.timedelta(seconds=int(time.time() - start)))
        print("Done after %s" % elapse)
    elif opt.data_format == 'mmem':
        print("Loading memory mapped data files ....")
        start = time.time()
        from onmt.data_utils.MMapIndexedDataset import MMapIndexedDataset

        dicts = torch.load(opt.data + ".dict.pt")

        train_path = opt.data + '.train'
        train_src = MMapIndexedDataset(train_path + '.src')
        train_tgt = MMapIndexedDataset(train_path + '.tgt')

        train_data = onmt.Dataset(train_src,
                                  train_tgt,
                                  batch_size_words=opt.batch_size_words,
                                  data_type="text",
                                  batch_size_sents=opt.batch_size_sents,
                                  multiplier=opt.batch_size_multiplier)

        valid_path = opt.data + '.valid'
        valid_src = MMapIndexedDataset(valid_path + '.src')
        valid_tgt = MMapIndexedDataset(valid_path + '.tgt')

        valid_data = onmt.Dataset(valid_src,
                                  valid_tgt,
                                  batch_size_words=opt.batch_size_words,
                                  data_type="text",
                                  batch_size_sents=opt.batch_size_sents)
        elapse = str(datetime.timedelta(seconds=int(time.time() - start)))
        print("Done after %s" % elapse)

    else:
        raise NotImplementedError

    additional_data = []
    if opt.additional_data != "none":
        add_data = opt.additional_data.split(";")
        add_format = opt.additional_data_format.split(";")
        assert(len(add_data) == len(add_format))
        for i in range(len(add_data)):
            if add_format[i] == 'raw':
                if add_data[i].endswith(".train.pt"):
                    print("Loading data from '%s'" % opt.data)
                    add_dataset = torch.load(add_data[i])
                else:
                    print("Loading data from %s" % opt.data + ".train.pt")
                    add_dataset = torch.load(add_data[i] + ".train.pt")

                additional_data.append(onmt.Dataset(add_dataset['train']['src'],
                                          dataset['train']['tgt'], batch_size_words=opt.batch_size_words,
                                          data_type=dataset.get("type", "text"),
                                          batch_size_sents=opt.batch_size_sents,
                                          multiplier=opt.batch_size_multiplier,
                                          reshape_speech=opt.reshape_speech,
                                          augment=opt.augment_speech))
            elif add_format[i] == 'bin':

                from onmt.data_utils.IndexedDataset import IndexedInMemoryDataset

                train_path = add_data[i] + '.train'
                train_src = IndexedInMemoryDataset(train_path + '.src')
                train_tgt = IndexedInMemoryDataset(train_path + '.tgt')

                additional_data.append(onmt.Dataset(train_src,
                                       train_tgt,
                                       batch_size_words=opt.batch_size_words,
                                       data_type=opt.encoder_type,
                                       batch_size_sents=opt.batch_size_sents,
                                       multiplier = opt.batch_size_multiplier))

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

    print('Building model...')

    if not opt.fusion:
        model = build_model(opt, dicts)

        """ Building the loss function """
        if opt.ctc_loss != 0:
            loss_function = NMTAndCTCLossFunc(dicts['tgt'].size(),
                                              label_smoothing=opt.label_smoothing,
                                              ctc_weight=opt.ctc_loss)
        else:
            loss_function = NMTLossFunc(dicts['tgt'].size(),
                                        label_smoothing=opt.label_smoothing)
    else:
        from onmt.ModelConstructor import build_fusion
        from onmt.modules.Loss import FusionLoss

        model = build_fusion(opt, dicts)

        loss_function = FusionLoss(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)

    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)

    if len(opt.gpus) > 1 or opt.virtual_gpu > 1:
        raise NotImplementedError("Warning! Multi-GPU training is not fully tested and potential bugs can happen.")
    else:
        trainer = XETrainer(model, loss_function, train_data, valid_data, dicts, opt)
        if len(additional_data) > 0:
            trainer.add_additional_data(additional_data, opt.data_ratio);

    trainer.run(checkpoint=checkpoint)


if __name__ == "__main__":
    main()
