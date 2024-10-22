#!/usr/bin/env python
from __future__ import division
import pickle
import types

import onmt
import onmt.markdown
import onmt.modules
import argparse
import torch
import time, datetime
from onmt.data.mmap_indexed_dataset import MMapIndexedDataset
from onmt.data.scp_dataset import SCPIndexDataset
from onmt.data.wav_dataset import WavDataset
from options import make_parser
from collections import defaultdict
from onmt.constants import add_tokenidx
import os
import numpy as np
import warnings
import dill
from torch.multiprocessing import Pool, Process, set_start_method

def pickle_trick(obj, max_depth=10):
    output = {}

    if max_depth <= 0:
        return output

    try:
        pickle.dumps(obj)
    except (pickle.PicklingError, TypeError) as e:
        failing_children = []

        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                result = pickle_trick(v, max_depth=max_depth - 1)
                if result:
                    failing_children.append(result)

        output = {
            "fail": obj,
            "err": e,
            "depth": max_depth,
            "failing_children": failing_children
        }

    return output


Dataset = onmt.Dataset

def numpy_to_torch(tensor_list):
    out_list = list()

    for tensor in tensor_list:
        if isinstance(tensor, np.ndarray):
            out_list.append(torch.from_numpy(tensor))
        else:
            out_list.append(tensor)

    return out_list


def run_process(gpu, train_data, valid_data, dicts, opt, checkpoint, constants):
    """
    Launch training for normal sequence2sequence models

    Args:
        gpu:
        train_data:
        valid_data:
        dicts:
        opt:
        checkpoint:
        constants:

    Returns:

    """
    # Online Continual Learning - each example is visited once
    if opt.ocl_training:
        if opt.bayes_by_backprop:
            from onmt.train_utils.bayes_by_backprop_trainer import BayesianOCLTrainer
            trainer = BayesianOCLTrainer(gpu, dicts, opt, constants)
        elif opt.vat_training > 0:
            from onmt.train_utils.vat_trainer import VAT_OCLTrainer
            trainer = VAT_OCLTrainer(gpu, dicts, opt, constants)
        elif opt.meta_learning:
            from onmt.train_utils.meta_trainer import MetaOCLTrainer
            trainer = MetaOCLTrainer(gpu, dicts, opt, constants)
        else:
            from onmt.train_utils.ocl_trainer import OCLTrainer
            trainer = OCLTrainer(gpu, dicts, opt, constants)
    # Continual Learning, but Offline. When we want to add more data into the current system
    # There are more algorithms we can try:
    # - EWC
    # - Reservoir
    # - combined with VAT training?
    elif opt.offline_cl_training:

        from onmt.train_utils.offline_cl_trainer import OfflineCLTrainer
        trainer = OfflineCLTrainer(gpu, dicts, opt, constants)
    else:
        from onmt.train_utils.mp_trainer import Trainer
        trainer = Trainer(gpu, dicts, opt, constants)

    # if opt.clip_learning:
    #     from onmt.train_utils.clip_trainer import ClipTrainer
    #     trainer = ClipTrainer(gpu, dicts, opt, constants)
    #     trainer.run(checkpoint=checkpoint, train_data=train_data, valid_data=valid_data)
    # else:

    trainer.run(checkpoint=checkpoint, train_data=train_data, valid_data=valid_data)


def run_gem_process(gpu, train_data, valid_data, dicts, opt, checkpoint, constants):
    """
    Launch training for Gradient Episodic Memory

    Args:
        gpu:
        train_data:
        valid_data:
        dicts:
        opt:
        checkpoint:
        constants:

    Returns:

    """
    from onmt.train_utils.gem_trainer import GEMTrainer

    trainer = GEMTrainer(gpu, train_data, valid_data, dicts, opt, constants)
    trainer.run(checkpoint=checkpoint)


def main(gpu, opt):

    def is_main():
        return gpu == 0

    def print_main(*args, **kwargs):
        if is_main():
            print(*args, **kwargs)

    def lprint(*args, **kwargs):
        if gpu == 0:
            print(*args, **kwargs, flush=True)

    if opt.char_ctc:
        lprint("Loading char data ...")

        char_data = torch.load("char_data.pt")
    else:
        char_data = None

    if not opt.multi_dataset:
        if opt.data_format in ['bin', 'raw']:
            start = time.time()

            if opt.data.endswith(".train.pt"):
                lprint("Loading data from '%s'" % opt.data)
                dataset = torch.load(opt.data)
            else:
                lprint("Loading data from %s" % opt.data + ".train.pt")
                dataset = torch.load(opt.data + ".train.pt")

            elapse = str(datetime.timedelta(seconds=int(time.time() - start)))
            lprint("Done after %s" % elapse)

            dicts = dataset['dicts']
            onmt.constants = add_tokenidx(opt, onmt.constants, dicts)

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

            if train_dict['src_atb'] is not None:
                assert 'atbs' in dicts
                train_src_atbs = train_dict['src_atb']
                train_tgt_atbs = train_dict['tgt_atb']
            else:
                # allocate new languages
                dicts['atbs'] = {'nothingness': 0}
                train_src_atbs = list()
                train_tgt_atbs = list()
                train_src_atbs.append(torch.Tensor([dicts['atbs']['nothingness']]))
                train_tgt_atbs.append(torch.Tensor([dicts['atbs']['nothingness']]))

            if not opt.streaming:
                constants = dill.dumps(onmt.constants)

                train_data = onmt.Dataset(numpy_to_torch(train_dict['src']), numpy_to_torch(train_dict['tgt']),
                                          train_dict['src_sizes'], train_dict['tgt_sizes'],
                                          train_src_langs, train_tgt_langs,
                                          train_src_atbs, train_tgt_atbs,
                                          batch_size_words=opt.batch_size_words,
                                          batch_size_frames=opt.batch_size_frames,
                                          data_type=dataset.get("type", "text"), sorting=True, cleaning=True,
                                          batch_size_sents=opt.batch_size_sents,
                                          multiplier=opt.batch_size_multiplier,
                                          augment=opt.augment_speech, sa_f=opt.sa_f, sa_t=opt.sa_t,
                                          max_src_len=opt.max_src_length,
                                          max_tgt_len=opt.max_tgt_length,
                                          input_size=opt.input_size,
                                          upsampling=opt.upsampling,
                                          num_split=1,
                                          constants=constants,
                                          use_memory=hasattr(opt, "use_memory") and opt.use_memory, validation=False,
                                          concat=opt.concat_dataset,
                                          char_data=char_data,
                                          use_char_level=opt.char_ctc,
                                          create_reverse=(opt.mirror_loss > 0),
                                          device=gpu)
            else:
                train_data = onmt.StreamDataset(train_dict['src'], train_dict['tgt'],
                                                train_src_langs, train_tgt_langs,
                                                batch_size_words=opt.batch_size_words,
                                                data_type=dataset.get("type", "text"), sorting=True,
                                                batch_size_sents=opt.batch_size_sents,
                                                multiplier=opt.batch_size_multiplier,
                                                augment=opt.augment_speech,
                                                upsampling=opt.upsampling)

            dicts['tgt_pad'] = train_data.tgt_pad

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

            if valid_dict['src_atb'] is not None:
                assert 'atbs' in dicts
                valid_src_atbs = valid_dict['src_atb']
                valid_tgt_atbs = valid_dict['tgt_atb']
            else:
                # allocate new languages
                valid_src_atbs = list()
                valid_tgt_atbs = list()

                valid_src_atbs.append(torch.Tensor([dicts['atbs']['nothingness']]))
                valid_tgt_atbs.append(torch.Tensor([dicts['atbs']['nothingness']]))

            if not opt.streaming:
                constants = dill.dumps(onmt.constants)

                valid_data = onmt.Dataset(numpy_to_torch(valid_dict['src']), numpy_to_torch(valid_dict['tgt']),
                                          valid_dict['src_sizes'], valid_dict['tgt_sizes'],
                                          valid_src_langs, valid_tgt_langs,
                                          valid_src_atbs, valid_tgt_atbs,
                                          batch_size_words=opt.batch_size_words,
                                          batch_size_frames=opt.batch_size_frames,
                                          data_type=dataset.get("type", "text"), sorting=True,
                                          batch_size_sents=opt.batch_size_sents,
                                          max_src_len=opt.max_src_length,
                                          max_tgt_len=opt.max_tgt_length,
                                          multiplier=opt.batch_size_multiplier,
                                          upsampling=opt.upsampling,
                                          input_size=opt.input_size,
                                          constants=constants,
                                          use_memory=hasattr(opt, "use_memory") and opt.use_memory,
                                          char_data=char_data,
                                          use_char_level=opt.char_ctc,
                                          create_reverse=(opt.mirror_loss > 0),
                                          device=gpu)
            else:
                valid_data = onmt.StreamDataset(numpy_to_torch(valid_dict['src']), numpy_to_torch(valid_dict['tgt']),
                                                valid_src_langs, valid_tgt_langs,
                                                batch_size_words=opt.batch_size_words,
                                                data_type=dataset.get("type", "text"), sorting=True,
                                                batch_size_sents=opt.batch_size_sents,
                                                upsampling=opt.upsampling)

            lprint(' * number of training sentences. %d' % len(dataset['train']['src']))
            lprint(' * maximum batch size (words per batch). %d' % opt.batch_size_words)

        # Loading asr data structures
        elif opt.data_format in ['scp', 'scpmem', 'mmem', 'wav']:

            from onmt.data.mmap_indexed_dataset import MMapIndexedDataset

            lprint("Loading memory mapped data files ....")
            start = time.time()

            from onmt.data.scp_dataset import SCPIndexDataset

            dicts = torch.load(opt.data + ".dict.pt")
            onmt.constants = add_tokenidx(opt, onmt.constants, dicts)

            if opt.data_format in ['scp', 'scpmem']:
                audio_data = torch.load(opt.data + ".scp_path.pt")
            elif opt.data_format in ['wav']:
                audio_data = torch.load(opt.data + ".wav_path.pt")
                # # TODO: maybe having another option like -past_context
                # if os.path.exists(opt.data + '.prev_src_path.pt'):
                #     prev_audio_data = torch.load(opt.data + '.prev_src_path.pt')
                # else:
                #     prev_audio_data = None

            # allocate languages if not
            if 'langs' not in dicts:
                dicts['langs'] = {'src': 0, 'tgt': 1}
            else:
                lprint(dicts['langs'])

            train_path = opt.data + '.train'
            if opt.data_format in ['scp', 'scpmem']:
                train_src = SCPIndexDataset(audio_data['train'], concat=opt.concat)
                if 'train_past' in audio_data:
                    past_train_src = SCPIndexDataset(audio_data['train_past'],
                                                     concat=opt.concat, shared_object=train_src)
                else:
                    past_train_src = None
            elif opt.data_format in ['wav']:
                train_src = WavDataset(audio_data['train'], cache_size=opt.data_cache_size,
                                       wav_path_replace=opt.wav_path_replace, num_mel_bin=opt.num_mel_bin,
                                       specaugment=opt.augment_speech)
                past_train_src = None
            else:
                train_src = MMapIndexedDataset(train_path + '.src')
                past_train_src = None

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

            if os.path.exists(train_path + '.src_atb.bin'):
                assert 'atbs' in dicts
                train_src_atbs = MMapIndexedDataset(train_path + '.src_atb')
                train_tgt_atbs = MMapIndexedDataset(train_path + '.tgt_atb')
            else:
                dicts['atbs'] = {'nothingness': 0}
                train_src_atbs = list()
                train_tgt_atbs = list()
                train_src_atbs.append(torch.Tensor([dicts['atbs']['nothingness']]))
                train_tgt_atbs.append(torch.Tensor([dicts['atbs']['nothingness']]))

            # check the length files if they exist
            if os.path.exists(train_path + '.src_sizes.npy'):
                train_src_sizes = np.load(train_path + '.src_sizes.npy')
                train_tgt_sizes = np.load(train_path + '.tgt_sizes.npy')
            else:
                train_src_sizes, train_tgt_sizes = None, None

            # check the length files if they exist
            if os.path.exists(train_path + '.past_src_sizes.npy'):
                past_train_src_sizes = np.load(train_path + '.past_src_sizes.npy')
            else:
                past_train_src_sizes = None

            if opt.data_format in ['scp', 'scpmem']:
                data_type = 'audio'
            elif opt.data_format in ['wav']:
                data_type = 'wav'
            else:
                data_type = 'text'

            if not opt.streaming:
                constants = dill.dumps(onmt.constants)

                train_data = onmt.Dataset(train_src,
                                          train_tgt,
                                          train_src_sizes, train_tgt_sizes,
                                          train_src_langs, train_tgt_langs,
                                          train_src_atbs, train_tgt_atbs,
                                          batch_size_words=opt.batch_size_words,
                                          batch_size_frames=opt.batch_size_frames,
                                          data_type=data_type, sorting=True,
                                          batch_size_sents=opt.batch_size_sents,
                                          multiplier=opt.batch_size_multiplier,
                                          augment=opt.augment_speech, sa_f=opt.sa_f, sa_t=opt.sa_t,
                                          cleaning=True, verbose=True,
                                          input_size=opt.input_size,
                                          past_src_data=past_train_src,
                                          past_src_data_sizes=past_train_src_sizes,
                                          max_src_len=opt.max_src_length,
                                          max_tgt_len=opt.max_tgt_length,
                                          constants=constants,
                                          use_memory=hasattr(opt, "use_memory") and opt.use_memory, validation=False,
                                          concat=opt.concat_dataset,
                                          char_data=char_data,
                                          use_char_level=opt.char_ctc,
                                          create_reverse=(opt.mirror_loss > 0),
                                          device=gpu)
            else:
                train_data = onmt.StreamDataset(train_src,
                                                train_tgt,
                                                train_src_langs, train_tgt_langs,
                                                batch_size_words=opt.batch_size_words,
                                                data_type=data_type, sorting=False,
                                                batch_size_sents=opt.batch_size_sents,
                                                multiplier=opt.batch_size_multiplier,
                                                upsampling=opt.upsampling)

            dicts['tgt_pad'] = train_data.tgt_pad

            valid_path = opt.data + '.valid'
            if opt.data_format in ['scp', 'scpmem']:
                valid_src = SCPIndexDataset(audio_data['valid'], concat=opt.concat)
                if 'valid_past' in audio_data:
                    past_valid_src = SCPIndexDataset(audio_data['valid_past'],
                                                     concat=opt.concat, shared_object=valid_src)
                else:
                    past_valid_src = None
            elif opt.data_format in ['wav']:
                valid_src = WavDataset(audio_data['valid'], cache_size=opt.data_cache_size,
                                       wav_path_replace=opt.wav_path_replace, num_mel_bin=opt.num_mel_bin,
                                       specaugment=False)
                past_valid_src = None
            else:
                valid_src = MMapIndexedDataset(valid_path + '.src')
                past_valid_src = None

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

            if os.path.exists(valid_path + '.src_atb.bin'):
                assert 'atbs' in dicts
                valid_src_atbs = MMapIndexedDataset(valid_path + '.src_atb')
                valid_tgt_atbs = MMapIndexedDataset(valid_path + '.tgt_atb')
            else:
                valid_src_atbs = list()
                valid_tgt_atbs = list()
                valid_src_atbs.append(torch.Tensor([dicts['atbs']['nothingness']]))
                valid_tgt_atbs.append(torch.Tensor([dicts['atbs']['nothingness']]))

            # check the length files if they exist
            if os.path.exists(valid_path + '.src_sizes.npy'):
                valid_src_sizes = np.load(valid_path + '.src_sizes.npy')
                valid_tgt_sizes = np.load(valid_path + '.tgt_sizes.npy')
            else:
                valid_src_sizes, valid_tgt_sizes = None, None

            # check the length files if they exist
            if os.path.exists(valid_path + '.past_src_sizes.npy'):
                past_valid_src_sizes = np.load(valid_path + '.past_src_sizes.npy')
            else:
                past_valid_src_sizes = None

            if not opt.streaming:
                constants = dill.dumps(onmt.constants)

                valid_data = onmt.Dataset(valid_src, valid_tgt,
                                          valid_src_sizes, valid_tgt_sizes,
                                          valid_src_langs, valid_tgt_langs,
                                          valid_src_atbs, valid_tgt_atbs,
                                          batch_size_words=opt.batch_size_words,
                                          batch_size_frames=opt.batch_size_frames,
                                          multiplier=opt.batch_size_multiplier,
                                          data_type=data_type, sorting=True,
                                          input_size=opt.input_size,
                                          batch_size_sents=opt.batch_size_sents,
                                          cleaning=True, verbose=True, debug=True,
                                          past_src_data=past_valid_src,
                                          past_src_data_sizes=past_valid_src_sizes,
                                          max_src_len=opt.max_src_length,
                                          max_tgt_len=opt.max_tgt_length,
                                          min_src_len=1, min_tgt_len=3,
                                          constants=constants,
                                          use_memory=hasattr(opt, "use_memory") and opt.use_memory,
                                          char_data=char_data,
                                          use_char_level=opt.char_ctc,
                                          create_reverse=(opt.mirror_loss > 0),
                                          device=gpu)
            else:
                # for validation data, we have to go through sentences (very slow but to ensure correctness)
                valid_data = onmt.StreamDataset(valid_src, valid_tgt,
                                                valid_src_langs, valid_tgt_langs,
                                                batch_size_words=opt.batch_size_words,
                                                data_type=data_type, sorting=True,
                                                batch_size_sents=opt.batch_size_sents)

            elapse = str(datetime.timedelta(seconds=int(time.time() - start)))
            lprint("Done after %s" % elapse)

        else:
            raise NotImplementedError

        lprint(' * number of sentences in training data: %d' % train_data.size())
        lprint(' * number of sentences in validation data: %d' % valid_data.size())

    # Multi-data set handling
    else:
        if opt.dataset_factors is not None:
            dataset_factors = {int(df.split(":")[0]): int(df.split(":")[1]) for df in opt.dataset_factors.split(",")}
            lprint("Dataset factors:", dataset_factors)
        else:
            dataset_factors = {}

        lprint("[INFO] Reading multiple dataset ...")

        dicts = torch.load(opt.data + ".dict.pt")
        lprint("Languages: ", dicts['langs'])
        if 'atbs' not in dicts or len(dicts['atbs']) == 0:  # backward compatible
            dicts['atbs'] = {'nothingness': 0}
        lprint("Atributes: ", dicts['atbs'])

        onmt.constants = add_tokenidx(opt, onmt.constants, dicts)

        root_dir = os.path.dirname(opt.data)

        lprint("Loading training data ...")

        train_dirs, valid_dirs = dict(), dict()

        # scan the data directory to find the training data
        for dir_ in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, dir_)):
                if str(dir_).startswith("train"):
                    idx = int(dir_.split(".")[1])
                    train_dirs[idx] = dir_
                if dir_.startswith("valid"):
                    idx = int(dir_.split(".")[1])
                    valid_dirs[idx] = dir_

        train_sets, valid_sets = list(), list()

        c = 0
        for (idx_, dir_) in sorted(train_dirs.items()):
            c += 1
            data_dir = os.path.join(root_dir, dir_)
            lprint("[INFO] Loading training data %i from %s" % (idx_, dir_))

            if opt.data_format in ['bin', 'raw']:
                raise NotImplementedError

            elif opt.data_format in ['scp', 'scpmem', 'mmem', 'wav']:
                from onmt.data.mmap_indexed_dataset import MMapIndexedDataset
                from onmt.data.scp_dataset import SCPIndexDataset

                if opt.data_format in ['scp', 'scpmem']:
                    audio_data = torch.load(os.path.join(data_dir, "data.scp_path.pt"))
                    src_data = SCPIndexDataset(audio_data, concat=opt.concat)
                elif opt.data_format in ['wav']:
                    audio_data = torch.load(os.path.join(data_dir, "data.scp_path.pt"))
                    src_data = WavDataset(audio_data, cache_size=opt.data_cache_size,
                                          wav_path_replace=opt.wav_path_replace, num_mel_bin=opt.num_mel_bin,
                                          specaugment=opt.augment_speech)
                else:
                    src_data = MMapIndexedDataset(os.path.join(data_dir, "data.src"))

                tgt_data = MMapIndexedDataset(os.path.join(data_dir, "data.tgt"))

                src_lang_data = MMapIndexedDataset(os.path.join(data_dir, 'data.src_lang'))
                tgt_lang_data = MMapIndexedDataset(os.path.join(data_dir, 'data.tgt_lang'))

                if os.path.exists(os.path.join(data_dir, 'data.src_atb.bin')):
                    src_atbs_data = MMapIndexedDataset(os.path.join(data_dir, 'data.src_atb'))
                    tgt_atbs_data = MMapIndexedDataset(os.path.join(data_dir, 'data.tgt_atb'))
                else:
                    src_atbs_data = list()
                    tgt_atbs_data = list()
                    src_atbs_data.append(torch.Tensor([dicts['atbs']['nothingness']]))
                    tgt_atbs_data.append(torch.Tensor([dicts['atbs']['nothingness']]))

                if os.path.exists(os.path.join(data_dir, 'data.src_sizes.npy')):
                    src_sizes = np.load(os.path.join(data_dir, 'data.src_sizes.npy'))
                    tgt_sizes = np.load(os.path.join(data_dir, 'data.tgt_sizes.npy'))
                else:
                    src_sizes, sizes = None, None

                if opt.encoder_type in ['audio', 'wav2vec2_scp']:
                    data_type = 'audio'
                elif opt.encoder_type == 'wav2vec2':
                    data_type = 'wav'
                else:
                    data_type = 'text'

                if not opt.streaming:


                    constants = dill.dumps(onmt.constants)

                    train_data = onmt.Dataset(src_data,
                                              tgt_data,
                                              src_sizes, tgt_sizes,
                                              src_lang_data, tgt_lang_data,
                                              src_atbs_data, tgt_atbs_data,
                                              batch_size_words=opt.batch_size_words,
                                              batch_size_frames=opt.batch_size_frames,
                                              data_type=data_type, sorting=True,
                                              batch_size_sents=opt.batch_size_sents,
                                              multiplier=opt.batch_size_multiplier,
                                              upsampling=opt.upsampling,
                                              augment=opt.augment_speech, sa_f=opt.sa_f, sa_t=opt.sa_t,
                                              cleaning=True, verbose=True,
                                              max_src_len=opt.max_src_length,
                                              max_tgt_len=opt.max_tgt_length,
                                              input_size=opt.input_size,
                                              constants=constants,
                                              dataset_factor=dataset_factors.get(idx_),
                                              use_memory=hasattr(opt, "use_memory") and opt.use_memory, validation=False,
                                              concat=opt.concat_dataset,
                                              char_data=char_data,
                                              use_char_level=opt.char_ctc,
                                              create_reverse=(opt.mirror_loss > 0),
                                              device=gpu)

                    if c == 1:
                        dicts['tgt_pad'] = train_data.get_tgt_pad()
                        del src_sizes, tgt_sizes, src_data, tgt_data, src_lang_data, tgt_lang_data

                    train_sets.append(train_data)

                else:
                    lprint("Multi-dataset not implemented for Streaming tasks.")
                    raise NotImplementedError

        for (idx_, dir_) in sorted(valid_dirs.items()):

            data_dir = os.path.join(root_dir, dir_)

            lprint("[INFO] Loading validation data %i from %s" % (idx_, dir_))

            if opt.data_format in ['bin', 'raw']:
                raise NotImplementedError

            elif opt.data_format in ['scp', 'scpmem', 'mmem', 'wav']:

                if opt.data_format in ['scp', 'scpmem']:
                    audio_data = torch.load(os.path.join(data_dir, "data.scp_path.pt"))
                    src_data = SCPIndexDataset(audio_data, concat=opt.concat)
                elif opt.data_format in ['wav']:
                    audio_data = torch.load(os.path.join(data_dir, "data.scp_path.pt"))
                    src_data = WavDataset(audio_data, cache_size=opt.data_cache_size,
                                          wav_path_replace=opt.wav_path_replace, num_mel_bin=opt.num_mel_bin,
                                          specaugment=False)
                else:
                    src_data = MMapIndexedDataset(os.path.join(data_dir, "data.src"))

                tgt_data = MMapIndexedDataset(os.path.join(data_dir, "data.tgt"))

                src_lang_data = MMapIndexedDataset(os.path.join(data_dir, 'data.src_lang'))
                tgt_lang_data = MMapIndexedDataset(os.path.join(data_dir, 'data.tgt_lang'))

                # load data attributes
                if os.path.exists(os.path.join(data_dir, 'data.src_atb.bin')):
                    src_atbs_data = MMapIndexedDataset(os.path.join(data_dir, 'data.src_atb'))
                    tgt_atbs_data = MMapIndexedDataset(os.path.join(data_dir, 'data.tgt_atb'))
                else:
                    src_atbs_data = list()
                    tgt_atbs_data = list()
                    src_atbs_data.append(torch.Tensor([dicts['atbs']['nothingness']]))
                    tgt_atbs_data.append(torch.Tensor([dicts['atbs']['nothingness']]))

                # load data size
                if os.path.exists(os.path.join(data_dir, 'data.src_sizes.npy')):
                    src_sizes = np.load(os.path.join(data_dir, 'data.src_sizes.npy'))
                    tgt_sizes = np.load(os.path.join(data_dir, 'data.tgt_sizes.npy'))
                else:
                    src_sizes, sizes = None, None

                if opt.encoder_type in ['audio', 'wav2vec2_scp']:
                    data_type = 'audio'
                elif opt.encoder_type == 'wav2vec2':
                    data_type = 'wav'
                else:
                    data_type = 'text'

                if not opt.streaming:
                    constants = dill.dumps(onmt.constants)

                    valid_data = onmt.Dataset(src_data, tgt_data,
                                              src_sizes, tgt_sizes,
                                              src_lang_data, tgt_lang_data,
                                              src_atbs_data, tgt_atbs_data,
                                              batch_size_words=opt.batch_size_words,
                                              batch_size_frames=opt.batch_size_frames,
                                              multiplier=opt.batch_size_multiplier,
                                              data_type=data_type, sorting=True,
                                              batch_size_sents=opt.batch_size_sents,
                                              min_src_len=1, min_tgt_len=3,
                                              input_size=opt.input_size,
                                              cleaning=True, verbose=True,
                                              constants=constants,
                                              use_memory=hasattr(opt, "use_memory") and opt.use_memory,
                                              char_data=char_data,
                                              use_char_level=opt.char_ctc,
                                              create_reverse=(opt.mirror_loss > 0),
                                              device=gpu)

                    valid_sets.append(valid_data)

                else:
                    raise NotImplementedError

        train_data = train_sets
        valid_data = valid_sets

    if opt.load_from and not opt.reset_optim:
        lprint("Loading checkpoint: ", opt.load_from)
        checkpoint = torch.load(opt.load_from, map_location=lambda storage, loc: storage)
        lprint("* Loading dictionaries from the checkpoint")
        del checkpoint['model']
        del checkpoint['optim']
        if opt.override_dict_from_checkpoint:
            dicts = checkpoint['dicts']
    else:
        dicts['tgt'].patch(opt.patch_vocab_multiplier)
        checkpoint = None

    if opt.char_ctc:
        dicts['char_data'] = char_data

    if "src" in dicts:
        lprint(' * vocabulary size. source = %d; target = %d' %
              (dicts['src'].size(), dicts['tgt'].size()))
    else:
        lprint(' * vocabulary size. target = %d' %
              (dicts['tgt'].size()))

    os.environ['MASTER_ADDR'] = opt.master_addr  # default 'localhost'
    os.environ['MASTER_PORT'] = opt.master_port  # default '8888'

    # spawn N processes for N gpus
    # each process has a different trainer
    constants = dill.dumps(onmt.constants)

    if opt.gem_training:
        run_gem_process(gpu, train_data, valid_data, dicts, opt, checkpoint, constants)
    else:
        run_process(gpu, train_data, valid_data, dicts, opt, checkpoint, constants)



if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="The given NumPy array is not writeable ")
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser(description='train_distributed.py')
    onmt.markdown.add_md_help_argument(parser)

    # Please look at the options file to see the options regarding models and data
    parser = make_parser(parser)

    opt = parser.parse_args()

    # An ugly hack to have weight norm on / off
    onmt.constants.weight_norm = opt.weight_norm
    onmt.constants.checkpointing = opt.checkpointing
    onmt.constants.max_position_length = opt.max_position_length

    # Use static dropout if checkpointing > 0
    if opt.checkpointing > 0:
        onmt.constants.static = True

    if torch.cuda.is_available() and not opt.gpus:
        print("WARNING: You have a CUDA device, should run with -gpus 0")

    if len(opt.gpus) == 1:
        main(0, opt)
    else:
        torch.multiprocessing.spawn(main, args=(opt, ),
                                    nprocs=len(opt.gpus),
                                    join=True)