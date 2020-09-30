import onmt
import onmt.markdown
import argparse
import torch
import subprocess
import time, datetime
from onmt.data.binarizer import Binarizer
from onmt.data.binarizer import SpeechBinarizer

from onmt.data.indexed_dataset import IndexedDatasetBuilder

import h5py as h5
import numpy as np
import warnings
import os
from os.path import dirname, abspath

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='preprocess.py')
onmt.markdown.add_md_help_argument(parser)

# **Preprocess Options**

parser.add_argument('-config', help="Read options from this file")

parser.add_argument('-src_type', default="text",
                    help="Type of the source input. Options are [text|img|audio].")
parser.add_argument('-sort_type', default="ascending",
                    help="Type of sorting. Options are [ascending|descending].")
parser.add_argument('-src_img_dir', default=".",
                    help="Location of source images")
parser.add_argument('-stride', type=int, default=1,
                    help="Stride on input features")
parser.add_argument('-concat', type=int, default=1,
                    help="Concate sequential audio features to decrease sequence length")
parser.add_argument('-previous_context', type=int, default=0,
                    help="Number of previous sentence for context")
parser.add_argument('-input_type', default="word",
                    help="Input type: word/char")
parser.add_argument('-data_type', default="int64",
                    help="Input type for storing text (int64|int32|int|int16) to reduce memory load")
parser.add_argument('-format', default="raw",
                    help="Save data format: binary or raw. Binary should be used to load faster")

parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")
parser.add_argument('-valid_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=True,
                    help="Path to the validation target data")

parser.add_argument('-train_src_lang', default="src",
                    help="Language(s) of the source sequences.")

parser.add_argument('-train_tgt_lang', default="tgt",
                    help="Language(s) of the target sequences.")

parser.add_argument('-valid_src_lang', default="src",
                    help="Language(s) of the source sequences.")

parser.add_argument('-valid_tgt_lang', default="tgt",
                    help="Language(s) of the target sequences.")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=9999999,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=9999999,
                    help="Size of the target vocabulary")

parser.add_argument('-load_dict',
                    help="Path to an existing source vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")

parser.add_argument('-src_seq_length', type=int, default=10000,
                    help="Maximum source sequence length")
parser.add_argument('-src_seq_length_trunc', type=int, default=0,
                    help="Truncate source sequence length.")
parser.add_argument('-tgt_seq_length', type=int, default=10000,
                    help="Maximum target sequence length to keep.")
parser.add_argument('-tgt_seq_length_trunc', type=int, default=0,
                    help="Truncate target sequence length.")

parser.add_argument('-shuffle', type=int, default=1,
                    help="Shuffle data")

parser.add_argument('-asr', action='store_true',
                    help="prepare data for asr task")
parser.add_argument('-asr_format', default="h5",
                    help="Format of asr data h5 or scp")
parser.add_argument('-lm', action='store_true',
                    help="prepare data for LM task")
parser.add_argument('-fp16', action='store_true',
                    help="store ASR data in fp16")

parser.add_argument('-seed', type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')
parser.add_argument('-load_bpe_voc', action='store_true', help='lowercase data')
parser.add_argument('-no_bos', action='store_true', help='not adding bos word (this is done manually in the data)')
parser.add_argument('-sort_by_target', action='store_true', help='lowercase data')
parser.add_argument('-join_vocab', action='store_true', help='Using one dictionary for both source and target')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")
parser.add_argument('-reshape_speech', type=int, default=1,
                    help="Reshaping the speech segments here. Mostly for compatibility..")
parser.add_argument('-num_threads', type=int, default=1,
                    help="Number of threads for multiprocessing")
parser.add_argument('-verbose', action='store_true',
                    help="Print out information during preprocessing")

parser.add_argument('-starting_train_idx', type=int, default=0,
                    help="")
parser.add_argument('-starting_valid_idx', type=int, default=0,
                    help="")

opt = parser.parse_args()

torch.manual_seed(opt.seed)


def make_vocab(filenames, size, tokenizer, num_workers=1):
    vocab = onmt.Dict([onmt.constants.PAD_WORD, onmt.constants.UNK_WORD,
                       onmt.constants.BOS_WORD, onmt.constants.EOS_WORD],
                      lower=opt.lower)

    for filename in filenames:
        print("Generating vocabulary from file %s ... " % filename)
        onmt.Dict.gen_dict_from_file(filename, vocab, tokenizer, num_workers=num_workers)

    original_size = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), original_size))

    return vocab


def init_vocab(name, data_files, vocab_file, vocab_size, tokenizer, num_workers=1):
    vocab = None
    if vocab_file is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocab_file + '\'...')
        if not opt.load_bpe_voc:
            vocab = onmt.Dict()
        else:
            vocab = onmt.Dict([onmt.constants.PAD_WORD, onmt.constants.UNK_WORD,
                               onmt.constants.BOS_WORD, onmt.constants.EOS_WORD],
                              lower=opt.lower)
        vocab.loadFile(vocab_file)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        print('Building ' + name + ' vocabulary...')
        gen_word_vocab = make_vocab(data_files, vocab_size, tokenizer, num_workers=num_workers)

        vocab = gen_word_vocab

    print()
    return vocab


def save_vocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def make_lm_data(tgt_file, tgt_dicts, max_tgt_length=1000, input_type='word', data_type='int32'):
    tgt = []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s ...' % (tgt_file))
    tgtf = open(tgt_file)

    eos = torch.LongTensor(1).fill_(onmt.constants.EOS)
    # print(eos.size())
    tensors = [eos]

    # find the number of words in the sentence
    while True:
        tline = tgtf.readline()

        # normal end of file
        if tline == "":
            break
        tline = tline.strip()
        # source and/or target are empty
        if tline == "":
            print('WARNING: ignoring an empty line (' + str(count + 1) + ')')
            continue

        if input_type == 'word':
            tgt_words = tline.split()
        elif input_type == 'char':
            tgt_words = split_line_by_char(tline)

        tensor = tgt_dicts.convertToIdx(tgt_words,
                                        onmt.constants.UNK_WORD,
                                        None,
                                        onmt.constants.EOS_WORD,
                                        type=data_type)
        # print(tensor.size())
        tensors.append(tensor)

        count = count + 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    tgtf.close()

    # concatenate all tensors into one
    tensor = torch.cat(tensors, dim=-1)

    return tensor


def make_translation_data(src_file, tgt_file, src_dicts, tgt_dicts, tokenizer, max_src_length=64, max_tgt_length=64,
                          add_bos=True, data_type='int64', num_workers=1, verbose=False):
    src, tgt = [], []
    src_sizes = []
    tgt_sizes = []

    print("[INFO] Binarizing file %s ..." % src_file)
    binarized_src = Binarizer.binarize_file(src_file, src_dicts, tokenizer,
                                            bos_word=None, eos_word=None,
                                            data_type=data_type,
                                            num_workers=num_workers, verbose=verbose)

    if add_bos:
        tgt_bos_word = onmt.constants.BOS_WORD
    else:
        tgt_bos_word = None

    print("[INFO] Binarizing file %s ..." % tgt_file)
    binarized_tgt = Binarizer.binarize_file(tgt_file, tgt_dicts, tokenizer,
                                            bos_word=tgt_bos_word, eos_word=onmt.constants.EOS_WORD,
                                            data_type=data_type,
                                            num_workers=num_workers, verbose=verbose)

    src = binarized_src['data']
    src_sizes = binarized_src['sizes']

    tgt = binarized_tgt['data']
    tgt_sizes = binarized_tgt['sizes']

    # currently we don't ignore anything :D
    ignored = 0

    print(('Prepared %d sentences ' +
           '(%d ignored due to length == 0 or src len > %d or tgt len > %d)') %
          (len(src), ignored, max_src_length, max_tgt_length))

    return src, tgt, src_sizes, tgt_sizes


def make_asr_data(src_file, tgt_file, tgt_dicts, tokenizer,
                  max_src_length=64, max_tgt_length=64, add_bos=True, data_type='int64', num_workers=1, verbose=False,
                  input_type='word', stride=1, concat=4, prev_context=0, fp16=False, reshape=True,
                  asr_format="h5", output_format="raw"):
    src, tgt = [], []
    src_sizes = []
    tgt_sizes = []
    count, ignored = 0, 0
    n_unk_words = 0

    print('[INFO] Processing %s  ...' % src_file)
    binarized_src = SpeechBinarizer.binarize_file(src_file, input_format=asr_format,
                                                  output_format=output_format, concat=concat,
                                                  stride=stride, fp16=fp16, prev_context=prev_context,
                                                  num_workers=num_workers)

    src = binarized_src['data']
    src_sizes = binarized_src['sizes']

    if add_bos:
        tgt_bos_word = onmt.constants.BOS_WORD
    else:
        tgt_bos_word = None

    print("[INFO] Binarizing file %s ..." % tgt_file)
    binarized_tgt = Binarizer.binarize_file(tgt_file, tgt_dicts, tokenizer,
                                            bos_word=tgt_bos_word, eos_word=onmt.constants.EOS_WORD,
                                            data_type=data_type,
                                            num_workers=num_workers, verbose=verbose)

    tgt = binarized_tgt['data']
    tgt_sizes = binarized_tgt['sizes']

    ignored = 0

    if len(src_sizes) != len(tgt_sizes):
        print("Warning: data size mismatched.")

    print(('Prepared %d sentences ' +
           '(%d ignored due to length == 0 or src len > %d or tgt len > %d)') %
          (len(src), ignored, max_src_length, max_tgt_length))

    return src, tgt, src_sizes, tgt_sizes


def save_dataset(path, data, format, dicts, src_type):
    # Each dataset is comprised of the following components:
    # src: tensors for the source vectors, or the scp_path (in ASR case)
    # tgt: tensors for the target vectors
    # src_lang: tensors for the source language ids (simplified)
    # tgt_lang: tensors for the target language ids (simplified)

    # convert all datasets to pytorch tensors and save to .pt
    if format in ['raw', 'bin']:
        print('Saving data to ' + os.path.join(path, 'data.pt') + '...')

        save_data = {'type': opt.src_type,
                     'data': data}
        torch.save(save_data, os.path.join(path, 'data.pt'))
        print("Done")

    # for ASR only
    elif format in ['scp', 'scpmem']:
        print('Saving target data to memory indexed data files. Source data is stored only as scp path.')
        from onmt.data.mmap_indexed_dataset import MMapIndexedDatasetBuilder

        assert opt.asr, "ASR data format is required for this memory indexed format"

        # TODO: changing this to before saving everything
        # torch.save(dicts, opt.save_data + '.dict.pt')

        # binarize the training set first
        for set_ in ['tgt', 'src_lang', 'tgt_lang']:
            if data[set_] is None:
                continue

            if opt.data_type == 'int64':
                dtype = np.int64
            else:
                dtype = np.int32

            indexed_data = MMapIndexedDatasetBuilder(os.path.join(path, "data.%s.bin" % set_), dtype=dtype)

            # add item from training data to the indexed data
            for tensor in data[set_]:
                indexed_data.add_item(tensor)

            indexed_data.finalize(os.path.join(path, "data.%s.idx" % set_))

            del indexed_data

        for set_ in ['src_sizes', 'tgt_sizes']:

            if data[set_] is not None:

                np_array = np.asarray(data[set_])
                np.save(os.path.join(path, "data.%s.npy") % set_, np_array)
            else:
                print("Training %s not found " % set_)

        # Finally save the audio path
        # save_data = {'data': data['src']}
        torch.save(data['src'], os.path.join(path, 'data.scp_path.pt'))

        print("Done")

    elif opt.format in ['mmap', 'mmem']:
        # print('Saving data to memory indexed data files')
        from onmt.data.mmap_indexed_dataset import MMapIndexedDatasetBuilder

        if opt.asr:
            print("ASR data format isn't compatible with memory indexed format")
            raise AssertionError

        # save dicts in this format
        # torch.save(dicts, opt.save_data + '.dict.pt')

        # binarize the training set firstd
        for set_ in ['src', 'tgt', 'src_lang', 'tgt_lang']:
            if data[set_] is None:
                continue

            if opt.data_type == 'int64':
                dtype = np.int64
            else:
                dtype = np.int32

            indexed_data = MMapIndexedDatasetBuilder(os.path.join(path, "data.%s.bin" % set_), dtype=dtype)

            # add item from training data to the indexed data
            for tensor in data[set_]:
                indexed_data.add_item(tensor)

            indexed_data.finalize(os.path.join(path, "data.%s.idx" % set_))

            del indexed_data

        for set_ in ['src_sizes', 'tgt_sizes']:

            if data[set_] is not None:

                np_array = np.asarray(data[set_])
                np.save(os.path.join(path, "data.%s.npy" % set_), np_array)
            else:
                print("Set %s not found " % set_)


def main():
    dicts = {}

    tokenizer = onmt.Tokenizer(opt.input_type, opt.lower)

    # construct set of languages from the training languages
    src_langs = opt.train_src_lang.split("|")
    tgt_langs = opt.train_tgt_lang.split("|")
    langs = (src_langs + tgt_langs)
    langs = list(set(langs))

    if opt.load_dict is not None:
        loaded_dict = torch.load(opt.load_dict)

        new_languages = list()
        for lang in langs:
            if lang not in loaded_dict['langs']:
                new_languages.append(lang)

        dicts['langs'] = loaded_dict['langs']
        print("Loaded dictionary for languages: ", dicts['langs'])
        if len(new_languages) > 0:
            for lang in new_languages:
                idx = len(dicts['langs'])
                dicts['langs'][lang] = idx
            print("Added new languages: ", new_languages)

        # dicts['tgt'] = loaded_dict['tgt']
        # dicts['src'] = loaded_dict['src'] if 'src' in loaded_dict else None
    else:
        dicts['langs'] = dict()

        for lang in langs:
            idx = len(dicts['langs'])
            dicts['langs'][lang] = idx

        print(dicts['langs'])

    start = time.time()

    src_train_files = opt.train_src.split("|")
    tgt_train_files = opt.train_tgt.split("|")
    # for ASR and LM we only need to build vocab for the 'target' language

    # TODO: adding new words to the existing dictionary in case loading from previously created dict
    if opt.asr or opt.lm:
        dicts['tgt'] = init_vocab('target', tgt_train_files, opt.tgt_vocab,
                                  opt.tgt_vocab_size, tokenizer, num_workers=opt.num_threads)
    elif opt.join_vocab:
        dicts['src'] = init_vocab('source', set(src_train_files + tgt_train_files), opt.src_vocab,
                                  opt.tgt_vocab_size, tokenizer, num_workers=opt.num_threads)
        dicts['tgt'] = dicts['src']

    else:
        dicts['src'] = init_vocab('source', src_train_files, opt.src_vocab,
                                  opt.src_vocab_size, tokenizer, num_workers=opt.num_threads)

        dicts['tgt'] = init_vocab('target', tgt_train_files, opt.tgt_vocab,
                                  opt.tgt_vocab_size, tokenizer, num_workers=opt.num_threads)

    elapse = str(datetime.timedelta(seconds=int(time.time() - start)))
    print("Vocabulary generated after %s" % elapse)

    # DATA GENERATION starts from here

    if opt.lm:
        raise NotImplementedError

    elif opt.asr:
        print('Preparing training acoustic model ...')

        src_input_files = opt.train_src.split("|")
        tgt_input_files = opt.train_tgt.split("|")

        src_langs = opt.train_src_lang.split("|")
        tgt_langs = opt.train_tgt_lang.split("|")

        assert len(src_input_files) == len(src_langs)
        assert len(src_input_files) == len(tgt_input_files)
        assert len(tgt_input_files) == len(tgt_langs)

        n_input_files = len(src_input_files)

        idx = opt.starting_train_idx
        for (src_file, tgt_file, src_lang, tgt_lang) in zip(src_input_files, tgt_input_files, src_langs, tgt_langs):
            # First, read and convert data to tensor format

            src_data, tgt_data, src_sizes, tgt_sizes = make_asr_data(src_file, tgt_file,
                                                                     dicts['tgt'], tokenizer,
                                                                     max_src_length=opt.src_seq_length,
                                                                     max_tgt_length=opt.tgt_seq_length,
                                                                     input_type=opt.input_type,
                                                                     stride=opt.stride, concat=opt.concat,
                                                                     prev_context=opt.previous_context,
                                                                     fp16=opt.fp16,
                                                                     asr_format=opt.asr_format,
                                                                     output_format=opt.format,
                                                                     num_workers=opt.num_threads)

            # save each dataset as bilingual (no multi parallel data)
            # we only need to have 1 language per file
            # which will be broadcasted
            n_samples = len(src_data)
            src_lang_data = [torch.Tensor([dicts['langs'][src_lang]])]
            tgt_lang_data = [torch.Tensor([dicts['langs'][tgt_lang]])]

            data = dict()

            data['src'] = src_data
            data['tgt'] = tgt_data

            data['src_sizes'] = src_sizes
            data['tgt_sizes'] = tgt_sizes
            data['src_lang'] = src_lang_data
            data['tgt_lang'] = tgt_lang_data

            print("Saving training set %i %s-%s to disk ..." % (idx, src_lang, tgt_lang))

            # take basedir from opt.save_data
            path = os.path.join(dirname(opt.save_data), "train.%i.%s-%s" % (idx, src_lang, tgt_lang))
            os.makedirs(path, exist_ok=True)

            # save data immediately
            save_dataset(path, data, opt.format, dicts, opt.src_type)
            idx = idx + 1
            # create

        print('Preparing validation ...')

        src_input_files = opt.valid_src.split("|")
        tgt_input_files = opt.valid_tgt.split("|")

        src_langs = opt.valid_src_lang.split("|")
        tgt_langs = opt.valid_tgt_lang.split("|")

        assert len(src_input_files) == len(src_langs)
        assert len(src_input_files) == len(tgt_input_files)
        assert len(tgt_input_files) == len(tgt_langs)

        n_input_files = len(src_input_files)

        idx = opt.starting_valid_idx

        for (src_file, tgt_file, src_lang, tgt_lang) in zip(src_input_files, tgt_input_files, src_langs, tgt_langs):
            src_data, tgt_data, src_sizes, tgt_sizes = make_asr_data(src_file, tgt_file,
                                                                     dicts['tgt'], tokenizer,
                                                                     max_src_length=max(1024, opt.src_seq_length),
                                                                     max_tgt_length=max(1024, opt.tgt_seq_length),
                                                                     input_type=opt.input_type,
                                                                     stride=opt.stride, concat=opt.concat,
                                                                     prev_context=opt.previous_context,
                                                                     fp16=opt.fp16,
                                                                     asr_format=opt.asr_format,
                                                                     output_format=opt.format)

            # save each dataset as bilingual (no multi parallel data)
            # we only need to have 1 language per file
            # which will be broadcasted
            n_samples = len(src_data)
            src_lang_data = [torch.Tensor([dicts['langs'][src_lang]])]
            tgt_lang_data = [torch.Tensor([dicts['langs'][tgt_lang]])]

            data = dict()

            data['src'] = src_data
            data['tgt'] = tgt_data

            data['src_sizes'] = src_sizes
            data['tgt_sizes'] = tgt_sizes
            data['src_lang'] = src_lang_data
            data['tgt_lang'] = tgt_lang_data

            print("Saving validation set %i %s-%s to disk ..." % (idx, src_lang, tgt_lang))

            # take basedir from opt.save_data
            path = os.path.join(dirname(opt.save_data), "valid.%i.%s-%s" % (idx, src_lang, tgt_lang))
            os.makedirs(path, exist_ok=True)

            # save data immediately
            save_dataset(path, data, opt.format, dicts, opt.src_type)
            idx = idx + 1

    else:

        # Translation dataset
        src_input_files = opt.train_src.split("|")
        tgt_input_files = opt.train_tgt.split("|")

        src_langs = opt.train_src_lang.split("|")
        tgt_langs = opt.train_tgt_lang.split("|")

        assert len(src_input_files) == len(src_langs)
        assert len(src_input_files) == len(tgt_input_files)
        assert len(tgt_input_files) == len(tgt_langs)

        n_input_files = len(src_input_files)

        start = time.time()
        print('Binarizing data to train translation models...')
        idx = opt.starting_train_idx

        for (src_file, tgt_file, src_lang, tgt_lang) in zip(src_input_files, tgt_input_files, src_langs, tgt_langs):
            src_data, tgt_data, src_sizes, tgt_sizes = make_translation_data(src_file, tgt_file,
                                                                             dicts['src'], dicts['tgt'], tokenizer,
                                                                             max_src_length=opt.src_seq_length,
                                                                             max_tgt_length=opt.tgt_seq_length,
                                                                             add_bos=(not opt.no_bos),
                                                                             data_type=opt.data_type,
                                                                             num_workers=opt.num_threads,
                                                                             verbose=opt.verbose)

            # save each dataset as bilingual (no multi parallel data)
            # we only need to have 1 language per file
            # which will be broadcasted
            n_samples = len(src_data)
            src_lang_data = [torch.Tensor([dicts['langs'][src_lang]])]
            tgt_lang_data = [torch.Tensor([dicts['langs'][tgt_lang]])]

            data = dict()
            data['src'] = src_data
            data['tgt'] = tgt_data

            data['src_sizes'] = src_sizes
            data['tgt_sizes'] = tgt_sizes
            data['src_lang'] = src_lang_data
            data['tgt_lang'] = tgt_lang_data

            print("Saving training set %i %s-%s to disk ..." % (idx, src_lang, tgt_lang))

            # take basedir from opt.save_data
            path = os.path.join(dirname(opt.save_data), "train.%i.%s-%s" % (idx, src_lang, tgt_lang))
            os.makedirs(path, exist_ok=True)

            # save data immediately
            save_dataset(path, data, opt.format, dicts, opt.src_type)
            idx = idx + 1

        print('Preparing validation ...')

        src_input_files = opt.valid_src.split("|")
        tgt_input_files = opt.valid_tgt.split("|")

        src_langs = opt.valid_src_lang.split("|")
        tgt_langs = opt.valid_tgt_lang.split("|")

        assert len(src_input_files) == len(src_langs)
        assert len(src_input_files) == len(tgt_input_files)
        assert len(tgt_input_files) == len(tgt_langs)

        n_input_files = len(src_input_files)

        idx = opt.starting_valid_idx
        for (src_file, tgt_file, src_lang, tgt_lang) in zip(src_input_files, tgt_input_files, src_langs, tgt_langs):
            src_data, tgt_data, src_sizes, tgt_sizes = make_translation_data(src_file, tgt_file,
                                                                             dicts['src'], dicts['tgt'], tokenizer,
                                                                             max_src_length=max(1024,
                                                                                                opt.src_seq_length),
                                                                             max_tgt_length=max(1024,
                                                                                                opt.tgt_seq_length),
                                                                             add_bos=(not opt.no_bos),
                                                                             data_type=opt.data_type,
                                                                             num_workers=opt.num_threads,
                                                                             verbose=opt.verbose)

            # save each dataset as bilingual (no multi parallel data)
            # we only need to have 1 language per file
            # which will be broadcasted
            n_samples = len(src_data)
            src_lang_data = [torch.Tensor([dicts['langs'][src_lang]])]
            tgt_lang_data = [torch.Tensor([dicts['langs'][tgt_lang]])]

            data = dict()
            data['src'] = src_data
            data['tgt'] = tgt_data

            data['src_sizes'] = src_sizes
            data['tgt_sizes'] = tgt_sizes
            data['src_lang'] = src_lang_data
            data['tgt_lang'] = tgt_lang_data

            print("Saving validation set %i %s-%s to disk ..." % (idx, src_lang, tgt_lang))

            # take basedir from opt.save_data
            path = os.path.join(dirname(opt.save_data), "valid.%i.%s-%s" % (idx, src_lang, tgt_lang))
            os.makedirs(path, exist_ok=True)

            # save data immediately
            save_dataset(path, data, opt.format, dicts, opt.src_type)
            idx = idx + 1

        elapse = str(datetime.timedelta(seconds=int(time.time() - start)))
        print("Binarization finished after %s" % elapse)

    print("Saving dictionary to %s" % (opt.save_data + '.dict.pt'))
    torch.save(dicts, opt.save_data + '.dict.pt')

    if opt.src_vocab is None and opt.asr == False and opt.lm == False:
        save_vocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        save_vocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')

    print("Finished.")


if __name__ == "__main__":
    main()