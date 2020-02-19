import onmt
import onmt.markdown
import argparse
import torch
import subprocess
import time, datetime
from onmt.data.binarizer import Binarizer

from onmt.data.indexed_dataset import IndexedDatasetBuilder

import h5py as h5
import numpy as np

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
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")

parser.add_argument('-src_seq_length', type=int, default=64,
                    help="Maximum source sequence length")
parser.add_argument('-src_seq_length_trunc', type=int, default=0,
                    help="Truncate source sequence length.")
parser.add_argument('-tgt_seq_length', type=int, default=66,
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
        if tline == "": break
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

    return src, tgt


def make_asr_data(src_file, tgt_file, tgt_dicts, max_src_length=64, max_tgt_length=64,
                  input_type='word', stride=1, concat=1, prev_context=0, fp16=False, reshape=True, asr_format="h5"):
    src, tgt = [], []
    # sizes = []
    src_sizes = []
    tgt_sizes = []
    count, ignored = 0, 0
    n_unk_words = 0

    print('Processing %s & %s ...' % (src_file, tgt_file))

    if (asr_format == "h5"):
        fileIdx = -1;
        if (src_file[-2:] == "h5"):
            srcf = h5.File(src_file, 'r')
        else:
            fileIdx = 0
            srcf = h5.File(src_file + "." + str(fileIdx) + ".h5", 'r')
    elif (asr_format == "scp"):
        import kaldiio
        from kaldiio import ReadHelper
        audio_data = iter(ReadHelper('scp:' + src_file))

    tgtf = open(tgt_file)

    index = 0

    s_prev_context = []
    t_prev_context = []

    while True:
        tline = tgtf.readline()
        # normal end of file
        if tline == "":
            break

        if (asr_format == "h5"):
            if (str(index) in srcf):
                featureVectors = np.array(srcf[str(index)])
            elif (fileIdx != -1):
                srcf.close()
                fileIdx += 1
                srcf = h5.File(src_file + "." + str(fileIdx) + ".h5", 'r')
                featureVectors = np.array(srcf[str(index)])
            else:
                print("No feature vector for index:", index, file=sys.stderr)
                exit(-1)
        elif (asr_format == "scp"):
            _, featureVectors = next(audio_data)

        if (stride == 1):
            sline = torch.from_numpy(featureVectors)
        else:
            sline = torch.from_numpy(featureVectors[0::opt.stride])

        if reshape:
            if concat != 1:
                add = (concat - sline.size()[0] % concat) % concat
                z = torch.FloatTensor(add, sline.size()[1]).zero_()
                sline = torch.cat((sline, z), 0)
                sline = sline.reshape((int(sline.size()[0] / concat), sline.size()[1] * concat))
        index += 1;

        tline = tline.strip()

        if prev_context > 0:
            print("Multiple ASR context isn't supported at the moment   ")
            raise NotImplementedError

            # s_prev_context.append(sline)
            # t_prev_context.append(tline)
            # for i in range(1,prev_context+1):
            #     if i < len(s_prev_context):
            #         sline = torch.cat((torch.cat((s_prev_context[-i-1],torch.zeros(1,sline.size()[1]))),sline))
            #         tline = t_prev_context[-i-1]+" # "+tline
            # if len(s_prev_context) > prev_context:
            #     s_prev_context = s_prev_context[-1*prev_context:]
            #     t_prev_context = t_prev_context[-1*prev_context:]

        # source and/or target are empty
        if tline == "":
            print('WARNING: ignoring an empty line (' + str(count + 1) + ')')
            continue

        if input_type == 'word':
            tgt_words = tline.split()
        elif input_type == 'char':
            tgt_words = split_line_by_char(tline)

        if len(tgt_words) <= max_tgt_length - 2 and sline.size(0) <= max_src_length:

            # Check truncation condition.
            if opt.tgt_seq_length_trunc != 0:
                tgt_words = tgt_words[:opt.tgt_seq_length_trunc]

            if fp16:
                sline = sline.half()
            src += [sline]

            tgt_tensor = tgt_dicts.convertToIdx(tgt_words,
                                                onmt.constants.UNK_WORD,
                                                onmt.constants.BOS_WORD,
                                                onmt.constants.EOS_WORD)
            tgt += [tgt_tensor]
            src_sizes += [len(sline)]
            tgt_sizes += [len(tgt_words)]

            unks = tgt_tensor.eq(onmt.constants.UNK).sum().item()
            n_unk_words += unks

            if unks > 0:
                if "<unk>" not in tline:
                    print("DEBUGGING: This line contains UNK: %s" % tline)

        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)
    if (asr_format == "h5"):
        srcf.close()
    tgtf.close()

    print('Total number of unk words: %d' % n_unk_words)

    print(('Prepared %d sentences ' +
           '(%d ignored due to length == 0 or src len > %d or tgt len > %d)') %
          (len(src), ignored, max_src_length, max_tgt_length))

    return src, tgt


def main():
    dicts = {}

    tokenizer = onmt.Tokenizer(opt.input_type, opt.lower)

    # construct set of languages from the training languages
    src_langs = opt.train_src_lang.split("|")
    tgt_langs = opt.train_tgt_lang.split("|")
    langs = (src_langs + tgt_langs)
    langs = list(set(langs))

    dicts['langs'] = dict()

    for lang in langs:
        idx = len(dicts['langs'])
        dicts['langs'][lang] = idx

    print(dicts['langs'])

    start = time.time()

    src_train_files = opt.train_src.split("|")
    tgt_train_files = opt.train_tgt.split("|")
    # for ASR and LM we only need to build vocab for the 'target' language
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

    if opt.lm:
        print('Preparing training language model ...')
        train = dict()
        train['tgt'] = make_lm_data(opt.train_tgt,
                                    dicts['tgt'])
        train['src'] = None

        valid = dict()
        valid['tgt'] = make_lm_data(opt.valid_tgt,
                                    dicts['tgt'])
        valid['src'] = None

    elif opt.asr:
        print('Preparing training acoustic model ...')
        train = dict()
        train['src'], train['tgt'] = make_asr_data(opt.train_src, opt.train_tgt,
                                                   dicts['tgt'],
                                                   max_src_length=opt.src_seq_length,
                                                   max_tgt_length=opt.tgt_seq_length,
                                                   input_type=opt.input_type,
                                                   stride=opt.stride, concat=opt.concat,
                                                   prev_context=opt.previous_context,
                                                   fp16=opt.fp16, reshape=(opt.reshape_speech == 1),
                                                   asr_format=opt.asr_format)

        print('Preparing validation ...')
        valid = dict()
        valid['src'], valid['tgt'] = make_asr_data(opt.valid_src, opt.valid_tgt,
                                                   dicts['tgt'],
                                                   max_src_length=max(1024, opt.src_seq_length),
                                                   max_tgt_length=max(1024, opt.tgt_seq_length),
                                                   input_type=opt.input_type,
                                                   stride=opt.stride, concat=opt.concat,
                                                   prev_context=opt.previous_context,
                                                   fp16=opt.fp16, reshape=(opt.reshape_speech == 1),
                                                   asr_format=opt.asr_format)

    else:

        src_input_files = opt.train_src.split("|")
        tgt_input_files = opt.train_tgt.split("|")

        src_langs = opt.train_src_lang.split("|")
        tgt_langs = opt.train_tgt_lang.split("|")

        assert len(src_input_files) == len(src_langs)
        assert len(src_input_files) == len(tgt_input_files)
        assert len(tgt_input_files) == len(tgt_langs)

        n_input_files = len(src_input_files)

        train = dict()
        train['src'], train['tgt'] = list(), list()
        train['src_lang'], train['tgt_lang'] = list(), list()

        start = time.time()
        print('Binarizing data to train translation models...')

        for (src_file, tgt_file, src_lang, tgt_lang) in zip(src_input_files, tgt_input_files, src_langs, tgt_langs):

            src_data, tgt_data = make_translation_data(src_file, tgt_file,
                                                       dicts['src'], dicts['tgt'], tokenizer,
                                                       max_src_length=opt.src_seq_length,
                                                       max_tgt_length=opt.tgt_seq_length,
                                                       add_bos=(not opt.no_bos),
                                                       data_type=opt.data_type,
                                                       num_workers=opt.num_threads,
                                                       verbose=opt.verbose)

            n_samples = len(src_data)
            if n_input_files == 1:
                # For single-file cases we only need to have 1 language per file
                # which will be broadcasted
                src_lang_data = [torch.Tensor([dicts['langs'][src_lang]])]
                tgt_lang_data = [torch.Tensor([dicts['langs'][tgt_lang]])]
            else:
                # each sample will have a different language id
                src_lang_data = [torch.Tensor([dicts['langs'][src_lang]]) for _ in range(n_samples)]
                tgt_lang_data = [torch.Tensor([dicts['langs'][tgt_lang]]) for _ in range(n_samples)]

            train['src'] += src_data
            train['tgt'] += tgt_data
            train['src_lang'] += src_lang_data
            train['tgt_lang'] += tgt_lang_data

        print('Preparing validation ...')

        src_input_files = opt.valid_src.split("|")
        tgt_input_files = opt.valid_tgt.split("|")

        src_langs = opt.valid_src_lang.split("|")
        tgt_langs = opt.valid_tgt_lang.split("|")

        assert len(src_input_files) == len(src_langs)
        assert len(src_input_files) == len(tgt_input_files)
        assert len(tgt_input_files) == len(tgt_langs)

        n_input_files = len(src_input_files)

        valid = dict()
        valid['src'], valid['tgt'] = list(), list()
        valid['src_lang'], valid['tgt_lang'] = list(), list()

        for (src_file, tgt_file, src_lang, tgt_lang) in zip(src_input_files, tgt_input_files, src_langs, tgt_langs):

            src_data, tgt_data = make_translation_data(src_file, tgt_file,
                                                               dicts['src'], dicts['tgt'], tokenizer,
                                                               max_src_length=max(1024, opt.src_seq_length),
                                                               max_tgt_length=max(1024, opt.tgt_seq_length),
                                                               add_bos=(not opt.no_bos),
                                                               data_type=opt.data_type,
                                                               num_workers=opt.num_threads,
                                                               verbose=opt.verbose)

            n_samples = len(src_data)
            if n_input_files == 1:
                # For single-file cases we only need to have 1 language per file
                # which will be broadcasted
                src_lang_data = [torch.Tensor([dicts['langs'][src_lang]])]
                tgt_lang_data = [torch.Tensor([dicts['langs'][tgt_lang]])]
            else:
                # each sample will have a different language id
                src_lang_data = [torch.Tensor([dicts['langs'][src_lang]]) for _ in range(n_samples)]
                tgt_lang_data = [torch.Tensor([dicts['langs'][tgt_lang]]) for _ in range(n_samples)]

            valid['src'] += src_data
            valid['tgt'] += tgt_data
            valid['src_lang'] += src_lang_data
            valid['tgt_lang'] += tgt_lang_data

        elapse = str(datetime.timedelta(seconds=int(time.time() - start)))
        print("Binarization finished after %s" % elapse)

    if opt.src_vocab is None and opt.asr == False and opt.lm == False:
        save_vocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        save_vocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')

    # SAVE DATA
    if opt.format in ['raw', 'bin']:

        print('Saving data to \'' + opt.save_data + '.train.pt\'...')
        save_data = {'dicts': dicts,
                     'type': opt.src_type,
                     'train': train,
                     'valid': valid}
        torch.save(save_data, opt.save_data + '.train.pt')
        print("Done")

    elif opt.format in ['mmap', 'mmem']:
        print('Saving data to memory indexed data files')
        from onmt.data.mmap_indexed_dataset import MMapIndexedDatasetBuilder

        if opt.asr:
            print("ASR data format isn't compatible with memory indexed format")
            raise AssertionError

        # save dicts in this format
        torch.save(dicts, opt.save_data + '.dict.pt')

        # binarize the training set first
        for set_ in ['src', 'tgt', 'src_lang', 'tgt_lang']:
            if train[set_] is None:
                continue

            if opt.data_type == 'int64':
                dtype = np.int64
            else:
                dtype = np.int32

            if set_ == 'src' and opt.asr:
                dtype = np.double

            train_data = MMapIndexedDatasetBuilder(opt.save_data + ".train.%s.bin" % set_, dtype=dtype)

            # add item from training data to the indexed data
            for tensor in train[set_]:
                train_data.add_item(tensor)

            train_data.finalize(opt.save_data + ".train.%s.idx" % set_)

            del train_data

            if valid[set_] is None:
                continue

            if set_ == 'src' and opt.asr:
                dtype = np.double

            valid_data = MMapIndexedDatasetBuilder(opt.save_data + ".valid.%s.bin" % set_, dtype=dtype)

            # add item from training data to the indexed data
            for tensor in valid[set_]:
                valid_data.add_item(tensor)

            valid_data.finalize(opt.save_data + ".valid.%s.idx" % set_)

            del valid_data

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins
