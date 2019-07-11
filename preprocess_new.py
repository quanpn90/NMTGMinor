import onmt
import onmt.Markdown
import argparse
import torch

from onmt.data_utils.IndexedDataset import IndexedDatasetBuilder

import h5py as h5
import numpy as np

parser = argparse.ArgumentParser(description='preprocess.py')
onmt.Markdown.add_md_help_argument(parser)

# **Preprocess Options**

parser.add_argument('-config', help="Read options from this file")

parser.add_argument('-src_type', default="text",
                    help="Type of the source input. Options are [text|audio].")
parser.add_argument('-sort_type', default="ascending",
                    help="Type of sorting. Options are [ascending|descending].")
parser.add_argument('-stride', type=int, default=1,
                    help="Stride on input features")
parser.add_argument('-concat', type=int, default=1,
                    help="Concate sequential audio features to decrease sequence length")
parser.add_argument('-previous_context', type=int, default=0,
                    help="Number of previous sentence for context")
parser.add_argument('-input_type', default="word",
                    help="Input type: word/char")
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

# For sentence-level textual features (target side only)
parser.add_argument('-train_src_atb', default="",
                    help="Path to the training source attributes")
parser.add_argument('-train_tgt_atb', default="",
                    help="Path to the training target attributes")
parser.add_argument('-valid_src_atb', default="",
                    help="Path to the validation source attributes")
parser.add_argument('-valid_tgt_atb', default="",
                    help="Path to the validation target attributes  ")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=50000,
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
parser.add_argument('-sort_by_target', action='store_true', help='lowercase data')
parser.add_argument('-join_vocab', action='store_true', help='Using one dictionary for both source and target')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")
parser.add_argument('-reshape_speech', type=int, default=1,
                    help="Reshaping the speech segments here. Mostly for compatibility..")

opt = parser.parse_args()

torch.manual_seed(opt.seed)


def split_line_by_char(line, word_list=["<unk>"]):
    # first we split by words
    chars = list()

    words = line.strip().split()

    for i, word in enumerate(words):
        if word in word_list:
            chars.append(word)
        else:
            for c in word:
                chars.append(c)

        if i < (len(words) - 1):
            chars.append(' ')

    return chars


def make_join_vocab(filenames, size, input_type="word"):
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD],
                      lower=opt.lower)

    for filename in filenames:
        print("Reading file %s ... " % filename)
        with open(filename) as f:
            for sent in f.readlines():

                if input_type == "word":
                    for word in sent.split():
                        vocab.add(word)
                elif input_type == "char":
                    chars = split_line_by_char(sent)
                    for char in chars:
                        vocab.add(char)
                else:
                    raise NotImplementedError("Input type not implemented")

    original_size = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), original_size))

    return vocab


def make_vocab(filename, size, input_type='word'):
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD],
                      lower=opt.lower)

    unk_count = 0

    with open(filename) as f:
        for sent in f.readlines():
            if input_type == "word":
                for word in sent.split():
                    idx = vocab.add(word)
            elif input_type == "char":
                chars = split_line_by_char(sent)
                for char in chars:
                    idx = vocab.add(char)
            else:
                raise NotImplementedError("Input type not implemented")

            if idx == 'onmt.Constants.UNK':
                unk_count += 1

    original_size = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), original_size))

    return vocab


def init_vocab(name, dataFile, vocabFile, vocabSize, join=False, input_type='word'):
    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = onmt.Dict()
        vocab.loadFile(vocabFile)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:

        # If a dictionary is still missing, generate it.
        if join:

            print('Building ' + 'shared' + ' vocabulary...')
            gen_word_vocab = make_join_vocab(dataFile, vocabSize, input_type=input_type)
        else:
            print('Building ' + name + ' vocabulary...')
            gen_word_vocab = make_vocab(dataFile, vocabSize, input_type=input_type)

        vocab = gen_word_vocab

    print()
    return vocab


def save_vocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def make_lm_data(tgt_file, tgt_dicts, max_tgt_length=1000, input_type='word'):
    tgt = []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s ...' % (tgt_file))
    tgtf = open(tgt_file)

    eos = torch.LongTensor(1).fill_(onmt.Constants.EOS)
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
                                        onmt.Constants.UNK_WORD,
                                        None,
                                        onmt.Constants.EOS_WORD)
        tensors.append(tensor)

        count = count + 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    tgtf.close()

    # concatenate all tensors into one
    tensor = torch.cat(tensors, dim=-1)

    return tensor


def read_text_file(text_file, dicts, input_type='word', max_length=100000,
                   bos_word=None, eos_word=None):
    tensors = []
    tensor_sizes = []

    bad_indices = []

    reader = open(text_file)
    line_idx = -1

    while True:
        line = reader.readline()
        line_idx += 1

        if line == "":
            break  # end of file

        line = line.strip()

        if line == "":
            # empty line
            tensors += [dicts.convertToIdx([], onmt.Constants.UNK_WORD)]
            tensor_sizes += [0]
            bad_indices.append(line_idx)
            continue

        if input_type == 'word':
            words = line.split()
        elif input_type == 'char':
            words = split_line_by_char(line)

        if len(words) <= max_length:
            tensors += [dicts.convertToIdx(words, onmt.Constants.UNK_WORD,
                                           bos_word=bos_word, eos_word=eos_word)]

            tensor_sizes += [len(words)]
        else:
            # too long line
            tensors += [dicts.convertToIdx([], onmt.Constants.UNK_WORD)]
            tensor_sizes += [0]
            bad_indices.append(line_idx)
            continue

    reader.close()

    return tensors, tensor_sizes, bad_indices


def read_atb_file(atb_file, atb_dicts):
    all_atbs = dict()

    atb_num = len(atb_dicts)

    for i in range(atb_num):
        all_atbs[i] = []

    reader = open(atb_file)

    line_idx = -1

    while True:

        line = reader.readline()
        line_idx += 1

        if line == "":
            break  # end of file

        line = line.strip()

        atbs = line.split()

        for i, atb in enumerate(atbs):
            tensor = atb_dicts[i].convertToIdx2([atb], None, eos_word=None)
            all_atbs[i].append(tensor)

    return all_atbs


def make_translation_data(src_file, tgt_file, src_dicts, tgt_dicts,
                          src_atb_file=None, tgt_atb_file=None, atb_dicts=None,
                          max_src_length=64, max_tgt_length=64, sort_by_target=False,
                          input_type='word'):
    src, tgt = [], []
    src_sizes = []
    tgt_sizes = []
    count, ignored = 0, 0
    output_dict = dict()

    print('Processing %s & %s ...' % (src_file, tgt_file))

    src, src_sizes, src_bad_indices = read_text_file(src_file, src_dicts, input_type, max_src_length)
    tgt, tgt_sizes, tgt_bad_indices = read_text_file(tgt_file, tgt_dicts, input_type, max_tgt_length,
                                                     bos_word=onmt.Constants.BOS_WORD,
                                                     eos_word=onmt.Constants.EOS_WORD)

    # remove the items from the bad indices
    bad_indices = list(set(src_bad_indices + tgt_bad_indices))
    good_indices = list(set(list(range(len(src)))) - set(bad_indices))

    src = [src[i] for i in good_indices]
    tgt = [tgt[i] for i in good_indices]
    src_sizes = [src_sizes[i] for i in good_indices]
    tgt_sizes = [tgt_sizes[i] for i in good_indices]

    assert len(src) == len(tgt), "Two languages must have the same number of sentences"

    if src_atb_file:
        src_atbs = read_atb_file(src_atb_file, atb_dicts)
        for i in src_atbs:
            src_atbs[i] = [src_atbs[i][j] for j in good_indices]
    else:
        src_atbs = None

    if tgt_atb_file:
        tgt_atbs = read_atb_file(tgt_atb_file, atb_dicts)
        for i in tgt_atbs:
            tgt_atbs[i] = [tgt_atbs[i][j] for j in good_indices]
    else:
        tgt_atbs = None

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        src_sizes = [src_sizes[idx] for idx in perm]
        tgt_sizes = [tgt_sizes[idx] for idx in perm]

    print('... sorting sentences by size')

    ids = list(range(len(src)))
    z = zip(src_sizes, tgt_sizes, ids)

    # ultimately sort by target size
    sorted_z = sorted(sorted(z, key=lambda x: x[0]), key=lambda x: x[1])

    ids = [z_[-1] for z_ in sorted_z]

    src = [src[id] for id in ids]
    tgt = [tgt[id] for id in ids]

    if src_atbs:
        for i in src_atbs:
            src_atbs[i] = [src_atbs[i][j] for j in ids]
    if tgt_atbs:
        for i in tgt_atbs:
            tgt_atbs[i] = [tgt_atbs[i][j] for j in ids]

    print(('Prepared %d sentences ' +
           '(%d ignored due to length == 0 or src len > %d or tgt len > %d)') %
          (len(src), ignored, max_src_length, max_tgt_length))

    output_dict['src'] = src
    output_dict['tgt'] = tgt
    output_dict['src_atbs'] = src_atbs
    output_dict['tgt_atbs'] = tgt_atbs

    return output_dict


def make_asr_data(src_file, tgt_file, tgt_dicts,
                  src_atb_file=None, tgt_atb_file=None,
                  max_src_length=64, max_tgt_length=64,
                  input_type='word',
                  stride=1, concat=1, prev_context=0,
                  fp16=False, reshape=True, asr_format="h5"):
    src, tgt = [], []
    src_sizes = []
    tgt_sizes = []
    count, ignored = 0, 0
    n_unk_words = 0

    output_dict = dict()

    print('Processing %s & %s ...' % (src_file, tgt_file))

    if asr_format == "h5":
        fileIdx = -1;
        if src_file[-2:] == "h5":
            srcf = h5.File(src_file, 'r')
        else:
            fileIdx = 0
            srcf = h5.File(src_file + "." + str(fileIdx) + ".h5", 'r')

    elif asr_format == "scp":
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

        if asr_format == "h5":
            if str(index) in srcf:
                featureVectors = np.array(srcf[str(index)])
            elif fileIdx != -1:
                srcf.close()
                fileIdx += 1
                srcf = h5.File(src_file + "." + str(fileIdx) + ".h5", 'r')
                featureVectors = np.array(srcf[str(index)])
            else:
                print("No feature vector for index:", index, file=sys.stderr)
                exit(-1)
        elif asr_format == "scp":
            _, featureVectors = next(audio_data)

        if stride == 1:
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

            # convert the line to half precision to save 50% memory
            if fp16:
                sline = sline.half()
            src += [sline]

            tgt_tensor = tgt_dicts.convertToIdx(tgt_words,
                                                onmt.Constants.UNK_WORD,
                                                onmt.Constants.BOS_WORD,
                                                onmt.Constants.EOS_WORD)
            tgt += [tgt_tensor]
            src_sizes += [len(sline)]
            tgt_sizes += [len(tgt_words)]

            unks = tgt_tensor.eq(onmt.Constants.UNK).sum().item()
            n_unk_words += unks

            if unks > 0:
                if "<unk>" not in tline:
                    print("DEBUGGING: This line contains UNK: %s" % tline)

        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    if asr_format == "h5":
        srcf.close()
    tgtf.close()

    print('Total number of unk words: %d' % n_unk_words)

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        src_sizes = [src_sizes[idx] for idx in perm]
        tgt_sizes = [tgt_sizes[idx] for idx in perm]

    print('... sorting sentences by size')

    # _, perm = torch.sort(torch.Tensor(sizes), descending=(opt.sort_type == 'descending'))
    # src = [src[idx] for idx in perm]
    # tgt = [tgt[idx] for idx in perm]
    z = zip(src, tgt, src_sizes, tgt_sizes)

    # ultimately sort by source size
    sorted_z = sorted(sorted(z, key=lambda x: x[3]), key=lambda x: x[2])

    src = [z_[0] for z_ in sorted_z]
    tgt = [z_[1] for z_ in sorted_z]

    print(('Prepared %d sentences ' +
           '(%d ignored due to length == 0 or src len > %d or tgt len > %d)') %
          (len(src), ignored, max_src_length, max_tgt_length))

    output_dict['src'] = src
    output_dict['tgt'] = tgt

    return output_dict


def collect_attributes(atb_files):
    # the files can contain multiple attributes
    # each of them will be stored in one onmt.Dict
    print("* Reading attributes ...")
    atb_dicts = dict()

    for file_ in atb_files:

        if not file_:
            continue

        reader = open(file_)

        while True:
            line = reader.readline()

            # normal end of file
            if line == "":
                break

            # attributes are split by space
            atbs = line.strip().split()

            for i, atb in enumerate(atbs):
                if i not in atb_dicts:
                    atb_dicts[i] = onmt.Dict()

                atb_dicts[i].add(atb)

    return atb_dicts


def main():
    dicts = {}

    # READING IN ATTRIBUTES
    if opt.train_tgt_atb or opt.train_src_atb:

        if opt.train_tgt_atb:
            assert (len(opt.valid_tgt_atb) > 0)

        if opt.train_src_atb:
            assert (len(opt.valid_src_atb) > 0)

        # the dicts['atb'] should be a collection of dicts, each element is a onmt.Dict of attributes

        dicts['atb'] = collect_attributes([opt.train_src_atb, opt.train_tgt_atb])

        print("Found %d types of attributes in the dataset" % len(dicts['atb']))

        for i in dicts['atb']:
            print("Found %d items in attribute type %d " % (dicts['atb'][i].size(), i))

    else:
        dicts['atb'] = None

    # for ASR and LM we only need to build vocab for the 'target' language
    if opt.asr or opt.lm:
        dicts['tgt'] = init_vocab('target', opt.train_tgt, opt.tgt_vocab,
                                  opt.tgt_vocab_size, input_type=opt.input_type)
    elif opt.join_vocab:
        dicts['src'] = init_vocab('source', [opt.train_src, opt.train_tgt], opt.src_vocab,
                                  opt.tgt_vocab_size, join=True, input_type=opt.input_type)
        dicts['tgt'] = dicts['src']

    else:
        dicts['src'] = init_vocab('source', opt.train_src, opt.src_vocab,
                                  opt.src_vocab_size, input_type=opt.input_type)

        dicts['tgt'] = init_vocab('target', opt.train_tgt, opt.tgt_vocab,
                                  opt.tgt_vocab_size, input_type=opt.input_type)

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
        output = make_asr_data(opt.train_src, opt.train_tgt,
                               dicts['tgt'],
                               max_src_length=opt.src_seq_length,
                               max_tgt_length=opt.tgt_seq_length,
                               input_type=opt.input_type,
                               stride=opt.stride, concat=opt.concat,
                               prev_context=opt.previous_context,
                               fp16=opt.fp16,
                               reshape=(opt.reshape_speech == 1),
                               asr_format=opt.asr_format)

        train['src'] = output['src']
        train['tgt'] = output['tgt']

        print('Preparing validation ...')
        valid = dict()
        output = make_asr_data(opt.valid_src, opt.valid_tgt,
                               dicts['tgt'],
                               max_src_length=max(1024, opt.src_seq_length),
                               max_tgt_length=max(1024, opt.tgt_seq_length),
                               input_type=opt.input_type,
                               stride=opt.stride, concat=opt.concat,
                               prev_context=opt.previous_context,
                               fp16=opt.fp16, reshape=(opt.reshape_speech == 1), asr_format=opt.asr_format)

        valid['src'], valid['tgt'] = output['src'], output['tgt']

    else:
        print('Preparing training translation model...')
        train = dict()
        output = make_translation_data(opt.train_src, opt.train_tgt,
                                       dicts['src'], dicts['tgt'],
                                       opt.train_src_atb, opt.train_tgt_atb, dicts['atb'],
                                       max_src_length=opt.src_seq_length,
                                       max_tgt_length=opt.tgt_seq_length,
                                       sort_by_target=opt.sort_by_target,
                                       input_type=opt.input_type)

        train['src'], train['tgt'] = output['src'], output['tgt']
        train['src_atbs'], train['tgt_atbs'] = output['src_atbs'], output['tgt_atbs']

        print('Preparing validation ...')
        valid = dict()
        output = make_translation_data(opt.valid_src, opt.valid_tgt,
                                       dicts['src'], dicts['tgt'],
                                       opt.valid_src_atb, opt.valid_tgt_atb, dicts['atb'],
                                       max_src_length=max(1024, opt.src_seq_length),
                                       max_tgt_length=max(1024, opt.tgt_seq_length),
                                       input_type=opt.input_type)

        valid['src'], valid['tgt'] = output['src'], output['tgt']
        valid['src_atbs'], valid['tgt_atbs'] = output['src_atbs'], output['tgt_atbs']

    if opt.src_vocab is None and opt.asr == False and opt.lm == False:
        save_vocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        save_vocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')

    if opt.format == 'raw':

        print('Saving data to \'' + opt.save_data + '.train.pt\'...')
        save_data = {'dicts': dicts,
                     'type': opt.src_type,
                     'train': train,
                     'valid': valid}
        torch.save(save_data, opt.save_data + '.train.pt')
        print("Done")

    elif opt.format == 'bin':
        print('Saving data to indexed data files')

        if opt.asr:
            print("ASR data format isn't compatible with binary")
            raise AssertionError
        # save dicts in this format
        torch.save(dicts, opt.save_data + '.dict.pt')

        # binarize the training set first
        for set in ['src', 'tgt']:

            if train[set] is None:
                continue
            dtype = np.int32

            if set == 'src' and opt.asr:
                dtype = np.double

            data = IndexedDatasetBuilder(opt.save_data + ".train.%s.bin" % set, dtype=dtype)

            # add item from training data to the indexed data
            for tensor in train[set]:
                data.add_item(tensor)

            data.finalize(opt.save_data + ".train.%s.idx" % set)

        # binarize the validation set
        for set in ['src', 'tgt']:

            if valid[set] is None:
                continue

            dtype = np.int32

            if set == 'src' and opt.asr:
                dtype = np.double

            data = IndexedDatasetBuilder(opt.save_data + ".valid.%s.bin" % set, dtype=dtype)

            # add item from training data to the indexed data
            for tensor in valid[set]:
                data.add_item(tensor)

            data.finalize(opt.save_data + ".valid.%s.idx" % set)

        print("Done")

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()