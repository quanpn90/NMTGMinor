import onmt
import onmt.Markdown
import argparse
import torch

from onmt.data_utils.IndexedDataset import IndexedDatasetBuilder


def loadImageLibs():
    "Conditional import of torch image libs."
    global Image, transforms
    from PIL import Image
    from torchvision import transforms


parser = argparse.ArgumentParser(description='preprocess.py')
onmt.Markdown.add_md_help_argument(parser)

# **Preprocess Options**

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-src_type', default="text",
                    help="Type of the source input. Options are [text|img].")
parser.add_argument('-sort_type', default="ascending",
                    help="Type of sorting. Options are [ascending|descending].")
parser.add_argument('-input_type', default="word",
                    help="Input type: word/char")
parser.add_argument('-format', default="bin",
                    help="Save data format: binary or raw. Binary should be used to load faster")


parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")
parser.add_argument('-valid_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=True,
                    help="Path to the validation target data")

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
parser.add_argument('-attb_vocab',
                    help="Path to an existing attribute vocabulary")
parser.add_argument('-src_seq_length', type=int, default=64,
                    help="Maximum source sequence length")
parser.add_argument('-src_seq_length_trunc', type=int, default=0,
                    help="Truncate source sequence length.")
parser.add_argument('-tgt_seq_length', type=int, default=66,
                    help="Maximum target sequence length to keep.")
parser.add_argument('-tgt_seq_length_trunc', type=int, default=0,
                    help="Truncate target sequence length.")

parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
                    
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')
parser.add_argument('-remove_duplicate', action='store_true', help='remove duplicated sequences')
parser.add_argument('-join_vocab', action='store_true', help='Using one dictionary for both source and target')
parser.add_argument('-bos_word', default="default",
                    help="Type of the source input. Options are [text|img].")

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()

torch.manual_seed(opt.seed)


"""
About the data format used in this project: 

- Multilingual dataset are jointly concatenated without the need of a multilingual dataset
- Each source and target sentence has a language attribute assigned to it (src is #en#, tgt is #de# for example)
- We need to create word dictionary (for source and target) and attb dictionary

"""

#
# def make_attb_vocabulary(filenames):
#
#     vocab = onmt.Dict()
#
#     for filename in filenames:
#         print("Reading file %s ... for attribute reading" % filename)
#         with open(filename) as f:
#             for sent in f.readlines():
#                 # the first token (#de#) is attribute
#                 attb = sent.strip().split()[0]
#                 vocab.add(attb)
#
#     return vocab


def make_join_vocabulary(filenames, size, input_type="word", dict_atb=None):
    
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD],
                      lower=opt.lower)

    if dict_atb is None:
        dict_atb = onmt.Dict()
    
    for filename in filenames:
        print("Reading file %s ... " % filename)
        with open(filename) as f:
            for sent in f.readlines():
                
                if input_type == "word":
                    tokens = sent.split()
                    attb = tokens[0] # the first token is the language atb
                    words = tokens[1:] # normal words from the second token
                    for word in words:
                        vocab.add(word)
                    dict_atb.add(attb)
                elif input_type == "char":
                    tokens = sent.split()
                    attb = tokens[0]

                    sent_wo_atb = " ".join(tokens[1:])
                    for char in sent_wo_atb:
                        vocab.add(char)

                    dict_atb.add(attb)
                else:
                    raise NotImplementedError("Input type not implemented")

    original_size = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), original_size))

    return vocab, dict_atb


def make_vocabulary(filename, size, input_type='word', dict_atb=None):
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD],
                      lower=opt.lower)

    if dict_atb is None:
        dict_atb = onmt.Dict()

    with open(filename) as f:
        for sent in f.readlines():
            if input_type == "word":
                tokens = sent.strip().split()
                attb = tokens[0]  # the first token is the language atb
                words = tokens[1:]  # normal words from the second token
                for word in words:
                    vocab.add(word)
                dict_atb.add(attb)
            elif input_type == "char":
                tokens = sent.strip().split()
                attb = tokens[0]

                sent_wo_atb = " ".join(tokens[1:])
                for char in sent_wo_atb:
                    vocab.add(char)

                dict_atb.add(attb)
            else:
                raise NotImplementedError("Input type not implemented")

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab, dict_atb


def init_vocabulary(name, dataFile, vocabFile, vocabSize, join=False, input_type='word', dict_atb=None):

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
            gen_word_vocab = make_join_vocabulary(dataFile, vocabSize, input_type=input_type, dict_atb=dict_atb)
        else:
            print('Building ' + name + ' vocabulary...')
            gen_word_vocab = make_vocabulary(dataFile, vocabSize, input_type=input_type, dict_tab=dict_atb)

        vocab = gen_word_vocab

    print()
    return vocab


def save_vocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(srcFile, tgtFile, src_dicts, tgt_dicts, atb_dict, max_src_length=64, max_tgt_length=64,
             input_type='word', remove_duplicate=False, bos_word="default"):
    src, tgt = [], []
    sizes = []
    count, ignored = 0, 0
    n_duplicate = 0
    src_sizes = []
    tgt_sizes = []
    src_attbs = []
    tgt_attbs = []

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile)
    tgtF = open(tgtFile)

    if bos_word == "default":
        bos_word = onmt.Constants.BOS_WORD

    elif bos_word == "none":
        print(" * Warning: no BOS WORD used in data preprocessing!")
        bos_word = None

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            print('WARNING: src and tgt do not have the same # of sentences')
            break

        # note: here we have to remove the
        sline = sline.strip()
        tline = tline.strip()

        src_words = sline.split()
        tgt_words = tline.split()

        src_attb = src_words[0]
        tgt_attb = tgt_words[0]

        sline = sline[1:]
        tline = tline[1:]

        # source and/or target are empty
        if remove_duplicate:
            if sline == tline:
                n_duplicate += 1
                # ~ print('WARNING: ignoring a duplicated pair ('+str(count+1)+')')
                continue
            
        if sline == "" or tline == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            continue
        
        if input_type == 'word':
            src_words = src_words[1:]
            tgt_words = tgt_words[1:]
        elif input_type == 'char':
            src_words = list(" ".join(src_words[1:]))
            tgt_words = list(" ".join(tgt_words[1:]))

        if len(src_words) <= max_src_length \
           and len(tgt_words) <= max_tgt_length - 2:

            # Check truncation condition.
            if opt.src_seq_length_trunc != 0:
                src_words = src_words[:opt.src_seq_length_trunc]
            if opt.tgt_seq_length_trunc != 0:
                tgt_words = tgt_words[:opt.tgt_seq_length_trunc]

            src_sent = src_dicts.convertToIdx(src_words,
                                              onmt.Constants.UNK_WORD)

            src_attb = atb_dict.convertToIdx([src_attb], None)

            src += [src_sent]
            src_attbs += [src_attb]

            tgt_sent = tgt_dicts.convertToIdx(tgt_words,
                                          onmt.Constants.UNK_WORD,
                                          bosWord=bos_word,
                                          eosWord=onmt.Constants.EOS_WORD)

            # convert the atb into index
            # this should be a vector of 1 element
            tgt_attb = atb_dict.convertToIdx([tgt_attb], None)

            tgt += [tgt_sent]
            tgt_attbs += [tgt_attb]

            src_sizes += [len(src_sent)]
            tgt_sizes += [len(tgt_sent)]

        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)
    
    if remove_duplicate:
        print(' ... %d sentences removed for duplication' % n_duplicate)
    srcF.close()
    tgtF.close()

    print('... sorting sentences by size')
    z = zip(src, tgt, src_sizes, tgt_sizes, src_attbs, tgt_attbs)

    # sort by source first, then sort by target
    # should we sort by descending order or ascending order (the default is ascending)
    sorted_by_src_z = sorted(z, key=lambda x: x[2])
    sorted_by_tgt_z = sorted(sorted_by_src_z, key=lambda x: x[3])

    src = [z_[0] for z_ in sorted_by_tgt_z]
    tgt = [z_[1] for z_ in sorted_by_tgt_z]

    src_attbs = [z_[4] for z_ in sorted_by_tgt_z]
    tgt_attbs = [z_[5] for z_ in sorted_by_tgt_z]

    print(('Prepared %d sentences ' +
          '(%d ignored due to error or src len > %d or tgt len > %d)') %
          (len(src), ignored, max_src_length, max_tgt_length))

    return src, tgt, src_attbs, tgt_attbs


def main():

    dicts = dict()

    # first, we read the attb
    dicts['atb'] = None
    if opt.attb_vocab is not None:
        # If given, load existing word dictionary.
        print('Reading attribute vocabulary from \'' + opt.attb_vocab + '\'...')
        dicts['atb'] = onmt.Dict()
        dicts['atb'].loadFile(opt.attb_vocab)
        print('Loaded ' + str(dicts['atb'].size()) + ' ' + name + ' attributes')

    if opt.join_vocab:
        dicts['src'], dicts['atb'] = init_vocabulary('joined', [opt.train_src, opt.train_tgt], opt.src_vocab,
                                      opt.tgt_vocab_size, join=True, input_type=opt.input_type, dict_atb=dicts['atb'])

        dicts['tgt'] = dicts['src']
    else:
        dicts['src'], dicts['atb'] = init_vocabulary('source', opt.train_src, opt.src_vocab,
                                      opt.src_vocab_size, input_type=opt.input_type, dict_atb=dicts['atb'])

        dicts['tgt'], dicts['atb'] = init_vocabulary('target', opt.train_tgt, opt.tgt_vocab,
                                      opt.tgt_vocab_size, input_type=opt.input_type, dict_atb=dicts['atb'])

    print("Created a language attribute dictionary with %d languages" % dicts['atb'].size())

    train = {}
    valid = {}

    train['src'], train['tgt'] = dict(), dict()

    print('Preparing training ...')
        
    train['src']['words'], train['tgt']['words'], train['src']['attbs'], train['tgt']['attbs'] = makeData(
                                          opt.train_src, opt.train_tgt,
                                          dicts['src'], dicts['tgt'], dicts['atb'],
                                          max_src_length=opt.src_seq_length,
                                          max_tgt_length=opt.tgt_seq_length, 
                                          input_type=opt.input_type,
                                          remove_duplicate=opt.remove_duplicate,
                                          bos_word=opt.bos_word)

    print('Preparing validation ...')

    valid['src'], valid['tgt'] = dict(), dict()

    valid['src']['words'], valid['tgt']['words'], valid['src']['attbs'], valid['tgt']['attbs'] = makeData(
                                          opt.valid_src, opt.valid_tgt,
                                          dicts['src'], dicts['tgt'], dicts['atb'],
                                          max_src_length=9999,
                                          max_tgt_length=9999,
                                          input_type=opt.input_type,
                                          bos_word=opt.bos_word)

    # saving data to disk
    # save dicts in this format
    torch.save(dicts, opt.save_data + '.dict.pt')

    # save dictionary in readable format
    if opt.src_vocab is None:
        save_vocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        save_vocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')
    if opt.attb_vocab is None:
        save_vocabulary('atb', dicts['atb'], opt.save_data + '.atb.dict')

    # saving data
    for set in ['src', 'tgt']:
        data = IndexedDatasetBuilder(opt.save_data + ".train.words.%s.bin" % set)
        attb_data = IndexedDatasetBuilder(opt.save_data + ".train.langs.%s.bin" % set)

        # add item from training data to the indexed data
        for tensor_sent in train[set]['words']:
            data.add_item(tensor_sent)

        for attb in train[set]['attbs']:
            attb_data.add_item(attb)

        data.finalize(opt.save_data + ".train.words.%s.idx" % set)
        attb_data.finalize(opt.save_data + ".train.langs.%s.idx" % set)

    for set in ['src', 'tgt']:
        data = IndexedDatasetBuilder(opt.save_data + ".valid.words.%s.bin" % set)
        attb_data = IndexedDatasetBuilder(opt.save_data + ".valid.langs.%s.bin" % set)

        # add item from training data to the indexed data
        for tensor_sent in valid[set]['words']:
            data.add_item(tensor_sent)

        for attb in valid[set]['attbs']:
            attb_data.add_item(attb)

        data.finalize(opt.save_data + ".valid.words.%s.idx" % set)
        attb_data.finalize(opt.save_data + ".valid.langs.%s.idx" % set)

    print("Done")


if __name__ == "__main__":
    main()
