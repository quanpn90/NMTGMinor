import os

import numpy as np

from nmtg.data import Dictionary
from nmtg.preprocessors import register_preprocessor
from nmtg.preprocessors.preprocessor import NLPPreprocessor


@register_preprocessor('multilingual')
class MultilingualPreprocessor(NLPPreprocessor):
    @classmethod
    def add_options(cls, parser):
        super().add_options(parser)
        parser.add_argument('in_data')
        parser.add_argument('-langs', nargs='+',
                            help='Language pairs (src-tgt) or languages')

        parser.add_argument('-join_lang_vocab', action='sore_true',
                            help='Share vocabularies across languages')
        parser.add_argument('-join_src_tgt_vocab', action='store_true',
                            help='Share vocabularies for source and target side')
        parser.add_argument('-vocab_size', type=int, default=50000,
                            help='Size of the source vocabulary')
        parser.add_argument('-vocab_threshold', type=int,
                            help='Discard vocabulary words that occur less often than this threshold')

    @classmethod
    def preprocess(cls, args, save_data=True):
        split_words = args.input_type == 'word'

        os.makedirs(args.data_dir_out, exist_ok=True)

        dictionaries = []
        dataset_lengths = []
        for filename in args.in_data:
            ((offsets, lengths, counter), ) = cls.get_indices_and_vocabulary([filename], split_words, args.lower,
                                                                             not args.no_progress, args.report_every)
            basename = os.path.basename(filename)
            out_offsets = os.path.join(args.data_dir_out, basename + '.idx.npy')
            out_lengths = os.path.join(args.data_dir_out, basename + '.len.npy')
            np.save(out_offsets, offsets)
            np.save(out_lengths, lengths)
            dictionaries.append(counter)
            dataset_lengths.append(len(lengths))

        pairs = len(args.langs[0].split('-')) == 2
        data_mode = 'pairs' if pairs else 'all_to_all'

        if data_mode == 'all_to_all':
            assert len(set(args.langs)) == len(args.langs)
            if not args.join_src_tgt_vocab:
                raise ValueError('In order to use all_to_all data mode, vocabularies must be shared across'
                                 'source and target languages')
            if not len(set(dataset_lengths)) == 1:
                raise ValueError('Datasets are not the same length')
            src_langs = args.langs
            tgt_langs = args.langs
            src_counters = dictionaries
            tgt_counters = dictionaries
        else:
            src_langs, tgt_langs = zip(*(lang.split('-') for lang in args.langs))
            src_counters, tgt_counters = {}, {}
            for lang, counter in zip(src_langs, dictionaries[::2]):
                if lang in src_counters:
                    src_counters[lang].update(counter)
                else:
                    src_counters[lang] = counter
            
            for lang, counter in zip(tgt_langs, dictionaries[1::2]):
                if lang in tgt_counters:
                    tgt_counters[lang].update(counter)
                else:
                    tgt_counters[lang] = counter

            src_langs = list(src_counters.keys())
            tgt_langs = list(tgt_counters.keys())
            src_counters = [src_counters[lang] for lang in src_langs]
            tgt_counters = [tgt_counters[lang] for lang in tgt_langs]

        if args.join_lang_vocab and args.join_src_tgt_vocab:
            dictionary = Dictionary.from_counters(*(src_counters + tgt_counters))
            dictionary.finalize(nwords=args.vocab_size, threshold=args.vocab_threshold or -1)
            dictionary.save(os.path.join(args.data_dir_out, 'dict'))

        elif args.join_lang_vocab and not args.join_src_tgt_vocab:
            src_dict = Dictionary.from_counters(*src_counters)
            tgt_dict = Dictionary.from_counters(*tgt_counters)
            src_dict.finalize(nwords=args.vocab_size, threshold=args.vocab_threshold or -1)
            src_dict.save(os.path.join(args.data_dir_out, 'src.dict'))
            tgt_dict.finalize(nwords=args.vocab_size, threshold=args.vocab_threshold or -1)
            tgt_dict.save(os.path.join(args.data_dir_out, 'tgt.dict'))

        elif not args.join_lang_vocab and args.join_src_tgt_vocab:
            vocabs = {}
            for lang, counter in zip(src_langs + tgt_langs, src_counters + tgt_counters):
                if lang in vocabs:
                    for word, count in counter:
                        vocabs[lang].add_symbol(word, count)
                else:
                    vocabs[lang] = Dictionary.from_counters(counter)
            for lang, vocab in vocabs.items():
                vocab.finalize(nwords=args.vocab_size, threshold=args.vocab_threshold or -1)
                vocab.save(os.path.join(args.data_dir_out, lang + '.dict'))
        else:
            vocabs = {}
            for lang, counter in zip(src_langs, src_counters):
                if lang + '.src' in vocabs:
                    voc = vocabs[lang + '.src']
                    for word, count in counter:
                        voc.add_symbol(word, count)
                else:
                    vocabs[lang + '.src'] = Dictionary.from_counters(counter)
            for lang, counter in zip(tgt_langs, tgt_counters):
                if lang + '.tgt' in vocabs:
                    voc = vocabs[lang + '.tgt']
                    for word, count in counter:
                        voc.add_symbol(word, count)
                else:
                    vocabs[lang + '.tgt'] = Dictionary.from_counters(counter)
            for lang, vocab in vocabs.items():
                vocab.finalize(nwords=args.vocab_size, threshold=args.vocab_threshold or -1)
                vocab.save(os.path.join(args.data_dir_out, lang + '.dict'))
