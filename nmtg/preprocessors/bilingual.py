import os

import numpy as np

from nmtg.data import Dictionary
from nmtg.preprocessors import register_preprocessor
from nmtg.preprocessors.preprocessor import NLPPreprocessor


@register_preprocessor('bilingual')
class BilingualPreprocessor(NLPPreprocessor):

    @classmethod
    def add_options(cls, parser):
        super().add_options(parser)

        parser.add_argument('in_src')
        parser.add_argument('in_tgt')

        parser.add_argument('-join_vocab', action='store_true',
                            help='Share dictionary for source and target')
        parser.add_argument('-src_vocab_size', type=int, default=50000,
                            help='Size of the source vocabulary')
        parser.add_argument('-tgt_vocab_size', type=int, default=50000,
                            help='Size of the target vocabulary')
        parser.add_argument('-vocab_threshold', type=int,
                            help='Discard vocabulary words that occur less often than this threshold')

    @classmethod
    def preprocess(cls, args, save_data=True):
        split_words = args.input_type == 'word'

        os.makedirs(args.data_dir_out, exist_ok=True)
        src_name = os.path.basename(args.in_src)
        tgt_name = os.path.basename(args.in_tgt)

        (src_offsets, src_lengths, src_counter), \
        (tgt_offsets, tgt_lengths, tgt_counter) = \
            cls.get_indices_and_vocabulary((args.in_src, args.in_tgt),
                                           split_words,
                                           args.lower,
                                           not args.no_progress,
                                           args.report_every)

        out_offsets_src = os.path.join(args.data_dir_out, src_name + '.idx.npy')
        out_lengths_src = os.path.join(args.data_dir_out, src_name + '.len.npy')
        np.save(out_offsets_src, src_offsets)
        np.save(out_lengths_src, src_lengths)

        src_dictionary = Dictionary()
        for word, count in src_counter.items():
            src_dictionary.add_symbol(word, count)

        out_offsets_tgt = os.path.join(args.data_dir_out, tgt_name + '.idx.npy')
        out_lengths_tgt = os.path.join(args.data_dir_out, tgt_name + '.len.npy')
        np.save(out_offsets_tgt, tgt_offsets)
        np.save(out_lengths_tgt, tgt_lengths)

        tgt_dictionary = Dictionary()
        for word, count in tgt_counter.items():
            tgt_dictionary.add_symbol(word, count)

        if args.join_vocab:
            # If we explicitly load a target dictionary to merge
            # or we are inferring both dictionaries
            if args.tgt_vocab is not None or args.src_vocab is None:
                src_dictionary.update(tgt_dictionary)
            src_dictionary.finalize(nwords=args.src_vocab_size,
                                    threshold=args.vocab_threshold or -1)

            if save_data:
                src_dictionary.save(os.path.join(args.data_dir_out, 'dict'))
            else:
                return src_dictionary, src_dictionary
        else:
            src_dictionary.finalize(nwords=args.src_vocab_size,
                                    threshold=args.vocab_threshold or -1)
            tgt_dictionary.finalize(nwords=args.tgt_vocab_size,
                                    threshold=args.vocab_threshold or -1)

            if save_data:
                src_dictionary.save(os.path.join(args.data_dir_out, 'src.dict'))
                tgt_dictionary.save(os.path.join(args.data_dir_out, 'tgt.dict'))
            else:
                return src_dictionary, tgt_dictionary


