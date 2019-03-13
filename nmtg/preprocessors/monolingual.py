import os

import numpy as np

from nmtg.data import Dictionary
from nmtg.preprocessors import register_preprocessor
from nmtg.preprocessors.preprocessor import NLPPreprocessor


@register_preprocessor('monolingual')
class MonolingualPreprocessor(NLPPreprocessor):
    @classmethod
    def add_options(cls, parser):
        super().add_options(parser)
        parser.add_argument('in_data')

        parser.add_argument('-vocab_size', type=int, default=50000,
                            help='Size of the source vocabulary')
        parser.add_argument('-vocab_threshold', type=int,
                            help='Discard vocabulary words that occur less often than this threshold')

    @classmethod
    def preprocess(cls, args):
        split_words = args.input_type == 'word'

        os.makedirs(args.data_dir_out, exist_ok=True)
        basename = os.path.basename(args.in_data)

        ((offsets, lengths, counter),) = cls.get_indices_and_vocabulary([args.in_data], split_words, args.lower,
                                                                        not args.no_progress, args.report_every)

        out_offsets = os.path.join(args.data_dir_out, basename + '.idx.npy')
        out_lengths = os.path.join(args.data_dir_out, basename + '.len.npy')
        np.save(out_offsets, offsets)
        np.save(out_lengths, lengths)

        dictionary = Dictionary()
        for word, count in counter.items():
            dictionary.add_symbol(word, count)

        dictionary.finalize(nwords=args.vocab_size, threshold=args.vocab_threshold or -1)

        dictionary.save(os.path.join(args.data_dir_out, 'dict'))
