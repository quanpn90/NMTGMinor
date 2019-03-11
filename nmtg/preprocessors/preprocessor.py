import logging
from collections import Counter

from tqdm import tqdm


logger = logging.getLogger(__name__)


class Preprocessor:
    @classmethod
    def add_options(cls, parser):
        parser.add_argument('-data_dir_out', type=str, required=True,
                            help='Output directory for auxiliary data')
        parser.add_argument('-report_every', type=int, default=100000,
                            help='Report status every this many sentences')

    @classmethod
    def preprocess(cls, args, save_data=True):
        raise NotImplementedError


class NLPPreprocessor(Preprocessor):

    @classmethod
    def add_options(cls, parser):
        super().add_options(parser)
        parser.add_argument('-lower', action='store_true',
                            help='Construct a lower-case vocabulary')
        parser.add_argument('-input_type', default='word', choices=['word', 'char'],
                            help='Type of dictionary to create.')

    @staticmethod
    def get_indices_and_vocabulary(filenames, split_words=True, lower=False, progress_bar=True, report_every=100000):
        offsets = [[0] for _ in filenames]
        lengths = [[] for _ in filenames]
        counters = [Counter() for _ in filenames]

        file_handles = [open(filename) for filename in filenames]

        try:
            with tqdm(unit='lines', disable=not progress_bar) as pbar:
                i = 0
                lines = [f.readline() for f in file_handles]
                while all(line != '' for line in lines):
                    proc_lines = [line.rstrip() for line in lines]

                    if lower:
                        proc_lines = [line.lower() for line in lines]
                    if split_words:
                        proc_lines = [line.split() for line in lines]

                    for offset, f in zip(offsets, file_handles):
                        offset.append(f.tell())

                    for length, counter, line in zip(lengths, counters, proc_lines):
                        ll = len(line)
                        if ll == 0:
                            logger.warning('Empty line {}'.format(i))
                        length.append(ll)
                        counter.update(line)

                    i += 1

                    if i % report_every == 0:
                        logger.info('{:,} lines processed'.format(i))

                    lines = [f.readline() for f in file_handles]
                    pbar.update()

                if any(line != '' for line in lines):
                    logger.warning('Files are not the same length')
        finally:
            for f in file_handles:
                f.close()

        return list(zip(offsets, lengths, counters))
