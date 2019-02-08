import logging
import sacrebleu
from argparse import ArgumentParser

from nmtg.data import TextLineDataset
from nmtg.tasks import Task, register_task

logger = logging.getLogger(__name__)


@register_task('translation')
class TranslationTask(Task):
    @staticmethod
    def add_options(parser: ArgumentParser):
        parser.add_argument('-valid_src', required=True,
                            help='Path/filename prefix for source file')
        parser.add_argument('-valid_tgt',
                            help='Path/filename prefix for target file')
        parser.add_argument('-bpe_symbol', type=str, default='@@ ',
                            help='Strip this symbol from the output')
        parser.add_argument('-lower', action='store_true', help='lowercase data')

    def __init__(self, src_dataset, tgt_dataset=None, bpe_symbol='@@ ', lower=False):
        super().__init__(src_dataset)
        self.src_dataset = src_dataset
        self.tgt_dataset = tgt_dataset
        self.bpe_symbol = bpe_symbol
        self.lower = lower

    @classmethod
    def setup_task(cls, args):
        logger.info("Loading validation data")

        src_dataset = TextLineDataset.load_into_memory(args.valid_src)
        tgt_dataset = None

        if args.valid_tgt is not None:
            tgt_dataset = TextLineDataset.load_into_memory(args.valid_tgt)

        logger.info('Number of validation sentences: {}'.format(len(src_dataset)))

        return cls(src_dataset, tgt_dataset, args.bpe_symbol, args.lower)

    def score_results(self, results):
        if self.tgt_dataset is None:
            return []

        ref_stream = (line.replace(self.bpe_symbol, '') for line in self.tgt_dataset)
        sys_stream = (line.replace(self.bpe_symbol, '') for line in results[::len(results) // len(self.dataset)])
        bleu = sacrebleu.raw_corpus_bleu(sys_stream, ref_stream)
        return ["{:.2f} BLEU".format(bleu.score)]

    def save_results(self, results, out_filename):
        with open(out_filename, 'w') as out:
            out.writelines(r + '\n' for r in results)
