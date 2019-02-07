import logging
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
        parser.add_argument('-batch_size_words', type=int, default=2048,
                            help='Maximum number of words in a batch')
        parser.add_argument('-batch_size_sents', type=int, default=128,
                            help='Maximum number of sentences in a batch')
        parser.add_argument('-input_type', default='word', choices=['word', 'char'],
                            help='Word or character-based input')

        parser.add_argument('-output', default='pred.txt',
                            help="Path to output the predictions (each line will be the decoded sequence")

    def __init__(self, args, src_dataset, tgt_dataset=None):
        super().__init__(args, src_dataset)
        self.src_dataset = src_dataset
        self.tgt_dataset = tgt_dataset

    @classmethod
    def setup_task(cls, args):
        logger.info("Loading validation data")

        src_dataset = TextLineDataset.load_into_memory(args.valid_src)
        tgt_dataset = None

        if args.valid_tgt is not None:
            tgt_dataset = TextLineDataset.load_into_memory(args.valid_tgt)

        logger.info('Number of validation sentences: {}'.format(len(src_dataset)))

        return cls(args, src_dataset, tgt_dataset)

    def score_results(self, results):
        # TODO: Calculate BLEU here
        return []

    def save_results(self, results):
        with open(self.args.output, 'w') as out:
            out.writelines(r + '\n' for r in results)
