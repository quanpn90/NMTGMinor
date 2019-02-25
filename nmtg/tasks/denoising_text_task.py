import logging
from argparse import ArgumentParser

from nmtg.data import TextLineDataset
from nmtg.tasks import register_task
from nmtg.tasks.translation_task import TranslationTask


logger = logging.getLogger(__name__)


@register_task('denoising_text')
class DenoisingTextTask(TranslationTask):

    @staticmethod
    def add_options(parser: ArgumentParser):
        parser.add_argument('-valid_clean', required=True,
                            help='Filename for clean source file')
        parser.add_argument('-valid_noisy',
                            help='(Optional) Filename for file with pre-generated noise')
        parser.add_argument('-bpe_symbol', type=str, default='@@ ',
                            help='Strip this symbol from the output')
        parser.add_argument('-lower', action='store_true', help='lowercase data')
        parser.add_argument('-word_shuffle', type=int, default=3,
                            help='Maximum number of positions a word can move (0 to disable)')
        parser.add_argument('-word_blank', type=float, default=0.2,
                            help='Probability to replace a word with the unknown word (0 to disable)')
        parser.add_argument('-noise_word_dropout', type=float, default=0.1,
                            help='Probability to remove a word (0 to disable)')

    @classmethod
    def setup_task(cls, args):
        logger.info("Loading validation data")

        clean_dataset = TextLineDataset.load_into_memory(args.valid_clean)
        # We cannot make a NoisyTextDataset, because that would require a dictionary
        # We could do it on words ourselves, but that would be silly...

        if args.valid_noisy is not None:
            noisy_dataset = TextLineDataset.load_into_memory(args.valid_noisy)
        else:
            noisy_dataset = None

        logger.info('Number of validation sentences: {:,d}'.format(len(clean_dataset)))

        return cls(noisy_dataset or clean_dataset, clean_dataset, args.bpe_symbol, args.lower)
