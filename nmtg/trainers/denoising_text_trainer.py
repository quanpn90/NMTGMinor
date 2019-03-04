import logging
import os

import numpy as np

from nmtg.data import Dictionary, TextLineDataset, ParallelDataset
from nmtg.data.data_utils import get_indices_and_vocabulary, generate_length_based_batches_from_lengths
from nmtg.data.noisy_text import NoisyTextDataset
from nmtg.data.samplers import PreGeneratedBatchSampler
from nmtg.data.text_lookup_dataset import TextLookupDataset
from nmtg.models.nmt_model import NMTModel
from nmtg.trainers import register_trainer
from nmtg.trainers.nmt_trainer import NMTTrainer
from nmtg.trainers.trainer import TrainData

logger = logging.getLogger()


@register_trainer('denoising_text')
class DenoisingTextTrainer(NMTTrainer):

    @classmethod
    def add_preprocess_options(cls, parser):
        super(NMTTrainer, cls).add_preprocess_options(parser)
        parser.add_argument('-train_clean', type=str, required=True,
                            help='Path to the clean training data')
        parser.add_argument('-train_noisy', type=str,
                            help='(Optional) training data with pre-generated noise. Further noise will be applied')

        parser.add_argument('-vocab', type=str,
                            help='Path to an existing vocabulary')
        parser.add_argument('-data_dir_out', type=str, required=True,
                            help='Output directory for auxiliary data')
        parser.add_argument('-lower', action='store_true',
                            help='Construct a lower-case vocabulary')
        parser.add_argument('-vocab_threshold', type=int,
                            help='Discard vocabulary words that occur less often than this threshold')

        parser.add_argument('-vocab_size', type=int, default=50000,
                            help='Size of the source vocabulary')
        parser.add_argument('-input_type', default='word', choices=['word', 'char'],
                            help='Type of dictionary to create.')
        parser.add_argument('-report_every', type=int, default=100000,
                            help='Report status every this many sentences')

    @classmethod
    def add_training_options(cls, parser):
        # Skip NMTTrainer's arguments, because it adds the *required* argument train_tgt, which we don't need
        super(NMTTrainer, cls).add_training_options(parser)
        NMTModel.add_options(parser)
        parser.set_defaults(join_embedding=True, join_vocab=True)

        parser.add_argument('-train_clean', type=str, required=True,
                            help='Path to the clean training data')
        parser.add_argument('-train_noisy', type=str,
                            help='(Optional) training data with pre-generated noise. Further noise will be applied')
        parser.add_argument('-data_dir', type=str, required=True,
                            help='Path to an auxiliary data')
        parser.add_argument('-load_into_memory', action='store_true',
                            help='Load the dataset into memory')
        parser.add_argument('-batch_size_words', type=int, default=2048,
                            help='Maximum number of words in a batch')
        parser.add_argument('-batch_size_sents', type=int, default=128,
                            help='Maximum number of sentences in a batch')
        parser.add_argument('-batch_size_multiplier', type=int, default=1,
                            help='Number of sentences in a batch must be divisible by this number')
        parser.add_argument('-pad_count', action='store_true',
                            help='Count padding words when batching')
        parser.add_argument('-seq_length', type=int, default=64,
                            help='Discard examples with a source sequence length above this value')
        parser.add_argument('-seq_length_trunc', type=int, default=0,
                            help='Truncate source sequences to this length. 0 (default) to disable')

    @staticmethod
    def preprocess(args):
        split_words = args.input_type == 'word'

        os.makedirs(args.data_dir_out, exist_ok=True)
        train_clean_name = os.path.basename(args.train_clean)
        source_files = [args.train_clean]
        if args.train_noisy is not None:
            source_files.append(args.train_noisy)

        outputs = get_indices_and_vocabulary(source_files, split_words, args.lower,
                                             not args.no_progress, args.report_every)

        if args.train_noisy is not None:
            train_noisy_name = os.path.basename(args.train_noisy)
            (offsets, lengths, counter), \
            (noisy_offsets, noisy_lengths, noisy_counter) = outputs
            counter.update(noisy_counter)

            noisy_offset_filename = os.path.join(args.data_dir_out, train_noisy_name + '.idx.npy')
            np.save(noisy_offset_filename, noisy_offsets)
        else:
            ((offsets, lengths, counter),) = outputs

        out_offsets = os.path.join(args.data_dir_out, train_clean_name + '.idx.npy')
        out_lengths = os.path.join(args.data_dir_out, train_clean_name + '.len.npy')
        np.save(out_offsets, offsets)
        np.save(out_lengths, lengths)
        if args.vocab is not None:
            dictionary = Dictionary.load(args.vocab)
        else:
            dictionary = Dictionary()
            for word, count in counter.items():
                dictionary.add_symbol(word, count)

        dictionary.finalize(nwords=args.vocab_size, threshold=args.vocab_threshold or -1)
        dictionary.save(os.path.join(args.data_dir_out, 'dict'))

    def __init__(self, args):
        super().__init__(args)
        self.dictionary = self.src_dict

    def _load_noisy_data(self):
        logger.info('Loading training data')
        split_words = self.args.input_type == 'word'

        clean_data, lengths = TextLookupDataset.load(self.args.train_clean, self.dictionary, self.args.data_dir,
                                                     self.args.load_into_memory, split_words,
                                                     bos=True, eos=True, trunc_len=self.args.seq_length_trunc,
                                                     lower=self.args.lower)

        if self.args.train_noisy is not None:
            noisy_data, _ = TextLookupDataset.load(self.args.train_noisy, self.dictionary, self.args.data_dir,
                                                   self.args.load_into_memory, split_words,
                                                   bos=False, eos=False, trunc_len=self.args.seq_length_trunc,
                                                   lower=self.args.lower)
        else:
            noisy_data = TextLookupDataset(clean_data.source, self.dictionary, words=split_words, bos=False, eos=False,
                                           trunc_len=self.args.seq_length_trunc, lower=self.args.lower)

        noisy_data = NoisyTextDataset(
            noisy_data,
            self.args.word_shuffle,
            self.args.word_dropout,
            self.args.word_blank,
            self.args.bpe_symbol)

        dataset = ParallelDataset(noisy_data, clean_data)
        logger.info('Number of training sentences: {:,d}'.format(len(dataset)))

        def filter_fn(i):
            return lengths[i] <= self.args.seq_length

        logger.info('Generating batches')
        batches = generate_length_based_batches_from_lengths(
            lengths, self.args.batch_size_words,
            self.args.batch_size_sents,
            self.args.batch_size_multiplier,
            self.args.pad_count,
            filter_fn=filter_fn)
        logger.info('Number of training batches: {:,d}'.format(len(batches)))

        filtered = len(lengths) - sum(len(batch) for batch in batches)
        logger.info('Filtered {:,d}/{:,d} training examples for length'.format(filtered, len(lengths)))

        sampler = PreGeneratedBatchSampler(batches, self.args.curriculum == 0)
        return dataset, sampler

    def load_data(self, model_args=None):
        dataset, sampler = self._load_noisy_data()

        model = self.build_model(model_args)
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        lr_scheduler, optimizer = self._build_optimizer(params)
        return TrainData(model, dataset, sampler, lr_scheduler, optimizer, self._get_training_metrics())

    def _get_eval_iterator(self, task):
        split_words = self.args.input_type == 'word'
        noisy_data = NoisyTextDataset(
            TextLookupDataset(task.src_dataset, self.dictionary, split_words, bos=False, eos=False,
                              lower=self.args.lower),
            self.args.word_shuffle,
            self.args.noise_word_dropout,
            self.args.word_blank,
            self.args.bpe_symbol
        )

        clean_data = TextLookupDataset(task.tgt_dataset, self.dictionary, split_words, bos=True, eos=True,
                                       lower=self.args.lower)

        dataset = ParallelDataset(noisy_data, clean_data)
        return dataset.get_iterator(batch_size=self.args.batch_size,
                                    num_workers=self.args.data_loader_threads,
                                    cuda=self.args.cuda)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.dictionary = self.src_dict
