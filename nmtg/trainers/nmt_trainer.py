import datetime
import logging
import os
from collections import Counter
from typing import Sequence

import numpy as np
from torch import Tensor
from tqdm import tqdm

from nmtg.data import Dictionary, data_utils, ParallelDataset, TextLineDataset
from nmtg.data.samplers import PreGeneratedBatchSampler
from nmtg.data.text_lookup_dataset import TextLookupDataset
from nmtg.models.nmt_model import NMTModel
from nmtg.modules.loss import NMTLoss
from nmtg.sequence_generator import SequenceGenerator
from nmtg.trainers import Trainer
from nmtg.trainers.trainer import TrainData
from . import register_trainer

logger = logging.getLogger(__name__)


@register_trainer('nmt')
class NMTTrainer(Trainer):
    @staticmethod
    def add_preprocess_options(parser):
        super().add_preprocess_options(parser)
        parser.add_argument('-train_src', type=str, required=True,
                            help='Path to the training source data')
        parser.add_argument('-train_tgt', type=str, required=True,
                            help='Path to the training target data')
        parser.add_argument('-save_data', type=str, required=True,
                            help='Output folder for the prepared data')

        parser.add_argument('-src_vocab', type=str,
                            help='Path to an existing source vocabulary')
        parser.add_argument('-tgt_vocab', type=str,
                            help='Path to an existing target vocabulary')

        # parser.add_argument('-remove_duplicate', action='store_true',
        #                     help='Remove examples where source and target are the same')
        parser.add_argument('-join_vocab', action='store_true',
                            help='Share dictionary for source and target')
        parser.add_argument('-src_vocab_size', type=int, default=50000,
                            help='Size of the source vocabulary')
        parser.add_argument('-tgt_vocab_size', type=int, default=50000,
                            help='Size of the target vocabulary')
        parser.add_argument('-input_type', default='word', choices=['word', 'char'],
                            help='Type of dictionary to create.')
        parser.add_argument('-report_every', type=int, default=100000,
                            help='Report status every this many sentences')

    @staticmethod
    def add_general_options(parser):
        super().add_general_options(parser)

        # Currently used, but pointless
        parser.add_argument('-diverse_beam_strength', type=float, default=0.5,
                            help='Diverse beam strength in decoding')
        parser.add_argument('-beam_size', type=int, default=5, help='Beam size')
        parser.add_argument('-alpha', type=float, default=0.6,
                            help='Length Penalty coefficient')
        parser.add_argument('-beta', type=float, default=0.0,
                            help='Coverage penalty coefficient')
        parser.add_argument('-normalize', action='store_true',
                            help='To normalize the scores based on output length')
        parser.add_argument('-n_best', type=int, default=1,
                            help='Will output the n_best decoded sentences')

    @staticmethod
    def add_training_options(parser):
        super().add_training_options(parser)
        parser.add_argument('-join_vocab', action='store_true',
                            help='Share dictionary for source and target')
        parser.add_argument('-input_type', default='word', choices=['word', 'char'],
                            help='Type of dictionary to create.')
        parser.add_argument('-batch_size_words', type=int, default=2048,
                            help='Maximum number of words in a batch')
        parser.add_argument('-batch_size_sents', type=int, default=128,
                            help='Maximum number of sentences in a batch')
        parser.add_argument('-batch_size_multiplier', type=int, default=1,
                            help='Number of sentences in a batch must be divisible by this number')
        parser.add_argument('-pad_count', action='store_true',
                            help='Count padding words when batching')
        parser.add_argument('-label_smoothing', type=float, default=0.0,
                            help='Label smoothing value for loss functions.')
        parser.add_argument('-src_seq_length', type=int, default=64,
                            help='Discard examples with a source sequence length above this value')
        parser.add_argument('-src_seq_length_trunc', type=int, default=0,
                            help='Truncate source sequences to this length. 0 (default) to disable')
        parser.add_argument('-tgt_seq_length', type=int, default=64,
                            help='Discard examples with a target sequence length above this value')
        parser.add_argument('-tgt_seq_length_trunc', type=int, default=0,
                            help='Truncate target sequences to this length. 0 (default) to disable')

    @staticmethod
    def preprocess(args):
        split_words = args.input_type == 'word'

        os.makedirs(args.save_data, exist_ok=True)

        src_offsets, src_lengths, src_counter = [0], [], Counter()
        tgt_offsets, tgt_lengths, tgt_counter = [0], [], Counter()

        with open(args.train_src) as src, open(args.train_tgt) as tgt, tqdm(unit='lines') as pbar:
            src_line = None
            tgt_line = None
            i = 0
            while src_line != '' and tgt_line != '':
                src_line = src.readline()
                tgt_line = tgt.readline()

                proc_src_line = src_line.rstrip()
                proc_tgt_line = tgt_line.rstrip()

                src_offsets.append(src.tell())
                tgt_offsets.append(tgt.tell())

                if args.lower:
                    proc_src_line = proc_src_line.lower()
                    proc_tgt_line = proc_tgt_line.lower()
                if split_words:
                    proc_src_line = proc_src_line.split()
                    proc_tgt_line = proc_tgt_line.split()

                src_lengths.append(len(proc_src_line))
                tgt_lengths.append(len(proc_tgt_line))

                src_counter.update(proc_src_line)
                tgt_counter.update(proc_tgt_line)
                i += 1

                if i % args.report_every == 0:
                    logger.info('{} lines processed'.format(i))
                pbar.update()

            if src_line != '' or tgt_line != '':
                logger.warning('Source and target file were not the same length!')

        out_offsets_src = os.path.join(args.save_data, 'train.src.idx')
        out_lengths_src = os.path.join(args.save_data, 'train.src.len')
        np.save(out_offsets_src, src_offsets)
        np.save(out_lengths_src, src_lengths)
        if args.src_vocab is not None:
            src_dictionary = Dictionary.load(args.src_vocab)
        else:
            src_dictionary = Dictionary()
            for word, count in src_counter.items():
                src_dictionary.add_symbol(word, count)

        out_offsets_tgt = os.path.join(args.save_data, 'train.tgt.idx')
        out_lengths_tgt = os.path.join(args.save_data, 'train.tgt.len')
        np.save(out_offsets_tgt, tgt_offsets)
        np.save(out_lengths_tgt, tgt_lengths)
        if args.tgt_vocab is not None:
            tgt_dictionary = Dictionary.load(args.tgt_vocab)
        else:
            tgt_dictionary = Dictionary()
            for word, count in tgt_counter.items():
                tgt_dictionary.add_symbol(word, count)

        if args.join_vocab:
            # If we explicitly load a target dictionary to merge
            # or we are inferring both dictionaries
            if args.tgt_vocab is not None or args.src_vocab is None:
                src_dictionary.update(tgt_dictionary)
            src_dictionary.finalize(nwords=args.src_vocab_size)
            src_dictionary.save(os.path.join(args.save_data, 'dict'))
        else:
            src_dictionary.finalize(nwords=args.src_vocab_size)
            tgt_dictionary.finalize(nwords=args.tgt_vocab_size)
            src_dictionary.save(os.path.join(args.save_data, 'src.dict'))
            tgt_dictionary.save(os.path.join(args.save_data, 'tgt.dict'))

    def __init__(self, args):
        super().__init__(args)

        if hasattr(args, 'data_dir'):
            if args.join_vocab:
                self.src_dict = Dictionary.load(os.path.join(args.data_dir, 'dict'))
                self.tgt_dict = self.src_dict
            else:
                self.src_dict = Dictionary.load(os.path.join(args.data_dir, 'dict.src'))
                self.tgt_dict = Dictionary.load(os.path.join(args.data_dir, 'dict.tgt'))
        else:
            self.src_dict = None
            self.tgt_dict = None

        self.loss = NMTLoss(len(self.tgt_dict), self.tgt_dict.pad(), args.label_smoothing)

    def _build_model(self, args):
        model = super()._build_model(args)
        return NMTModel.wrap_model(args, model, self.src_dict, self.tgt_dict)

    def load_data(self, model_args=None):
        split_words = self.args.input_type == 'word'

        offsets_src = os.path.join(self.args.data_dir, 'train.src.idx')
        offsets_tgt = os.path.join(self.args.data_dir, 'train.tgt.idx')
        src_data = TextLineDataset.load_indexed(self.args.train_src, offsets_src)
        src_data = TextLookupDataset(src_data, self.src_dict, words=split_words, bos=False, eos=False,
                                     trunc_len=self.args.src_seq_length_trunc, lower=self.args.lower)
        tgt_data = TextLineDataset.load_indexed(self.args.train_tgt, offsets_tgt)
        tgt_data = TextLookupDataset(tgt_data, self.tgt_dict, words=split_words, bos=True, eos=True,
                                     trunc_len=self.args.tgt_seq_length_trunc, lower=self.args.lower)
        dataset = ParallelDataset(src_data, tgt_data)

        src_len_filename = os.path.join(self.args.data_dir, 'train.src.len')
        tgt_len_filename = os.path.join(self.args.data_dir, 'train.tgt.len')
        src_lengths = np.load(src_len_filename)
        tgt_lengths = np.load(tgt_len_filename)

        def filter_fn(i):
            return src_lengths[i] <= self.args.src_seq_length and tgt_lengths[i] <= self.args.tgt_seq_length

        batches = data_utils.generate_length_based_batches_from_lengths(
            tgt_lengths, self.args.batch_size_words,
            self.args.batch_size_sents,
            self.args.batch_size_multiplier,
            self.args.pad_count,
            key_fn=lambda i: (tgt_lengths[i], src_lengths[i]),
            filter_fn=filter_fn)

        filtered = len(src_lengths) - sum(len(batch) for batch in batches)
        logger.info('Filtered {}/{} training examples for length'.format(filtered, len(src_lengths)))

        sampler = PreGeneratedBatchSampler(batches, self.args.curriculum == 0)

        model = self._build_model(model_args or self.args)
        lr_scheduler, optimizer = self._build_optimizer(model)
        return TrainData(model, dataset, sampler, lr_scheduler, optimizer)

    def _get_loss(self, model, batch) -> (Tensor, float):
        encoder_input = batch.get('src_tokens')
        decoder_input = batch.get('tgt_input')
        targets = batch.get('tgt_output')

        if not self.model.batch_first:
            encoder_input = encoder_input.transpose(0, 1).contiguous()
            decoder_input = decoder_input.transpose(0, 1).contiguous()
            targets = targets.transpose(0, 1).contiguous()

        logits = self.model(encoder_input, decoder_input, optimized_decoding=True)
        targets = targets.mask_select(decoder_input.ne(self.tgt_dict.pad()))

        lprobs = self.model.get_normalized_probs(logits, log_probs=True)
        return self.loss(lprobs, targets)

    def _get_training_metrics(self, batch, step_time: datetime.timedelta):
        src_size = batch.get('src_size')
        tgt_size = batch.get('tgt_size')

        return ['{:5.0f} srctok/s'.format(src_size / step_time),
                '{:5.0f} tgttok/s'.format(tgt_size / step_time)]

    def solve(self, model_or_ensemble, task):
        models = model_or_ensemble
        if not isinstance(models, Sequence):
            models = [model_or_ensemble]

        for model in models:
            model.eval()

        generator = SequenceGenerator(models, self.tgt_dict,
                                      self.args.beam_size, maxlen_b=20, normalize_scores=self.args.normalize,
                                      len_penalty=self.args.alpha, unk_penalty=self.args.beta,
                                      diverse_beam_strength=self.args.diverse_beam_strength)

        iterator = self._get_eval_iterator(task)

        bpe_symbol = self.args.bpe_symbol if self.args.input_type == 'word' else None
        join_str = ' ' if self.args.input_type == 'word' else ''

        results = [self.tgt_dict.string(result['idx'], join_str=join_str, bpe_symbol=bpe_symbol)
                   for batch in tqdm(iterator)
                   for result in generator.generate(batch['src_tokens'], batch['src_lengths'],
                                                    batch['src_tokens'].ne(self.src_dict.pad()))]

        return results

    def _get_eval_iterator(self, task):
        split_words = self.args.input_type == 'word'
        src_data = TextLookupDataset(task.src_dataset, self.src_dict, split_words, bos=False, eos=False,
                                     lower=self.args.lower)
        tgt_data = None
        if task.tgt_dataset is not None:
            tgt_data = TextLookupDataset(task.tgt_dataset, self.tgt_dict, split_words,
                                         lower=self.args.lower)
        dataset = ParallelDataset(src_data, tgt_data)
        return dataset.get_iterator(self.args.batch_size, num_workers=self.args.data_loader_threads,
                                    cuda=self.args.cuda)

    def state_dict(self):
        res = super().state_dict()
        if self.args.join_vocab:
            res['dict'] = self.src_dict.state_dict()
        else:
            res['src_dict'] = self.src_dict.state_dict()
            res['tgt_dict'] = self.tgt_dict.state_dict()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.args.join_vocab:
            self.src_dict = Dictionary()
            self.src_dict.load_state_dict(state_dict['dict'])
            self.tgt_dict = self.src_dict
        else:
            self.src_dict = Dictionary()
            self.src_dict.load_state_dict(state_dict['src_dict'])
            self.tgt_dict = Dictionary()
            self.tgt_dict.load_state_dict(state_dict['tgt_dict'])
