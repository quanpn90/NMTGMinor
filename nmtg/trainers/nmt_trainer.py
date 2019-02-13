import datetime
import logging
import os
from collections import Counter
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from nmtg.data import Dictionary, data_utils, ParallelDataset, TextLineDataset
from nmtg.data.samplers import PreGeneratedBatchSampler
from nmtg.data.text_lookup_dataset import TextLookupDataset
from nmtg.meters import StopwatchMeter, AverageMeter
from nmtg.models.nmt_model import NMTModel
from nmtg.modules.loss import NMTLoss
from nmtg.sequence_generator import SequenceGenerator
from nmtg.trainers import Trainer
from nmtg.trainers.trainer import TrainData
from . import register_trainer

logger = logging.getLogger(__name__)


@register_trainer('nmt')
class NMTTrainer(Trainer):
    @classmethod
    def add_preprocess_options(cls, parser):
        super().add_preprocess_options(parser)
        parser.add_argument('-train_src', type=str, required=True,
                            help='Path to the training source file')
        parser.add_argument('-train_tgt', type=str, required=True,
                            help='Path to the training target file')

        parser.add_argument('-src_vocab', type=str,
                            help='Path to an existing source vocabulary')
        parser.add_argument('-tgt_vocab', type=str,
                            help='Path to an existing target vocabulary')
        parser.add_argument('-data_dir_out', type=str, required=True,
                            help='Output directory for auxiliary data')
        parser.add_argument('-lower', action='store_true',
                            help='Construct a lower-case vocabulary')
        parser.add_argument('-vocab_threshold', type=int,
                            help='Discard vocabulary words that occur less often than this threshold')

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

    @classmethod
    def add_general_options(cls, parser):
        super().add_general_options(parser)
        parser.add_argument('-beam_size', type=int, default=5, help='Beam size')
        parser.add_argument('-alpha', type=float, default=0.6,
                            help='Length Penalty coefficient')
        parser.add_argument('-beta', type=float, default=0.0,
                            help='Coverage penalty coefficient')
        parser.add_argument('-normalize', action='store_true',
                            help='To normalize the scores based on output length')
        parser.add_argument('-n_best', type=int, default=1,
                            help='Will output the n_best decoded sentences')
        parser.add_argument('-label_smoothing', type=float, default=0.0,
                            help='Label smoothing value for loss functions.')
        parser.add_argument('-print_translations', action='store_true',
                            help='Output finished translations as they are generated')

        # Currently used, but pointless
        parser.add_argument('-diverse_beam_strength', type=float, default=0.5,
                            help='Diverse beam strength in decoding')

    @classmethod
    def add_training_options(cls, parser):
        super().add_training_options(parser)
        NMTModel.add_options(parser)
        parser.add_argument('-train_src', type=str, required=True,
                            help='Path to the training source file')
        parser.add_argument('-train_tgt', type=str, required=True,
                            help='Path to the training target file')
        parser.add_argument('-data_dir', type=str, required=True,
                            help='Path to an auxiliary data')
        parser.add_argument('-load_into_memory', action='store_true',
                            help='Load the dataset into memory')
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
        parser.add_argument('-src_seq_length', type=int, default=64,
                            help='Discard examples with a source sequence length above this value')
        parser.add_argument('-src_seq_length_trunc', type=int, default=0,
                            help='Truncate source sequences to this length. 0 (default) to disable')
        parser.add_argument('-tgt_seq_length', type=int, default=64,
                            help='Discard examples with a target sequence length above this value')
        parser.add_argument('-tgt_seq_length_trunc', type=int, default=0,
                            help='Truncate target sequences to this length. 0 (default) to disable')

    @classmethod
    def add_eval_options(cls, parser):
        super().add_eval_options(parser)

    @staticmethod
    def preprocess(args):
        split_words = args.input_type == 'word'
        
        # since input and output dir are the same, this is no longer needed
        os.makedirs(args.data_dir_out, exist_ok=True)

        src_offsets, src_lengths, src_counter = [0], [], Counter()
        tgt_offsets, tgt_lengths, tgt_counter = [0], [], Counter()

        with open(args.train_src) as src, open(args.train_tgt) as tgt,\
                tqdm(unit='lines', disable=args.no_progress) as pbar:
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

        out_offsets_src = os.path.join(args.data_dir_out, 'train.src.idx.npy')
        out_lengths_src = os.path.join(args.data_dir_out, 'train.src.len.npy')
        np.save(out_offsets_src, src_offsets)
        np.save(out_lengths_src, src_lengths)
        if args.src_vocab is not None:
            src_dictionary = Dictionary.load(args.src_vocab)
        else:
            src_dictionary = Dictionary()
            for word, count in src_counter.items():
                src_dictionary.add_symbol(word, count)

        out_offsets_tgt = os.path.join(args.data_dir_out, 'train.tgt.idx.npy')
        out_lengths_tgt = os.path.join(args.data_dir_out, 'train.tgt.len.npy')
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
            src_dictionary.finalize(nwords=args.src_vocab_size,
                                    threshold=args.vocab_threshold or -1)
            src_dictionary.save(os.path.join(args.data_dir_out, 'dict'))
        else:
            src_dictionary.finalize(nwords=args.src_vocab_size,
                                    threshold=args.vocab_threshold or -1)
            tgt_dictionary.finalize(nwords=args.tgt_vocab_size,
                                    threshold=args.vocab_threshold or -1)
            src_dictionary.save(os.path.join(args.data_dir_out, 'src.dict'))
            tgt_dictionary.save(os.path.join(args.data_dir_out, 'tgt.dict'))

    def __init__(self, args):
        super().__init__(args)

        if hasattr(args, 'data_dir'):
            logger.info('Loading vocabularies from {}'.format(args.data_dir))
            if args.join_vocab:
                self.src_dict = Dictionary.load(os.path.join(args.data_dir, 'dict'))
                self.tgt_dict = self.src_dict
            else:
                self.src_dict = Dictionary.load(os.path.join(args.data_dir, 'src.dict'))
                self.tgt_dict = Dictionary.load(os.path.join(args.data_dir, 'tgt.dict'))
            self.loss = self._build_loss()
            logger.debug('Source vocabulary size: {}'.format(len(self.src_dict)))
            logger.debug('Target vocabulary size: {}'.format(len(self.tgt_dict)))
        else:
            self.src_dict = None
            self.tgt_dict = None

    def online_translate(self, model_or_ensemble, in_stream):
        models = model_or_ensemble
        if not isinstance(models, Sequence):
            models = [model_or_ensemble]

        for model in models:
            model.eval()

        split_words = self.args.input_type == 'words'

        generator = SequenceGenerator(models, self.tgt_dict, models[0].batch_first,
                                      self.args.beam_size, maxlen_b=20, normalize_scores=self.args.normalize,
                                      len_penalty=self.args.alpha, unk_penalty=self.args.beta,
                                      diverse_beam_strength=self.args.diverse_beam_strength)

        join_str = ' ' if self.args.input_type == 'word' else ''

        for line in in_stream:
            line = line.rstrip()
            if self.args.lower:
                line = line.lower()
            if split_words:
                line = line.split(' ')

            src_indices = self.src_dict.to_indices(line, bos=False, eos=False)
            encoder_inputs = src_indices.unsqueeze(0 if self.batch_first else 1)
            source_lengths = torch.tensor([len(line)])
            encoder_mask = encoder_inputs.ne(self.src_dict.pad())

            if self.args.cuda:
                encoder_inputs = encoder_inputs.cuda()
                source_lengths = source_lengths.cuda()
                encoder_mask = encoder_mask.cuda()

            res = [self.tgt_dict.string(tr['tokens'], join_str=join_str)
                   for tr in generator.generate(encoder_inputs, source_lengths, encoder_mask)[0][:self.args.n_best]]

            if self.args.print_translations:
                tqdm.write(line)
                for i, hyp in enumerate(res):
                    tqdm.write("Hyp {}/{}: {}".format(i+1, len(hyp), hyp))

            if len(res) == 1:
                res = res[0]
            yield res

    def _build_loss(self):
        loss = NMTLoss(len(self.tgt_dict), self.tgt_dict.pad(), self.args.label_smoothing)
        if self.args.cuda:
            loss.cuda()
        return loss

    def _build_model(self, args):
        model = super()._build_model(args)
        logger.info('Building embeddings and softmax')
        return NMTModel.wrap_model(args, model, self.src_dict, self.tgt_dict)

    def load_data(self, model_args=None):
        logger.info('Loading training data from {}'.format(self.args.train_src))
        split_words = self.args.input_type == 'word'

        if self.args.load_into_memory:
            src_data = TextLineDataset.load_into_memory(self.args.train_src)
            tgt_data = TextLineDataset.load_into_memory(self.args.train_tgt)
        else:
            offsets_src = os.path.join(self.args.data_dir, 'train.src.idx.npy')
            offsets_tgt = os.path.join(self.args.data_dir, 'train.tgt.idx.npy')
            src_data = TextLineDataset.load_indexed(self.args.train_src, offsets_src)
            tgt_data = TextLineDataset.load_indexed(self.args.train_tgt, offsets_tgt)
        src_data = TextLookupDataset(src_data, self.src_dict, words=split_words, bos=False, eos=False,
                                     trunc_len=self.args.src_seq_length_trunc, lower=self.args.lower)
        tgt_data = TextLookupDataset(tgt_data, self.tgt_dict, words=split_words, bos=True, eos=True,
                                     trunc_len=self.args.tgt_seq_length_trunc, lower=self.args.lower)
        dataset = ParallelDataset(src_data, tgt_data)

        src_len_filename = os.path.join(self.args.data_dir, 'train.src.len.npy')
        tgt_len_filename = os.path.join(self.args.data_dir, 'train.tgt.len.npy')
        src_lengths = np.load(src_len_filename)
        tgt_lengths = np.load(tgt_len_filename)

        def filter_fn(i):
            return src_lengths[i] <= self.args.src_seq_length and tgt_lengths[i] <= self.args.tgt_seq_length

        logger.info('Generating batches')
        batches = data_utils.generate_length_based_batches_from_lengths(
            np.maximum(src_lengths, tgt_lengths), self.args.batch_size_words,
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
        return TrainData(model, dataset, sampler, lr_scheduler, optimizer, self._get_training_metrics())

    def _get_loss(self, model, batch) -> (Tensor, float):
        encoder_input = batch.get('src_indices')
        decoder_input = batch.get('tgt_input')
        targets = batch.get('tgt_output')

        if not model.batch_first:
            encoder_input = encoder_input.transpose(0, 1).contiguous()
            decoder_input = decoder_input.transpose(0, 1).contiguous()
            targets = targets.transpose(0, 1).contiguous()

        decoder_mask = decoder_input.ne(self.tgt_dict.pad())
        logits = model(encoder_input, decoder_input, decoder_mask=decoder_mask, optimized_decoding=True)
        targets = targets.masked_select(decoder_mask)

        lprobs = model.get_normalized_probs(logits, log_probs=True)
        return self.loss(lprobs, targets)

    def _get_batch_weight(self, batch):
        return batch['tgt_size']

    def _get_training_metrics(self):
        meters = super()._get_training_metrics()
        meters['srctok'] = AverageMeter()
        meters['tgttok'] = AverageMeter()
        return meters

    def _update_training_metrics(self, train_data, batch):
        meters = train_data.meters
        batch_time = meters['fwbw_wall'].val
        src_tokens = batch['src_size']
        tgt_tokens = batch['tgt_size']

        meters['srctok'].update(src_tokens, batch_time)
        meters['tgttok'].update(tgt_tokens, batch_time)

        return ['{:5.0f}|{:5.0f} tok/s'.format(meters['srctok'].avg, meters['tgttok'].avg)]

    def _reset_training_metrics(self, train_data):
        meters = train_data.meters
        meters['srctok'].reset()
        meters['tgttok'].reset()
        meters['srcpad'].reset()
        meters['tgtpad'].reset()
        meters['eff'].reset()
        super()._reset_training_metrics(train_data)

    def solve(self, model_or_ensemble, task):
        models = model_or_ensemble
        if not isinstance(models, Sequence):
            models = [model_or_ensemble]

        for model in models:
            model.eval()

        generator = SequenceGenerator(models, self.tgt_dict, models[0].batch_first,
                                      self.args.beam_size, maxlen_b=20, normalize_scores=self.args.normalize,
                                      len_penalty=self.args.alpha, unk_penalty=self.args.beta,
                                      diverse_beam_strength=self.args.diverse_beam_strength)

        iterator = self._get_eval_iterator(task, self.args.batch_size)

        join_str = ' ' if self.args.input_type == 'word' else ''

        results = []
        for batch in tqdm(iterator, postfix='inference', disable=self.args.no_progress):
            encoder_inputs = batch['src_indices']
            if not generator.batch_first:
                encoder_inputs = encoder_inputs.transpose(0, 1)
            source_lengths = batch['src_lengths']
            encoder_mask = encoder_inputs.ne(self.src_dict.pad())

            res = [self.tgt_dict.string(tr['tokens'], join_str=join_str)
                   for beams in generator.generate(encoder_inputs, source_lengths, encoder_mask)
                   for tr in beams[:self.args.n_best]]

            if self.args.print_translations:
                for i in range(len(batch['src_indices'])):
                    reference = batch['src_indices'][i][:batch['src_lengths'][i]]
                    reference = self.src_dict.string(reference, join_str=join_str,
                                                     bpe_symbol=self.args.bpe_symbol)
                    tqdm.write("Ref {}: {}".format(len(results) + i, reference))
                    for j in range(self.args.n_best):
                        translation = res[i * self.args.n_best + j]
                        tqdm.write("Hyp {}.{}: {}".format(len(results) + i, j+1,
                                   translation.replace(self.args.bpe_symbol, '')))

            results.extend(res)

        return results

    def _get_eval_iterator(self, task, batch_size):
        split_words = self.args.input_type == 'word'
        src_data = TextLookupDataset(task.src_dataset, self.src_dict, split_words, bos=False, eos=False,
                                     lower=self.args.lower)
        tgt_data = None
        if task.tgt_dataset is not None:
            tgt_data = TextLookupDataset(task.tgt_dataset, self.tgt_dict, split_words,
                                         lower=self.args.lower)
        dataset = ParallelDataset(src_data, tgt_data)
        return dataset.get_iterator(batch_size=batch_size,
                                    num_workers=self.args.data_loader_threads,
                                    cuda=self.args.cuda)

    def state_dict(self):
        res = super().state_dict()
        if self.args.join_vocab:
            res['dict'] = self.src_dict.state_dict()
        else:
            res['src_dict'] = self.src_dict.state_dict()
            res['tgt_dict'] = self.tgt_dict.state_dict()
        return res

    def load_args(self, args):
        self.args.join_vocab = args.join_vocab
        self.args.input_type = args.input_type

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
        self.loss = self._build_loss()
