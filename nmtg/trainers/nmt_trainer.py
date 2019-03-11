import itertools
import logging
import math
import os
from typing import Sequence

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from nmtg.data import Dictionary, data_utils, ParallelDataset
from nmtg.data.noisy_text import NoisyTextDataset
from nmtg.data.samplers import PreGeneratedBatchSampler
from nmtg.data.text_lookup_dataset import TextLookupDataset
from nmtg.meters import AverageMeter
from nmtg.models import build_model
from nmtg.models.encoder_decoder import EncoderDecoderModel
from nmtg.models.nmt_model import NMTEncoder, NMTDecoder
from nmtg.modules.linear import XavierLinear
from nmtg.modules.loss import NMTLoss
from nmtg.sequence_generator import SequenceGenerator
from nmtg.tasks.translation_task import TranslationTask
from nmtg.trainers import Trainer
from . import register_trainer

logger = logging.getLogger(__name__)


@register_trainer('nmt')
class NMTTrainer(Trainer):
    @classmethod
    def add_inference_options(cls, parser, argv=None):
        super().add_inference_options(parser, argv)
        parser.add_argument('-input_type', default='word', choices=['word', 'char'],
                            help='Type of dictionary to create.')
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
        parser.add_argument('-return_scores', action='store_true',
                            help='Return scores in the online translation')
        parser.add_argument('-src_seq_length_trunc', type=int, default=0,
                            help='Truncate source sequences to this length. 0 (default) to disable')
        parser.add_argument('-tgt_seq_length_trunc', type=int, default=0,
                            help='Truncate target sequences to this length. 0 (default) to disable')

        parser.add_argument('-eval_noise', action='store_true',
                            help='Also apply noise when evaluating')
        parser.add_argument('-word_shuffle', type=int, default=3,
                            help='Maximum number of positions a word can move (0 to disable)')
        parser.add_argument('-word_blank', type=float, default=0.2,
                            help='Probability to replace a word with the unknown word (0 to disable)')
        parser.add_argument('-noise_word_dropout', type=float, default=0.1,
                            help='Probability to remove a word (0 to disable)')

    @classmethod
    def add_training_options(cls, parser, argv=None):
        super().add_training_options(parser, argv)
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
        parser.add_argument('-batch_size_words', type=int, default=2048,
                            help='Maximum number of words in a batch')
        parser.add_argument('-batch_size_sents', type=int, default=128,
                            help='Maximum number of sentences in a batch')
        parser.add_argument('-batch_size_multiplier', type=int, default=1,
                            help='Number of sentences in a batch must be divisible by this number')
        parser.add_argument('-batch_size_update', type=int, default=20000,
                            help='Perform a learning step after this many tokens')
        parser.add_argument('-normalize_gradient', action='store_true',
                            help='Divide gradient by the number of tokens')
        parser.add_argument('-pad_count', action='store_true',
                            help='Count padding words when batching')
        parser.add_argument('-src_seq_length', type=int, default=64,
                            help='Discard source sequences above this length')
        parser.add_argument('-tgt_seq_length', type=int, default=64,
                            help='Discard target sequences above this length')

        parser.add_argument('-translation_noise', action='store_true',
                            help='Apply noise to the source when translating')

        parser.add_argument('-tie_weights', action='store_true',
                            help='Share weights between embedding and softmax')
        parser.add_argument('-freeze_embeddings', action='store_true',
                            help='Do not train word embeddings')
        parser.add_argument('-pre_word_vecs_enc', type=str,
                            help='If a valid path is specified, then this will load '
                                 'pretrained word embeddings on the encoder side. '
                                 'See README for specific formatting instructions.')
        parser.add_argument('-pre_word_vecs_dec', type=str,
                            help='If a valid path is specified, then this will load '
                                 'pretrained word embeddings on the decoder side. '
                                 'See README for specific formatting instructions.')
        parser.add_argument('-word_vec_size', type=int,
                            help='Word embedding sizes')
        parser.add_argument('-word_dropout', type=float, default=0.0,
                            help='Dropout probability; applied on embedding indices.')
        parser.add_argument('-init_embedding', default='normal', choices=['xavier', 'normal'],
                            help="How to init the embedding matrices.")
        parser.add_argument('-copy_decoder', action='store_true',
                            help='Use a decoder that will copy tokens from the input when it thinks it appropriate')
        parser.add_argument('-freeze_model', action='store_true',
                            help='Only used when upgrading an NMT Model without copy decoder.'
                                 'Freeze the model and only learn the copy decoder parameters')
        parser.add_argument('-extra_attention', action='store_true',
                            help='Add an extra attention layer at the end of the model to predict alignment for '
                                 'the copy decoder. For models like transformer, that have no clear attention '
                                 'alignment.')

    def online_translate(self, model_or_ensemble, in_stream):
        # TODO: deprecated
        models = model_or_ensemble
        if not isinstance(models, Sequence):
            models = [model_or_ensemble]

        for model in models:
            model.eval()

        split_words = self.args.input_type == 'word'

        generator = SequenceGenerator(models, self.tgt_dict, models[0].batch_first,
                                      self.args.beam_size, maxlen_b=20, normalize_scores=self.args.normalize,
                                      len_penalty=self.args.alpha, unk_penalty=self.args.beta)

        join_str = ' ' if self.args.input_type == 'word' else ''

        for line in in_stream:
            line = line.rstrip()
            if self.args.lower:
                line = line.lower()
            if split_words:
                line = line.split(' ')

            src_indices = self.src_dict.to_indices(line, bos=False, eos=False)
            encoder_inputs = src_indices.unsqueeze(0 if models[0].batch_first else 1)
            source_lengths = torch.tensor([len(line)])
            encoder_mask = encoder_inputs.ne(self.src_dict.pad())

            if self.args.cuda:
                encoder_inputs = encoder_inputs.cuda()
                source_lengths = source_lengths.cuda()
                encoder_mask = encoder_mask.cuda()

            res = []
            scores = []
            positional_scores = []
            for tr in generator.generate(encoder_inputs, source_lengths, encoder_mask)[0][:self.args.n_best]:
                res.append(self.tgt_dict.string(tr['tokens'], join_str=join_str))
                scores.append(tr['score'])
                positional_scores.append(tr['positional_scores'])

            if self.args.print_translations:
                tqdm.write(line)
                for i, hyp in enumerate(res):
                    tqdm.write("Hyp {}/{}: {}".format(i + 1, len(hyp), hyp))

            if len(res) == 1:
                res = res[0]
                scores = scores[0]
                positional_scores = positional_scores[0]

            if self.args.return_scores:
                yield res, scores, positional_scores.tolist()
            else:
                yield res

    def _build_data(self):
        super()._build_data()

        if self.args.join_vocab:
            self.src_dict = Dictionary.load(os.path.join(self.args.data_dir, 'dict'))
            self.tgt_dict = self.src_dict
        else:
            self.src_dict = Dictionary.load(os.path.join(self.args.data_dir, 'src.dict'))
            self.tgt_dict = Dictionary.load(os.path.join(self.args.data_dir, 'tgt.dict'))
        logger.info('Vocabulary size: {:,d}|{:,d}'.format(len(self.src_dict), len(self.tgt_dict)))
        self._build_loss()

    def _load_data(self, checkpoint):
        super()._load_data(checkpoint)
        args = checkpoint['args']

        if args.join_vocab:
            self.src_dict = Dictionary()
            self.src_dict.load_state_dict(checkpoint['dict'])
            self.tgt_dict = self.src_dict
        else:
            self.src_dict = Dictionary()
            self.src_dict.load_state_dict(checkpoint['src_dict'])
            self.tgt_dict = Dictionary()
            self.tgt_dict.load_state_dict(checkpoint['tgt_dict'])
        self._build_loss()

    def _save_data(self, checkpoint):
        super()._save_data(checkpoint)
        if self.args.join_vocab:
            checkpoint['dict'] = self.src_dict.state_dict()
        else:
            checkpoint['src_dict'] = self.src_dict.state_dict()
            checkpoint['tgt_dict'] = self.tgt_dict.state_dict()

    def _build_loss(self):
        logger.info('Building loss')
        loss = NMTLoss(len(self.tgt_dict), self.tgt_dict.pad(), self.args.label_smoothing)
        if self.args.cuda:
            loss.cuda()
        self.loss = loss

    def _build_model(self, model_args):
        logger.info('Building {} model'.format(model_args.model))
        model = build_model(model_args.model, model_args)

        embedding_size = model_args.word_vec_size or getattr(model_args, 'model_size', None)
        if embedding_size is None:
            raise ValueError('Could not infer embedding size')

        if model_args.copy_decoder and not model_args.join_vocab:
            raise NotImplementedError('In order to use the copy decoder, the source and target language must '
                                      'use the same vocabulary')

        if model_args.join_vocab and model_args.pre_word_vecs_dec:
            raise ValueError('Cannot join vocabularies when loading pre-trained target embeddings')

        dummy_input = torch.zeros(1, 1, embedding_size)
        dummy_output, _ = model(dummy_input, dummy_input)
        output_size = dummy_output.size(-1)

        src_embedding = self._get_embedding(model_args, self.src_dict, embedding_size, model_args.pre_word_vecs_enc)

        if model_args.join_vocab:
            tgt_embedding = src_embedding
        else:
            tgt_embedding = self._get_embedding(model_args, self.tgt_dict, embedding_size, model_args.pre_word_vecs_dec)

        tgt_linear = XavierLinear(output_size, len(self.tgt_dict))

        if model_args.tie_weights:
            tgt_linear.weight = tgt_embedding.weight

        encoder = NMTEncoder(model.encoder, src_embedding, model_args.word_dropout)

        if model_args.copy_decoder:
            masked_layers = getattr(model_args, 'masked_layers', False)
            attention_dropout = getattr(model_args, 'attn_dropout', 0.0)
            decoder = NMTDecoder(model.decoder, tgt_embedding, model_args.word_dropout, tgt_linear,
                                 copy_decoder=True,
                                 batch_first=model_args.batch_first,
                                 extra_attention=model_args.extra_attention,
                                 masked_layers=masked_layers,
                                 attention_dropout=attention_dropout)
        else:
            decoder = NMTDecoder(model.decoder, tgt_embedding, model_args.word_dropout, tgt_linear)

        if model_args.freeze_model:
            logger.info('Freezing model parameters')
            for param in itertools.chain(encoder.parameters(), decoder.decoder.parameters(),
                                         tgt_embedding.parameters(),
                                         tgt_linear.parameters()):
                param.requires_grad_(False)

        self.model = EncoderDecoderModel(encoder, decoder)
        self.model.batch_first = model_args.batch_first

    @staticmethod
    def _get_embedding(args, dictionary, embedding_size, path):
        emb = nn.Embedding(len(dictionary), embedding_size, padding_idx=dictionary.pad())
        if path is not None:
            embed_dict = data_utils.parse_embedding(path)
            data_utils.load_embedding(embed_dict, dictionary, emb)
        elif args.init_embedding == 'xavier':
            nn.init.xavier_uniform_(emb.weight)
        elif args.init_embedding == 'normal':
            nn.init.normal_(emb.weight, mean=0, std=embedding_size ** -0.5)
        else:
            raise ValueError('Unknown initialization {}'.format(args.init_embedding))

        if args.freeze_embeddings:
            emb.weight.requires_grad_(False)

        return emb

    def _get_text_lookup_dataset(self, task, text_dataset, src=True):
        split_words = self.args.input_type == 'word'
        dataset = TextLookupDataset(text_dataset,
                                    self.src_dict if src else self.tgt_dict,
                                    words=split_words,
                                    lower=task.lower,
                                    bos=not src, eos=not src,
                                    trunc_len=self.args.src_seq_length_trunc if src else self.args.tgt_seq_length_trunc)
        if text_dataset.in_memory:
            if split_words:
                lengths = np.array([len(sample.split()) for sample in text_dataset])
            else:
                lengths = np.array([len(sample) for sample in text_dataset])
        else:
            basename = os.path.basename(text_dataset.filename)
            lengths = np.load(os.path.join(self.args.data_dir, basename + '.len.npy'))
        dataset.lengths = lengths
        return dataset

    def _get_train_dataset(self):
        logger.info('Loading training data')
        split_words = self.args.input_type == 'word'

        src_data, src_lengths = TextLookupDataset.load(self.args.train_src, self.src_dict, self.args.data_dir,
                                                       self.args.load_into_memory, split_words,
                                                       bos=False, eos=False, trunc_len=self.args.src_seq_length_trunc,
                                                       lower=self.args.lower)

        if self.args.translation_noise:
            src_data = NoisyTextDataset(src_data, self.args.word_shuffle, self.args.noise_word_dropout,
                                        self.args.word_blank, self.args.bpe_symbol)

        tgt_data, tgt_lengths = TextLookupDataset.load(self.args.train_tgt, self.tgt_dict, self.args.data_dir,
                                                       self.args.load_into_memory, split_words,
                                                       bos=True, eos=True, trunc_len=self.args.tgt_seq_length_trunc,
                                                       lower=self.args.lower)
        src_data.lengths = src_lengths
        tgt_data.lengths = tgt_lengths
        dataset = ParallelDataset(src_data, tgt_data)
        logger.info('Number of training sentences: {:,d}'.format(len(dataset)))
        return dataset

    def _get_eval_dataset(self, task: TranslationTask):
        split_words = self.args.input_type == 'word'
        src_dataset = TextLookupDataset(task.src_dataset,
                                        self.src_dict,
                                        words=split_words,
                                        lower=task.lower,
                                        bos=False, eos=False,
                                        trunc_len=self.args.src_seq_length_trunc)

        if self.args.eval_noise:
            src_dataset = NoisyTextDataset(src_dataset, self.args.word_shuffle, self.args.noise_word_dropout,
                                           self.args.word_blank, self.args.bpe_symbol)

        if task.tgt_dataset is not None:
            tgt_dataset = TextLookupDataset(task.tgt_dataset,
                                            self.tgt_dict,
                                            words=split_words,
                                            lower=task.lower,
                                            bos=True, eos=True,
                                            trunc_len=self.args.tgt_seq_length_trunc)
        else:
            tgt_dataset = None
        dataset = ParallelDataset(src_dataset, tgt_dataset)
        return dataset

    def _get_train_sampler(self, dataset: ParallelDataset):
        src_lengths = dataset.src_data.lengths
        tgt_lengths = dataset.tgt_data.lengths

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
        logger.info('Number of training batches: {:,d}'.format(len(batches)))

        filtered = len(src_lengths) - sum(len(batch) for batch in batches)
        logger.info('Filtered {:,d}/{:,d} training examples for length'.format(filtered, len(src_lengths)))
        sampler = PreGeneratedBatchSampler(batches, self.args.curriculum == 0)
        return sampler

    def _get_training_metrics(self):
        metrics = super()._get_training_metrics()
        metrics['nll'] = AverageMeter()
        metrics['src_tps'] = AverageMeter()
        metrics['tgt_tps'] = AverageMeter()
        metrics['total_words'] = AverageMeter()
        return metrics

    def _reset_training_metrics(self, metrics):
        super()._reset_training_metrics(metrics)
        metrics['src_tps'].reset()
        metrics['tgt_tps'].reset()
        metrics['nll'].reset()

    def _format_train_metrics(self, metrics):
        formatted = super()._format_train_metrics(metrics)
        perplexity = math.exp(metrics['nll'].avg)
        formatted.insert(1, 'ppl {:6.2f}'.format(perplexity))

        srctok = metrics['src_tps'].sum / metrics['fwbw_wall'].sum
        tgttok = metrics['tgt_tps'].sum / metrics['fwbw_wall'].sum
        formatted.append('{:5.0f}|{:5.0f} tok/s'.format(srctok, tgttok))
        return formatted

    def _forward(self, batch, model, loss, training=True):
        encoder_input = batch.get('src_indices')
        decoder_input = batch.get('tgt_input')
        targets = batch.get('tgt_output')

        if not self.args.batch_first:
            encoder_input = encoder_input.transpose(0, 1).contiguous()
            decoder_input = decoder_input.transpose(0, 1).contiguous()
            targets = targets.transpose(0, 1).contiguous()

        encoder_mask = encoder_input.ne(self.src_dict.pad())
        decoder_mask = decoder_input.ne(self.tgt_dict.pad())
        outputs, attn_out = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
        lprobs = model.get_normalized_probs(outputs, attn_out, encoder_input, encoder_mask, decoder_mask, True)
        if training:
            targets = targets.masked_select(decoder_mask)
        return loss(lprobs, targets)

    def _forward_backward_pass(self, batch, metrics):
        src_size = batch.get('src_size')
        tgt_size = batch.get('tgt_size')
        loss, display_loss = self._forward(batch, self.model, self.loss)
        self.optimizer.backward(loss)
        metrics['nll'].update(display_loss, tgt_size)
        metrics['src_tps'].update(src_size)
        metrics['tgt_tps'].update(tgt_size)
        metrics['total_words'].update(tgt_size)

    def _do_training_step(self, metrics, batch):
        return metrics['total_words'].sum >= self.args.batch_size_update

    def _learning_step(self, metrics):
        if self.args.normalize_gradient:
            self.optimizer.multiply_grads(1 / metrics['total_words'].sum)
        super()._learning_step(metrics)
        metrics['total_words'].reset()

    def _get_eval_metrics(self):
        metrics = super()._get_eval_metrics()
        metrics['nll'] = AverageMeter()
        return metrics

    def _format_eval_metrics(self, metrics):
        formatted = super()._format_eval_metrics(metrics)
        formatted.append('Validation perplexity: {:.2f}'.format(math.exp(metrics['nll'].avg)))
        return formatted

    def _eval_pass(self, batch, metrics):
        tgt_size = batch.get('tgt_size')
        _, display_loss = self._forward(batch, self.model, self.loss, False)
        metrics['nll'].update(display_loss, tgt_size)

    def _get_sequence_generator(self):
        return SequenceGenerator([self.model], self.tgt_dict, self.model.batch_first,
                                 self.args.beam_size, maxlen_b=20, normalize_scores=self.args.normalize,
                                 len_penalty=self.args.alpha, unk_penalty=self.args.beta)

    def solve(self, test_task):
        self.model.eval()

        generator = self._get_sequence_generator()

        test_dataset = self._get_eval_dataset(test_task)
        test_sampler = self._get_eval_sampler(test_dataset)
        test_iterator = self._get_iterator(test_dataset, test_sampler)

        join_str = ' ' if self.args.input_type == 'word' else ''

        results = []
        for batch in tqdm(test_iterator, desc='inference', disable=self.args.no_progress):
            encoder_input = batch.get('src_indices')
            source_lengths = batch.get('src_lengths')

            if not generator.batch_first:
                encoder_input = encoder_input.transpose(0, 1).contiguous()

            encoder_mask = encoder_input.ne(self.src_dict.pad())

            res = [self.tgt_dict.string(tr['tokens'], join_str=join_str)
                   for beams in generator.generate(encoder_input, source_lengths, encoder_mask)
                   for tr in beams[:self.args.n_best]]

            if self.args.print_translations:
                for i in range(len(batch['src_indices'])):
                    reference = batch['src_indices'][i][:batch['src_lengths'][i]]
                    reference = self.src_dict.string(reference, join_str=join_str,
                                                     bpe_symbol=self.args.bpe_symbol)
                    tqdm.write("Src {}: {}".format(len(results) + i, reference))
                    for j in range(self.args.n_best):
                        translation = res[i * self.args.n_best + j]
                        tqdm.write("Hyp {}.{}: {}".format(len(results) + i, j + 1,
                                                          translation.replace(self.args.bpe_symbol, '')))
                    tqdm.write("")

            results.extend(res)

        return results

    @classmethod
    def upgrade_checkpoint(cls, checkpoint):
        super().upgrade_checkpoint(checkpoint)
        args = checkpoint['args']
        if 'freeze_model' not in args:
            args.freeze_model = False
            args.copy_decoder = False
            args.extra_attention = False
        if 'eval_noise' not in args:
            args.translation_noise = getattr(args, 'translation_noise', False)
            args.eval_noise = False
