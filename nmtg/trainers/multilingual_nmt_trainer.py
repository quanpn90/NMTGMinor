import logging
import os
import torch

import numpy as np
from torch import nn

from nmtg.data import data_utils
from nmtg.data.dataset import ConcatDataset
from nmtg.data.dictionary import MultilingualDictionary
from nmtg.data.noisy_text import NoisyMultiParallelDataset, NoisyTextDataset
from nmtg.data.parallel_dataset import MultiParallelDataset, ParallelDataset
from nmtg.data.samplers import PreGeneratedBatchSampler
from nmtg.data.text_lookup_dataset import TextLookupDataset
from nmtg.models import build_model, Model
from nmtg.models.encoder_decoder import EncoderDecoderModel, IncrementalDecoder
from nmtg.modules.nmt import NMTEncoder, NMTDecoder
from nmtg.modules.linear import XavierLinear
from nmtg.modules.loss import NMTLoss
from nmtg.sequence_generator import SequenceGenerator
from nmtg.tasks.translation_task import TranslationTask
from nmtg.trainers import register_trainer
from nmtg.trainers.nmt_trainer import NMTTrainer


logger = logging.getLogger(__name__)


@register_trainer('multilingual')
class MultilingualNMTTrainer(NMTTrainer):
    class MultilingualNMTModel(Model):
        def __init__(self, encoders, decoders):
            super().__init__()
            self.encoders = nn.ModuleDict(encoders)
            self.decoders = nn.ModuleDict(decoders)

    class DecoderWrapper(IncrementalDecoder):
        def __init__(self, decoder: IncrementalDecoder, language):
            super().__init__(decoder.future_masking)
            self.decoder = decoder
            self.language = language

        def forward(self, decoder_inputs, encoder_outputs, decoder_mask=None, encoder_mask=None):
            return self.decoder.forward((decoder_inputs, self.language), encoder_outputs, decoder_mask, encoder_mask)

        def _step(self, decoder_inputs, encoder_outputs, incremental_state, decoder_mask=None, encoder_mask=None):
            return self.decoder._step((decoder_inputs, self.language), encoder_outputs, incremental_state,
                                      decoder_mask, encoder_mask)

    @classmethod
    def _add_inference_data_options(cls, parser, argv=None):
        parser.add_argument('-seq_length_trunc', type=int, default=0,
                            help='Truncate sequences to this length. 0 (default) to disable')

    @classmethod
    def _add_train_data_options(cls, parser, argv=None):
        parser.add_argument('-langs', nargs='+',
                            help='Language pairs (src-tgt) or languages')
        parser.add_argument('-train_data', nargs='+',
                            help='The input files.')
        parser.add_argument('-exclude_pairs', nargs='*', default=[],
                            help='Exclude some language pairs from training. Format: l1-l2')
        parser.add_argument('-balance_pairs', action='store_true',
                            help='Balance the datasets when using pairs')
        parser.add_argument('-invert_pairs', action='store_true',
                            help='When using language pairs, also consider the inverse direction')
        parser.add_argument('-join_lang_vocab', nargs='*', default=[],
                            help='Share vocabularies across languages these languages (or "all")')
        parser.add_argument('-join_src_tgt_vocab', action='store_true',
                            help='Share vocabularies for source and target side')
        parser.add_argument('-seq_length', type=int, default=64,
                            help='Discard source sequences above this length')
        parser.add_argument('-pre_word_vecs', nargs='*', default=[],
                            help='Paths to pretrained word embeddings, see vocabs for format')
        parser.add_argument('-translation_noise', nargs='*', default=[],
                            help='Apply noise to the source when translating. Specify language(s) or "all"')

    @classmethod
    def add_training_options(cls, parser, argv=None):
        super().add_training_options(parser, argv)

        parser.add_argument('-output_select', choices=['separate_decoders', 'encoder_bos', 'decoder_bos',
                                                       'decoder_every_step'],
                            required=True, help='How to select the output language')
        parser.add_argument('-separate_encoders', action='store_true',
                            help='Create one encoder per language')

    def _get_dict_keys(self):
        if self.args.join_lang_vocab == ['all']:
            if self.args.join_src_tgt_vocab:
                keys = {lang + '.' + srctgt: 'dict'
                        for lang in self.source_languages
                        for srctgt in ['src', 'tgt']}
            else:
                keys = {lang + '.src': 'src.dict' for lang in self.source_languages}
                keys.update({lang + '.tgt': 'tgt.dict' for lang in self.target_languages})
        else:
            if self.args.join_src_tgt_vocab:
                keys = {lang + '.src': lang + '.dict' for lang in self.source_languages}
                keys.update({lang + '.tgt': lang + '.dict' for lang in self.target_languages})
            else:
                keys = {lang + '.' + srctgt: lang + '.' + srctgt + '.dict'
                        for lang in self.source_languages
                        for srctgt in ['src', 'tgt']}

            if len(self.args.join_lang_vocab) > 0:
                share_lang = self.args.join_lang_vocab[0]
                for lang in self.args.join_lang_vocab[1:]:
                    if lang + '.src' in keys:
                        keys[lang + '.src'] = keys[share_lang + '.src']
                    if lang + '.tgt' in keys:
                        keys[lang + '.tgt'] = keys[share_lang + '.tgt']
        return keys

    def _build_data(self):
        super(NMTTrainer, self)._build_data()
        pairs = len(self.args.langs[0].split('-')) == 2
        self.data_mode = 'pairs' if pairs else 'all_to_all'

        logger.info('Loading vocabularies')
        if self.data_mode == 'all_to_all':
            if not self.args.join_src_tgt_vocab:
                raise ValueError('In order to use all_to_all data mode, vocabularies must be shared across'
                                 'source and target languages')
            if self.args.balance_pairs:
                raise ValueError('In all_to_all mode, all language pairs are automatically balanced.')

            if self.args.invert_pairs:
                raise ValueError('Cannot use invert_pairs and all_to_all mode')

            self.source_languages = self.target_languages = sorted(self.args.langs)
            if len(set(self.source_languages)) != len(self.source_languages):
                raise ValueError('Duplicate languages')

        elif self.data_mode == 'pairs':
            if len(self.args.exclude_pairs) != 0:
                raise ValueError('The exclude_pairs option is only for all_to_all')
            self.source_languages, self.target_languages = zip(*(lang.split('-') for lang in self.args.langs))

            if self.args.invert_pairs:
                self.source_languages = self.target_languages = sorted(
                    set(self.source_languages + self.target_languages))
            else:
                self.source_languages = sorted(set(self.source_languages))
                self.target_languages = sorted(set(self.target_languages))
        else:
            raise ValueError('Unknown data mode')

        logger.debug('{} -> {}'.format(','.join(self.source_languages), ','.join(self.target_languages)))

        filenames = self._get_dict_keys()
        logger.debug(filenames)

        # All dictionaries have language codes for the *target* languages, because the codes specify what language
        # to translate *into*, not *from*.
        files = {filename: MultilingualDictionary.load(os.path.join(self.args.data_dir, filename),
                                                       languages=self.target_languages)
                 for filename in set(filenames.values())}
        logger.debug(', '.join('{}: {:,}'.format(name, len(dictionary)) for name, dictionary in files.items()))

        self.dictionaries = {key: files[filename] for key, filename in filenames.items()}

        self._build_loss()

        if self.args.translation_noise == ['all']:
            self.noisy_languages = set(self.source_languages)
        else:
            self.noisy_languages = set(self.args.translation_noise)

    def _load_data(self, checkpoint):
        super(NMTTrainer, self)._load_data(checkpoint)
        args = checkpoint['args']
        self.args.join_lang_vocab = args.join_lang_vocab
        self.args.join_src_tgt_vocab = args.join_src_tgt_vocab

        self.source_languages = checkpoint['src_langs']
        self.target_languages = checkpoint['tgt_langs']

        checkpoint_keys = self._get_dict_keys()
        dictionaries = {}
        for key in set(checkpoint_keys.values()):
            dictionary = MultilingualDictionary(self.target_languages)
            dictionary.load_state_dict(checkpoint[key])
            dictionaries[key] = dictionary

        self.dictionaries = {key: dictionaries[value] for key, value in checkpoint_keys.items()}

        self._build_loss()

    def _save_data(self, checkpoint):
        super(NMTTrainer, self)._save_data(checkpoint)

        checkpoint['src_langs'] = self.source_languages
        checkpoint['tgt_langs'] = self.target_languages

        dict_keys = self._get_dict_keys()
        inv_keys = {v: k for k, v in dict_keys.items()}

        for checkpoint_key, dict_key in inv_keys.items():
            checkpoint[checkpoint_key] = self.dictionaries[dict_key].state_dict()
        return checkpoint

    def _build_loss(self):
        target_dicts = {lang: self.dictionaries[lang + '.tgt'] for lang in self.target_languages}

        losses = {}
        for dictionary in set(target_dicts.values()):
            loss = NMTLoss(len(dictionary), dictionary.pad(), self.args.label_smoothing)
            if self.args.cuda:
                loss.cuda()
            losses[dictionary] = loss
        self.losses = {lang: losses[dictionary] for lang, dictionary in target_dicts.items()}

    def _build_model(self, model_args):
        logger.info('Building {} model'.format(model_args.model))

        num_models = 1
        if model_args.output_select == 'separate_decoders':
            num_models = max(num_models, len(self.target_languages))
        if model_args.separate_encoders:
            num_models = max(num_models, len(self.source_languages))
        models = [build_model(model_args.model, model_args) for _ in range(num_models)]
        model = models[0]

        embedding_size = model_args.word_vec_size or getattr(model_args, 'model_size', None)
        if embedding_size is None:
            raise ValueError('Could not infer embedding size')

        if model_args.copy_decoder and not model_args.join_src_tgt_vocab:
            raise NotImplementedError('In order to use the copy decoder, the source and target language must '
                                      'use the same vocabulary')

        dummy_input = torch.zeros(1, 1, embedding_size)
        dummy_output, _ = model(dummy_input, dummy_input)
        output_size = dummy_output.size(-1)

        # Pre-trained embedding argument comes from self.args, not model_args, so we don't load them again
        # when loading the model form a checkpoint
        if hasattr(self.args, 'pre_word_vecs'):
            if len(self.args.pre_word_vecs) == 1 and '=' not in self.args.pre_word_vecs[0]:
                preload = {'': self.args.pre_word_vecs[0]}
            else:
                preload = {key: filename for key, filename in map(lambda x: x.split('='), self.args.pre_word_vecs)}
        else:
            preload = {}

        dict_keys = self._get_dict_keys()
        inv_dictionaries = {dictionary: key for key, dictionary in self.dictionaries.items()}

        embeddings = {}
        for dictionary, key in inv_dictionaries.items():
            embedding = self._get_embedding(model_args, dictionary, embedding_size,
                                            path=preload.get(dict_keys[key][:-5], None))
            embeddings[dictionary] = embedding

        if model_args.separate_encoders:
            encoders = {lang: NMTEncoder(m.encoder, embeddings[self.dictionaries[lang + '.src']],
                                         model_args.word_dropout)
                        for lang, m in zip(self.source_languages, models)}
            logger.debug('Number of encoders {}'.format(len(encoders)))
        else:
            source_dicts = set(self.dictionaries[lang + '.src'] for lang in self.source_languages)
            encoders = {dictionary: NMTEncoder(m.encoder, embeddings[dictionary], model_args.word_dropout)
                        for dictionary, m in zip(source_dicts, models)}
            logger.debug('Number of encoders {}'.format(len(encoders)))
            encoders = {lang: encoders[self.dictionaries[lang + '.src']] for lang in self.source_languages}

        if model_args.output_select == 'decoder_every_step':
            language_embedding = nn.Embedding(len(self.target_languages), embedding_size)
        else:
            language_embedding = None

        def make_decoder(decoder, embedding, linear):
            if model_args.copy_decoder:
                masked_layers = getattr(model_args, 'masked_layers', False)
                attention_dropout = getattr(model_args, 'attn_dropout', 0.0)
                return NMTDecoder(decoder, embedding, model_args.word_dropout, linear,
                                  copy_decoder=True,
                                  batch_first=model_args.batch_first,
                                  extra_attention=model_args.extra_attention,
                                  masked_layers=masked_layers,
                                  attention_dropout=attention_dropout,
                                  language_embedding=language_embedding)
            else:
                return NMTDecoder(decoder, embedding, model_args.word_dropout, linear,
                                  language_embedding=language_embedding)

        target_dicts = set(self.dictionaries[lang + '.tgt'] for lang in self.target_languages)
        linears = {dictionary: XavierLinear(output_size, len(dictionary)) for dictionary in target_dicts}

        if model_args.tie_weights:
            for dictionary in target_dicts:
                linears[dictionary].weight = embeddings[dictionary].weight

        if model_args.output_select == 'separate_decoders':
            decoders = {lang: make_decoder(m.decoder, embeddings[self.dictionaries[lang + '.tgt']],
                                           linears[self.dictionaries[lang + '.tgt']])
                        for lang, m in zip(self.target_languages, models)}
            logger.debug('Number of decoders {}'.format(len(decoders)))
        else:
            decoders = {dictionary: make_decoder(model.decoder, embeddings[dictionary], linears[dictionary])
                        for dictionary in target_dicts}
            logger.debug('Number of decoders {}'.format(len(decoders)))
            decoders = {lang: decoders[self.dictionaries[lang + '.tgt']] for lang in self.target_languages}

        # Share extra decoder parameters
        if model_args.copy_decoder:
            first_decoder = decoders[self.target_languages[0]]
            for decoder in decoders.values():
                if model_args.extra_attention:
                    decoder.attention = first_decoder.attention

            if model_args.join_lang_vocab == ['all']:
                for decoder in decoders.values():
                    decoder.merge_layer = first_decoder.merge_layer
            elif len(model_args.join_lang_vocab) > 0:
                first_decoder = decoders[model_args.join_lang_vocab[0]]
                for other_lang in model_args.join_lang_vocab[1:]:
                    decoders[other_lang].merge_layer = first_decoder.merge_layer

        self.model = self.MultilingualNMTModel(encoders, decoders)
        self.model.batch_first = model_args.batch_first
        self.model.output_select = model_args.output_select

    def _get_train_dataset(self):
        logger.info('Loading training data')
        split_words = self.args.input_type == 'word'
        src_bos = self.model.output_select == 'encoder_bos'
        tgt_lang_bos = self.model.output_select == 'decoder_bos'

        if self.data_mode == 'all_to_all':
            assert self.args.join_src_tgt_vocab
            filenames = {lang: name for lang, name in map(lambda x: x.split('='), self.args.train_data)}
            datasets = {}
            lengths = {}
            for lang in self.source_languages:  # == target_languages
                dictionary = self.dictionaries[lang + '.src']
                dataset, length = TextLookupDataset.load(filenames[lang], dictionary, self.args.data_dir,
                                                         self.args.load_into_memory, split_words,
                                                         bos=src_bos, eos=False, trunc_len=self.args.seq_length_trunc,
                                                         lower=self.args.lower)
                datasets[lang] = dataset
                lengths[lang] = length

            exclude_pairs = [pair.split('-') for pair in self.args.exclude_pairs]

            if len(self.noisy_languages) > 0:
                dataset = NoisyMultiParallelDataset(exclude_pairs, src_bos, tgt_lang_bos,
                                                    self.args.word_shuffle, self.args.noise_word_dropout,
                                                    self.args.word_blank, self.args.bpe_symbol,
                                                    self.noisy_languages,
                                                    **datasets)
            else:
                dataset = MultiParallelDataset(exclude_pairs, src_bos, tgt_lang_bos, **datasets)
            dataset.lengths = lengths
            logger.info('Number of training sentences: {:,d}'.format(dataset.num_sentences))
            logger.info('Number of training examples: {:,d}'.format(len(dataset)))
            return dataset
        else:
            pairs = [pair.split('-') for pair in self.args.langs]
            files = [(self.args.train_data[2 * i], self.args.train_data[2 * i + 1]) for i in range(len(pairs))]

            if self.args.invert_pairs:
                pairs.extend([(p[1], p[0]) for p in pairs])
                files.extend([(f[1], f[0]) for f in files])

            datasets = []
            src_lengths = []
            tgt_lengths = []
            for (source, target), (source_file, target_file) in zip(pairs, files):
                src_dataset, src_length = TextLookupDataset.load(source_file,
                                                                 self.dictionaries[source + '.src'],
                                                                 self.args.data_dir, self.args.load_into_memory,
                                                                 split_words, bos=src_bos, eos=False,
                                                                 trunc_len=self.args.seq_length_trunc,
                                                                 lang=target if src_bos else None)

                if source in self.noisy_languages:
                    src_dataset = NoisyTextDataset(src_dataset, self.args.word_shuffle, self.args.noise_word_dropout,
                                                   self.args.word_blank, self.args.bpe_symbol)

                tgt_dataset, tgt_length = TextLookupDataset.load(target_file,
                                                                 self.dictionaries[target + '.tgt'],
                                                                 self.args.data_dir, self.args.load_into_memory,
                                                                 split_words, bos=True, eos=True,
                                                                 trunc_len=self.args.seq_length_trunc,
                                                                 lang=target if tgt_lang_bos else None)
                dataset = ParallelDataset(src_dataset, tgt_dataset)
                datasets.append(dataset)
                src_lengths.append(src_length)
                tgt_lengths.append(tgt_length)

            dataset = ConcatDataset(*datasets, balance=self.args.balance_pairs)
            src_lengths = dataset.concat_lengths(*src_lengths)
            tgt_lengths = dataset.concat_lengths(*tgt_lengths)
            dataset.lengths = (src_lengths, tgt_lengths)
            logger.info('Number of training sentences: {:,d}'.format(sum(len(ds) for ds in datasets)))
            return dataset

    def _get_train_sampler(self, dataset):
        logger.info('Generating batches')

        def make_batches(src_lengths, tgt_lengths):
            def filter_fn(i):
                return src_lengths[i] <= self.args.seq_length and tgt_lengths[i] <= self.args.seq_length

            batches = data_utils.generate_length_based_batches_from_lengths(
                np.maximum(src_lengths, tgt_lengths), self.args.batch_size_words,
                self.args.batch_size_sents,
                self.args.batch_size_multiplier,
                self.args.pad_count,
                key_fn=lambda i: (tgt_lengths[i], src_lengths[i]),
                filter_fn=filter_fn)
            return batches

        num_encoders = len(set(map(id, self.model.encoders.items())))
        num_decoders = len(set(map(id, self.model.decoders.items())))
        mixed_batches = num_decoders == num_encoders == 1

        if self.data_mode == 'all_to_all':
            lengths = dataset.lengths
            if mixed_batches:
                logger.info('Creating mixed-language batches')
                src_lengths, tgt_lengths = dataset.concat_lengths(**lengths)
                batches = make_batches(src_lengths, tgt_lengths)
            else:
                logger.info('Creating single-language batches')
                batches = []
                for i, (source, target) in enumerate(dataset.get_pairs()):
                    part_batches = make_batches(lengths[source], lengths[target])

                    part_batches = [[x + i * dataset.num_sentences for x in batch] for batch in part_batches]
                    batches.extend(part_batches)

        else:
            src_lengths, tgt_lengths = dataset.lengths
            if mixed_batches:
                logger.info('Creating mixed-language batches')
                batches = make_batches(src_lengths, tgt_lengths)
            else:
                logger.info('Creating single-language batches')
                batches = []
                offset = 0
                src_lengths = np.split(src_lengths, len(dataset.datasets))
                tgt_lengths = np.split(tgt_lengths, len(dataset.datasets))
                for ds, src_length, tgt_length in zip(dataset.datasets, src_lengths, tgt_lengths):
                    part_batches = make_batches(src_length, tgt_length)
                    part_batches = [[x + offset for x in batch] for batch in part_batches]
                    batches.extend(part_batches)
                    offset += len(ds)
        logger.info('Number of training batches: {:,d}'.format(len(batches)))

        filtered = len(dataset) - sum(len(batch) for batch in batches)
        logger.info('Filtered {:,d}/{:,d} training examples for length'.format(filtered, len(dataset)))
        sampler = PreGeneratedBatchSampler(batches, self.args.curriculum == 0)
        return sampler

    def _forward_backward_pass(self, batch, metrics):
        src_size = batch.get('src_size')
        tgt_size = batch.get('tgt_size')

        # When args.join_lang_vocab is False, tgt_langs should be the same within a batch
        if self.args.input_type == 'all_to_all':
            src_langs = batch.get('src_lang')
            tgt_langs = batch.get('tgt_lang')
        else:
            inverted, index = zip(*[divmod(i, len(self.args.langs)) for i in batch.get('dataset_index')])
            pairs = [self.args.langs[i].split('-') for i in index]
            src_langs, tgt_langs = zip(*[(p[1], p[0]) if inv else p for p, inv in zip(pairs, inverted)])

        # Either all dictionaries are the same, or all languages within a batch are
        # The same is true for encoders
        # So either we have an encoder/decoder that can translate all languages, or all languages in
        # the batch are the same
        decoder = self.model.decoders[tgt_langs[0]]
        if self.model.output_select == 'decoder_every_step':
            decoder = self.DecoderWrapper(decoder, tgt_langs)
        model = EncoderDecoderModel(self.model.encoders[src_langs[0]], decoder)
        loss, display_loss = self._forward(batch, model, self.dictionaries[src_langs[0] + '.src'],
                                           self.dictionaries[tgt_langs[0] + '.tgt'], self.losses[tgt_langs[0]])
        self.optimizer.backward(loss)
        metrics['nll'].update(display_loss, tgt_size)
        metrics['src_tps'].update(src_size)
        metrics['tgt_tps'].update(tgt_size)
        metrics['total_words'].update(tgt_size)

    def _get_eval_dataset(self, task: TranslationTask):
        split_words = self.args.input_type == 'word'
        src_dataset = TextLookupDataset(task.src_dataset, self.dictionaries[task.source_language + '.src'],
                                        words=split_words, lower=task.lower, bos=False, eos=False,
                                        trunc_len=self.args.seq_length_trunc)

        if self.args.eval_noise:
            src_dataset = NoisyTextDataset(src_dataset, self.args.word_shuffle, self.args.noise_word_dropout,
                                           self.args.word_blank, self.args.bpe_symbol)

        if task.tgt_dataset is not None:
            tgt_dataset = TextLookupDataset(task.tgt_dataset, self.dictionaries[task.target_language + '.tgt'],
                                            words=split_words, lower=task.lower, bos=True, eos=True,
                                            trunc_len=self.args.seq_length_trunc)
        else:
            tgt_dataset = None
        dataset = ParallelDataset(src_dataset, tgt_dataset)
        return dataset

    def _eval_pass(self, task, batch, metrics):
        tgt_size = batch.get('tgt_size')
        src_dict = self.dictionaries[task.source_language + '.src']
        tgt_dict = self.dictionaries[task.target_language + '.tgt']
        loss = self.losses[task.target_language]
        model = EncoderDecoderModel(self.model.encoders[task.source_language],
                                    self.model.decoders[task.target_language])
        _, display_loss = self._forward(batch, model, src_dict, tgt_dict, loss, False)
        metrics['nll'].update(display_loss, tgt_size)

    def _inference_pass(self, task, batch, generator):
        encoder_input = batch.get('src_indices')
        source_lengths = batch.get('src_lengths')
        join_str = ' ' if self.args.input_type == 'word' else ''

        if not generator.batch_first:
            encoder_input = encoder_input.transpose(0, 1).contiguous()

        src_dict = self.dictionaries[task.source_language + '.src']
        tgt_dict = self.dictionaries[task.target_language + '.tgt']
        bos = tgt_dict.language_indices[task.target_language] if self.model.output_select == 'decoder_bos' else None

        encoder_mask = encoder_input.ne(src_dict.pad())

        res = [tgt_dict.string(tr['tokens'], join_str=join_str)
               for beams in generator.generate(encoder_input, source_lengths, encoder_mask, bos=bos)
               for tr in beams[:self.args.n_best]]
        src = []
        if self.args.print_translations:
            for i in range(len(batch['src_indices'])):
                ind = batch['src_indices'][i][:batch['src_lengths'][i]]
                ind = src_dict.string(ind, join_str=join_str, bpe_symbol=self.args.bpe_symbol)
                src.append(ind)
        return res, src

    def _restore_src_string(self, task, output, join_str, bpe_symbol):
        return self.dictionaries[task.source_language + '.src']\
            .string(output, join_str=join_str, bpe_symbol=bpe_symbol)

    def _restore_tgt_string(self, task, output, join_str, bpe_symbol):
        return self.dictionaries[task.target_language + '.tgt']\
            .string(output, join_str=join_str, bpe_symbol=bpe_symbol)

    def _get_sequence_generator(self, task):
        decoder = self.model.decoders[task.target_language]
        if self.model.output_select == 'decoder_every_step':
            decoder = self.DecoderWrapper(decoder, task.target_language)
        model = EncoderDecoderModel(self.model.encoders[task.source_language], decoder)
        return SequenceGenerator([model], self.dictionaries[task.target_language + '.tgt'], self.model.batch_first,
                                 self.args.beam_size, maxlen_b=20, normalize_scores=self.args.normalize,
                                 len_penalty=self.args.alpha, unk_penalty=self.args.beta)
