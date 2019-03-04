import logging
from collections import OrderedDict
from typing import Sequence

from torch import Tensor

from nmtg.data.dataset import MultiDataset
from nmtg.data.noisy_text import NoisyTextDataset
from nmtg.data.samplers import MultiSampler
from nmtg.models import Model
from nmtg.models.nmt_model import NMTModel, NMTDecoder
from nmtg.modules.linear import XavierLinear
from nmtg.modules.loss import NMTLoss
from nmtg.tasks.denoising_text_task import DenoisingTextTask
from nmtg.trainers import register_trainer
from nmtg.trainers.denoising_text_trainer import DenoisingTextTrainer
from nmtg.trainers.nmt_trainer import NMTTrainer
from nmtg.trainers.trainer import TrainData

logger = logging.getLogger(__name__)


class DialectTranslationModel(Model):
    def __init__(self, translation_model, denoising_model):
        super().__init__()
        self.translation_model = translation_model
        self.denoising_model = denoising_model


@register_trainer('dialect')
class DialectTrainer(NMTTrainer):

    @classmethod
    def add_preprocess_options(cls, parser):
        super().add_preprocess_options(parser)
        parser.add_argument('-train_clean', type=str, required=True,
                            help='Path to the clean training data')
        parser.add_argument('-train_noisy', type=str,
                            help='(Optional) training data with pre-generated noise. Further noise will be applied')

    @classmethod
    def add_training_options(cls, parser):
        super().add_training_options(parser)
        parser.add_argument('-tie_dual_weights', action='store_true',
                            help='Share weights between embedding and second softmax')
        parser.add_argument('-freeze_dual_embeddings', action='store_true',
                            help='Do not train other word embeddings')

        parser.add_argument('-train_clean', type=str, required=True,
                            help='Path to the clean training data')
        parser.add_argument('-train_noisy', type=str,
                            help='(Optional) training data with pre-generated noise. Further noise will be applied')
        parser.add_argument('-translation_noise', action='store_true',
                            help='Also apply noise to the translation input')
        parser.add_argument('-word_shuffle', type=int, default=3,
                            help='Maximum number of positions a word can move (0 to disable)')
        parser.add_argument('-word_blank', type=float, default=0.2,
                            help='Probability to replace a word with the unknown word (0 to disable)')
        parser.add_argument('-noise_word_dropout', type=float, default=0.1,
                            help='Probability to remove a word (0 to disable)')

    @staticmethod
    def preprocess(args):
        logger.info('Preprocessing parallel data')
        NMTTrainer.preprocess(args)

        args.vocab = args.src_vocab
        args.vocab_size = args.src_vocab_size
        logger.info('Preprocessing monolingual data')
        DenoisingTextTrainer.preprocess(args)

    def _build_loss(self):
        loss = NMTLoss(len(self.tgt_dict), self.tgt_dict.pad(), self.args.label_smoothing)
        if self.args.cuda:
            loss.cuda()
        self.translation_loss = loss
        self.loss = loss

        loss = NMTLoss(len(self.src_dict), self.src_dict.pad(), self.args.label_smoothing)
        if self.args.cuda:
            loss.cuda()
        self.denoising_loss = loss

    def _build_model(self, args):
        translation_model = super()._build_model(args)
        denoising_model = super(NMTTrainer, self)._build_model(args)
        del denoising_model.encoder

        embedding = translation_model.encoder.embedded_dropout.embedding
        linear = XavierLinear(translation_model.decoder.linear.weight.size(1), len(self.src_dict))

        if args.tie_dual_weights:
            linear.weight = embedding.weight

        denoising_decoder = NMTDecoder(denoising_model.decoder, embedding, args.word_dropout, linear)
        denoising_model = NMTModel(translation_model.encoder, denoising_decoder, self.src_dict, self.src_dict,
                                   translation_model.batch_first, translation_model.freeze_old)
        compound_model = DialectTranslationModel(translation_model, denoising_model)
        return compound_model

    def load_data(self, model_args=None):
        parallel_data, parallel_sampler = self._load_parallel_dataset()

        if self.args.translation_noise:
            src_dataset = parallel_data.src_data
            src_dataset = NoisyTextDataset(src_dataset, self.args.word_shuffle,
                                           self.args.noise_word_dropout, self.args.word_blank,
                                           self.args.bpe_symbol)
            # This means the batching will be less efficient, but still ok.
            # At worst, there will be somewhat more padding, but the batches will not become larger
            parallel_data.src_dataset = src_dataset

        self.dictionary = self.src_dict
        self.args.seq_length = self.args.src_seq_length
        self.args.seq_length_trunc = self.args.src_seq_length_trunc
        # noinspection PyProtectedMember
        noisy_data, noisy_sampler = DenoisingTextTrainer._load_noisy_data(self)

        dataset = MultiDataset(parallel_data, noisy_data)
        sampler = MultiSampler(parallel_sampler, noisy_sampler)
        model = self.build_model(model_args)
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        lr_scheduler, optimizer = self._build_optimizer(params)
        return TrainData(model, dataset, sampler, lr_scheduler, optimizer, self._get_training_metrics())

    def _get_loss(self, model, batch) -> (Tensor, float):
        if isinstance(batch, Sequence):
            # Multi evaluation
            self.loss = self.denoising_loss
            denoising_loss, denoising_display_loss = super()._get_loss(model.denoising_model, batch[1])

            self.loss = self.translation_loss
            translation_loss, translation_display_loss = super()._get_loss(model.translation_model, batch[0])

            return translation_loss + denoising_loss, translation_display_loss + denoising_display_loss
        else:
            # Single evaluation
            return super()._get_loss(model.translation_model, batch)

    def _get_batch_weight(self, batch):
        # This overnormalizes the decoders. Hopefully, that's ok...
        if isinstance(batch, Sequence):
            return batch[0]['tgt_size'] + batch[1]['tgt_size']
        else:
            return batch['tgt_size']

    def _update_training_metrics(self, train_data, batch):
        meters = train_data.meters
        batch_time = meters['fwbw_wall'].val
        src_tokens = batch[0]['src_size'] + batch[1]['src_size']
        tgt_tokens = batch[0]['tgt_size'] + batch[1]['tgt_size']

        meters['srctok'].update(src_tokens, batch_time)
        meters['tgttok'].update(tgt_tokens, batch_time)

        return ['{:5.0f}|{:5.0f} tok/s'.format(meters['srctok'].avg, meters['tgttok'].avg)]

    def solve(self, model_or_ensemble, task):
        if not isinstance(model_or_ensemble, Sequence):
            model_or_ensemble = [model_or_ensemble]

        if isinstance(task, DenoisingTextTask):
            return DenoisingTextTrainer.solve(self, [model.denoising_model for model in model_or_ensemble], task)
        else:
            return NMTTrainer.solve(self, [model.translation_model for model in model_or_ensemble], task)

    # noinspection PyProtectedMember
    @staticmethod
    def upgrade_checkpoint(checkpoint):
        if 'denoising_model' in checkpoint['train_data']:
            new_state_dict = _join_state_dicts(translation_model=checkpoint['train_data']['model'],
                                               denoising_model=checkpoint['train_data']['denoising_model'])
            checkpoint['train_data']['model'] = new_state_dict
            del checkpoint['train_data']['denoising_model']
        elif any(k.startswith('first_model') for k in checkpoint['train_data']['model'].keys()):
            translation, denoising = _split_by_key(checkpoint['train_data']['model'], 'first_model')
            new_state_dict = _join_state_dicts(translation_model=translation,
                                               denoising_model=denoising)
            checkpoint['train_data']['model'] = new_state_dict
        args = checkpoint['args']
        if 'translation_noise' not in args:
            args.translation_noise = False
        NMTTrainer.upgrade_checkpoint(checkpoint)


def _split_by_key(data, prefix):
    first_output = OrderedDict()
    second_output = OrderedDict()

    for key, value in data.items():
        parts = key.split('.')
        new_key = '.'.join(parts[1:])
        if parts[0] == prefix:
            first_output[new_key] = value
        else:
            second_output[new_key] = value

    if hasattr(data, '_metadata'):
        first_metadata, second_metadata = _split_by_key(data._metadata, prefix)
        first_output._metadata = first_metadata
        second_output._metadata = second_metadata
    return first_output, second_output


def _join_state_dicts(**state_dicts):
    new_state_dict = OrderedDict()
    new_metadata = OrderedDict()
    for name, state_dict in state_dicts.items():
        for k, v in state_dict.items():
            new_state_dict[name + '.' + k] = v
        if hasattr(state_dict, '_metadata'):
            for k, v in state_dict._metadata.items():
                if k == '':
                    new_metadata[name] = v
                else:
                    new_metadata[name + '.' + k] = v
    new_state_dict._metadata = new_metadata
    return new_state_dict
