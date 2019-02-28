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
from nmtg.trainers import register_trainer
from nmtg.trainers.denoising_text_trainer import DenoisingTextTrainer
from nmtg.trainers.nmt_trainer import NMTTrainer
from nmtg.trainers.trainer import TrainData

logger = logging.getLogger(__name__)


class DialectTrainData(TrainData):

    def __init__(self, model, denoising_model, dataset, sampler, lr_scheduler, optimizer, meters):
        super().__init__(model, dataset, sampler, lr_scheduler, optimizer, meters)
        self.denoising_model = denoising_model

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['denoising_mdoel'] = self.denoising_model
        return state_dict

    def load_state_dict(self, state_dict, reset_optim=False):
        self.denoising_model.load_state_dict(state_dict['denoising_model'])
        super().load_state_dict(state_dict, reset_optim)


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

        loss = NMTLoss(len(self.src_dict), self.src_dict.pad(), self.args.label_smoothing)
        if self.args.cuda:
            loss.cuda()
        self.denoising_loss = loss

    def _build_model(self, args):
        model = super()._build_model(args)
        second_model = super(NMTTrainer, self)._build_model(args)

        embedding = model.encoder.embedded_dropout.embedding

        linear = XavierLinear(model.decoder.linear.weight.size(1), len(self.src_dict))

        if args.tie_dual_weights:
            linear.weight = embedding.weight

        second_decoder = NMTDecoder(second_model.decoder, embedding, args.word_dropout, linear)
        second_model = NMTModel(model.encoder, second_decoder, self.src_dict, self.src_dict, model.batch_first)
        compound_model = Model()
        compound_model.first_model = model
        compound_model.second_model = second_model
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
        second_model = model.second_model
        model = model.first_model
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        lr_scheduler, optimizer = self._build_optimizer(params)
        return DialectTrainData(model, second_model, dataset, sampler, lr_scheduler, optimizer,
                                self._get_training_metrics())

    def _get_loss_train(self, train_data, batch) -> (Tensor, float):
        # Multi evaluation
        self.loss = self.translation_loss
        translation_loss, translation_display_loss = self._get_loss(train_data.model, batch[0])

        self.loss = self.denoising_loss
        denoising_loss, denoising_display_loss = self._get_loss(train_data.second_model, batch[1])

        return translation_loss + denoising_loss, translation_display_loss + denoising_display_loss

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

    def load_checkpoint(self, checkpoint, for_training=False, reset_optim=False):
        if not for_training:
            raise NotImplementedError('To evaluate, use NMTTrainer or DenoisingTextTrainer, '
                                      'this class if for training only.')
        else:
            return super().load_checkpoint(checkpoint, for_training, reset_optim)

    # noinspection PyProtectedMember
    @staticmethod
    def upgrade_checkpoint(checkpoint):
        if 'denoising_model' not in checkpoint['train_data']:
            model_dict = checkpoint['train_data']['model']
            metadata = checkpoint['train_data']['model']._metadata
            first, second = _split_by_key(model_dict, 'first_model')
            checkpoint['train_data']['model'] = first
            checkpoint['train_data']['denoising_model'] = second

            first, second = _split_by_key(metadata, 'first_model')
            checkpoint['train_data']['model']._metadata = first
            checkpoint['train_data']['denoising_model']._metadata = second

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
    return first_output, second_output
