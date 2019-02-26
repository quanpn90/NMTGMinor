import logging

from torch import Tensor
from typing import Sequence

from nmtg.data.dataset import MultiDataset
from nmtg.data.samplers import MultiSampler
from nmtg.models.nmt_model import NMTDualDecoder
from nmtg.modules.loss import NMTLoss
from nmtg.optim.optimizer import MultiOptimizer
from nmtg.trainers import register_trainer
from nmtg.trainers.denoising_text_trainer import DenoisingTextTrainer
from nmtg.trainers.nmt_trainer import NMTTrainer
from nmtg.trainers.trainer import TrainData


logger = logging.getLogger(__name__)


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
        NMTDualDecoder.add_options(parser)

        parser.add_argument('-train_clean', type=str, required=True,
                            help='Path to the clean training data')
        parser.add_argument('-train_noisy', type=str,
                            help='(Optional) training data with pre-generated noise. Further noise will be applied')
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
        model = super(NMTTrainer, self)._build_model(args)
        second_model = super(NMTTrainer, self)._build_model(args)
        return NMTDualDecoder.wrap_model(args, model, second_model, self.src_dict, self.tgt_dict, self.src_dict)

    def load_data(self, model_args=None):
        parallel_data, parallel_sampler = self._load_parallel_dataset()

        self.dictionary = self.src_dict
        self.args.seq_length = self.args.src_seq_length
        self.args.seq_length_trunc = self.args.src_seq_length_trunc
        # noinspection PyProtectedMember
        noisy_data, noisy_sampler = DenoisingTextTrainer._load_noisy_data(self)

        dataset = MultiDataset(parallel_data, noisy_data)
        sampler = MultiSampler(parallel_sampler, noisy_sampler)
        model = self.build_model(model_args)
        main_params = list(filter(lambda p: p.requires_grad, model.first_model.parameters()))
        second_params = list(filter(lambda p: p.requires_grad, model.second_model.parameters()))
        lr_scheduler, main_optim = self._build_optimizer(main_params)
        _, second_optim = self._build_optimizer(second_params)
        optimizer = MultiOptimizer(main_optim, second_optim)
        return TrainData(model, dataset, sampler, lr_scheduler, optimizer, self._get_training_metrics())

    def _get_loss(self, model, batch) -> (Tensor, float):
        if isinstance(batch, Sequence):
            # Multi evaluation
            self.loss = self.translation_loss
            translation_loss, display_loss = super()._get_loss(model.first_model, batch[0])
            self.loss = self.denoising_loss
            denoising_loss, _ = super()._get_loss(model.second_model, batch[1])
            return (translation_loss, denoising_loss), display_loss
        else:
            # Just translation
            self.loss = self.translation_loss
            return super()._get_loss(model.first_model, batch)

    def _get_batch_weight(self, batch):
        if isinstance(batch, Sequence):
            return batch[0]['tgt_size']
        else:
            return batch['tgt_size']

    def _update_training_metrics(self, train_data, batch):
        return super()._update_training_metrics(train_data, batch[0])

    def solve(self, model_or_ensemble, task):
        if isinstance(model_or_ensemble, Sequence):
            return super().solve([model.first_model for model in model_or_ensemble], task)
        else:
            return super().solve(model_or_ensemble.first_model, task)

