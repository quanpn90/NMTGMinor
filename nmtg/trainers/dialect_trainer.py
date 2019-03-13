import logging
from collections import OrderedDict

from nmtg.data.dataset import MultiDataset
from nmtg.data.samplers import MultiSampler
from nmtg.models import Model, build_model
from nmtg.models.encoder_decoder import EncoderDecoderModel
from nmtg.modules.nmt import NMTDecoder
from nmtg.modules.linear import XavierLinear
from nmtg.modules.loss import NMTLoss
from nmtg.sequence_generator import SequenceGenerator
from nmtg.trainers import register_trainer
from nmtg.trainers.nmt_trainer import NMTTrainer

logger = logging.getLogger(__name__)


# Deprecated, can be achieved through multilingualTrainer now
@register_trainer('dialect')
class DialectTrainer(NMTTrainer):

    class DialectTranslationModel(Model):
        def __init__(self, translation_model, denoising_model):
            super().__init__()
            self.translation_model = translation_model
            self.denoising_model = denoising_model

    @classmethod
    def add_training_options(cls, parser, argv=None):
        super().add_training_options(parser, argv)
        parser.add_argument('-train_clean', required=True,
                            help='Clean data for denoising')
        parser.add_argument('-train_noisy',
                            help='Noisy data for denoising')
        parser.add_argument('-tie_dual_weights', action='store_true',
                            help='Share weights between embedding and second softmax')
        parser.add_argument('-freeze_dual_embeddings', action='store_true',
                            help='Do not train other word embeddings')

        parser.add_argument('-train_clean', type=str, required=True,
                            help='Path to the clean training data')
        parser.add_argument('-train_noisy', type=str,
                            help='(Optional) training data with pre-generated noise. Further noise will be applied')

    def _build_loss(self):
        loss = NMTLoss(len(self.tgt_dict), self.tgt_dict.pad(), self.args.label_smoothing)
        if self.args.cuda:
            loss.cuda()
        self.translation_loss = loss

        loss = NMTLoss(len(self.src_dict), self.src_dict.pad(), self.args.label_smoothing)
        if self.args.cuda:
            loss.cuda()
        self.denoising_loss = loss

    def _build_model(self, model_args):
        logger.info('Building translation model')
        super()._build_model(model_args)
        translation_model = self.model

        logger.info('Building denoising model')
        denoising_model = build_model(model_args.model, model_args)
        del denoising_model.encoder

        embedding = translation_model.encoder.embedded_dropout.embedding
        linear = XavierLinear(translation_model.decoder.linear.weight.size(1), len(self.src_dict))

        if model_args.tie_dual_weights:
            linear.weight = embedding.weight

        if model_args.copy_decoder:
            masked_layers = getattr(model_args, 'masked_layers', False)
            attention_dropout = getattr(model_args, 'attn_dropout', 0.0)
            decoder = NMTDecoder(denoising_model.decoder, embedding, model_args.word_dropout, linear,
                                 copy_decoder=True,
                                 batch_first=model_args.batch_first,
                                 extra_attention=model_args.extra_attention,
                                 masked_layers=masked_layers,
                                 attention_dropout=attention_dropout)
        else:
            decoder = NMTDecoder(denoising_model.decoder, embedding, model_args.word_dropout, linear)

        denoising_model = EncoderDecoderModel(translation_model.encoder, decoder)
        compound_model = self.DialectTranslationModel(translation_model, denoising_model)
        compound_model.batch_first = translation_model.batch_first
        self.model = compound_model

    def _get_train_dataset(self):
        logger.info('Loading parallel data')
        parallel_dataset = super()._get_train_dataset()

        logger.info('Loading denoising data')
        translation_noise = self.args.translation_noise
        train_src = self.args.train_src
        train_tgt = self.args.train_tgt
        self.args.train_src = self.args.train_noisy or self.args.train_clean
        self.args.train_tgt = self.args.train_clean
        self.args.translation_noise = True

        denoising_dataset = super()._get_train_dataset()

        self.args.translation_noise = translation_noise
        self.args.train_src = train_src
        self.args.train_tgt = train_tgt

        dataset = MultiDataset(parallel_dataset, denoising_dataset)
        return dataset

    def _get_train_sampler(self, dataset):
        parallel_sampler = super()._get_train_sampler(dataset.datasets[0])
        denoising_sampler = super()._get_train_sampler(dataset.datasets[1])
        return MultiSampler(parallel_sampler, denoising_sampler)

    def _forward_backward_pass(self, batch, metrics):
        parallel_batch, denoising_batch = batch
        parallel_loss, parallel_display_loss = self._forward(parallel_batch, self.model.translation_model,
                                                             self.translation_loss)

        denoising_loss, denoising_display_loss = self._forward(parallel_batch, self.model.denoising_mdoel,
                                                               self.denoising_loss)
        loss = parallel_loss + denoising_loss
        self.optimizer.backward(loss)
        display_loss = parallel_display_loss + denoising_display_loss
        src_size = parallel_batch.get('src_size') + denoising_batch.get('src_size')
        tgt_size = parallel_batch.get('tgt_size') + denoising_batch.get('tgt_size')
        metrics['nll'].update(display_loss, tgt_size)
        metrics['src_tokens'] += src_size
        metrics['tgt_tokens'] += tgt_size

    def _eval_pass(self, batch, metrics):
        tgt_size = batch.get('tgt_size')
        _, display_loss = self._forward(batch, self.model.translation_model, self.loss.translation_loss, False)
        metrics['nll'].update(display_loss, tgt_size)

    def _get_sequence_generator(self):
        return SequenceGenerator([self.model.translation_model], self.tgt_dict, self.model.batch_first,
                                 self.args.beam_size, maxlen_b=20, normalize_scores=self.args.normalize,
                                 len_penalty=self.args.alpha, unk_penalty=self.args.beta)

    # noinspection PyProtectedMember
    @classmethod
    def upgrade_checkpoint(cls, checkpoint):
        super().upgrade_checkpoint(checkpoint)
        if 'denoising_model' in checkpoint:
            new_state_dict = _join_state_dicts(translation_model=checkpoint['model'],
                                               denoising_model=checkpoint['denoising_model'])
            checkpoint['model'] = new_state_dict
            del checkpoint['denoising_model']
        elif any(k.startswith('first_model') for k in checkpoint['model'].keys()):
            translation, denoising = _split_by_key(checkpoint['model'], 'first_model')
            new_state_dict = _join_state_dicts(translation_model=translation,
                                               denoising_model=denoising)
            checkpoint['model'] = new_state_dict


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
