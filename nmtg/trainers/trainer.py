import datetime
import logging
import math
import os
import re
from typing import Sequence

import torch
import torch.utils.data
from torch import Tensor
from tqdm import tqdm

import nmtg.models
import nmtg.optim
from nmtg.optim import MemoryEfficientFP16Optimizer, FP16Optimizer
from nmtg.optim.lr_scheduler import LRScheduler
from nmtg.tasks import Task

logger = logging.getLogger(__name__)


class TrainData:
    def __init__(self, model, dataset, sampler, lr_scheduler, optimizer):
        self.model = model
        self.dataset = dataset
        self.sampler = sampler
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.epoch = 1
        self.training_start_time = None
        self.num_updates = 0

    def state_dict(self):
        return {
            'model': self.model.state_dict(),
            'epoch': self.epoch,
            'sampler': self.sampler.state_dict(),
            'num_updates': self.num_updates,
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict, reset_optim=False):
        self.epoch = state_dict['epoch']
        self.num_updates = state_dict['num_updates']
        self.model.load_state_dict(state_dict['model'])
        self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        self.sampler.load_state_dict(state_dict['sampler'])

        if reset_optim:
            self.sampler.reset()
        else:
            self.optimizer.load_state_dict(state_dict['optimizer'])


class Trainer:
    """Trainer solves a Task using a Model"""

    @classmethod
    def add_preprocess_options(cls, parser):
        """Options relating to preprocessing"""
        pass

    @classmethod
    def add_general_options(cls, parser):
        parser.add_argument('-data_loader_threads', type=int, default=1,
                            help='Number of threads for data loading')
        parser.add_argument('-batch_size', type=int, default=32,
                            help='Batch size for evaluation')

    @classmethod
    def add_training_options(cls, parser):
        """Options relating to training"""
        cls.add_general_options(parser)
        parser.add_argument('-model', choices=nmtg.models.get_model_names(), required=True,
                            help='Model type')
        parser.add_argument('-update_method',
                            choices=nmtg.optim.lr_scheduler.get_lr_scheduler_names(),
                            help='Type of update rule to use')
        parser.add_argument('-optimizer',
                            choices=nmtg.optim.get_optimizer_names(),
                            help='Optimization method')

        args, _ = parser.parse_known_args()

        model_class = nmtg.models.get_model_type(args.model)
        model_class.add_options(parser)
        # model may have added new defaults for optimizer and learning rate scheduler
        args, _ = parser.parse_known_args()

        optimizer_class = nmtg.optim.get_optimizer_type(args.optimizer)
        optimizer_class.add_options(parser)
        lr_scheduler_class = nmtg.optim.lr_scheduler.get_lr_scheduler_type(args.update_method)
        lr_scheduler_class.add_options(parser)

        parser.add_argument('-curriculum', type=int, default=0,
                            help='For this many epochs, order the minibatches based '
                                 'on source sequence length. Sometimes setting this to 1 will '
                                 'increase convergence speed.')
        parser.add_argument('-batch_size_update', type=int, default=2048,
                            help='Perform a learning step when total batch size exceeds this value')
        parser.add_argument('-normalize_gradient', action='store_true',
                            help='Normalize the gradients by number of tokens before updates')
        parser.add_argument('-max_grad_norm', type=float, default=0,
                            help="If the norm of the gradient vector exceeds this, "
                                 "renormalize it to have the norm equal to max_grad_norm")
        parser.add_argument('-save_every', type=int, default=-1,
                            help='Save model after n training steps.')
        parser.add_argument('-log_interval', type=int, default=100,
                            help='Print stats at this interval.')
        parser.add_argument('-epochs', type=int, default=13,
                            help='Number of training epochs')
        parser.add_argument('-save_model', default='model',
                            help='Model filename (the model will be saved as '
                                 '<save_model>_epochN_PPL.pt where PPL is the '
                                 'validation perplexity')
        parser.add_argument('-keep_save_files', type=int, default=5,
                            help='Keep this many save files.')
        parser.add_argument('-memory_efficient_fp16', action='store_true',
                            help='Use alternative, more memory efficient implementation of fp16 training.')

    @classmethod
    def add_eval_options(cls, parser):
        """Options relating to evaluation"""
        cls.add_general_options(parser)

    @staticmethod
    def preprocess(args):
        """
        Perform any necessary preprocessing for training
        :param args: Command line arguments
        """
        pass

    def __init__(self, args):
        self.args = args

    def _build_model(self, args):
        logger.info('Building {} model'.format(args.model))
        model_class = nmtg.models.get_model_type(args.model)
        model = model_class.build_model(args)
        return model

    def load_data(self, model_args=None):
        raise NotImplementedError

    def _build_optimizer(self, model):
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        if self.args.fp16:
            if self.args.cuda and torch.cuda.get_device_capability(0)[0] < 7:
                logger.warning('Your device does NOT support faster training with --fp16, '
                               'please switch to FP32 which is likely to be faster')
            if self.args.memory_efficient_fp16:
                optimizer = MemoryEfficientFP16Optimizer.build_optimizer(self.args, params)
            else:
                optimizer = FP16Optimizer.build_optimizer(self.args, params)
        else:
            if self.args.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
                logger.info('Your device may support faster training with --fp16')
            optimizer = nmtg.optim.get_optimizer_type(self.args.optimizer).build_optimizer(self.args, params)
        lr_scheduler = nmtg.optim.lr_scheduler.get_lr_scheduler_type(self.args.update_method) \
            .build_lr_scheduler(self.args, optimizer)
        return lr_scheduler, optimizer

    def solve(self, model_or_ensemble, task):
        """
        Solve the task
        :return The results generated by this solution
        """
        models = model_or_ensemble
        if not isinstance(models, Sequence):
            models = [model_or_ensemble]

        for model in models:
            model.eval()

        iterator = self._get_eval_iterator(task, self.args.batch_size)

        results = [result
                   for batch in tqdm(iterator, disable=self.args.no_progress)
                   for model in models
                   for result in self._inference_step(model, batch)]

        return results

    def _inference_step(self, model, batch):
        return model(batch)

    def evaluate(self, model, task):
        """
        Calculate the average loss on the task
        :return (loss, perplexity)
        """
        model.eval()
        iterator = self._get_eval_iterator(task, self.args.batch_size)
        total_weight, total_loss = 0, 0

        with torch.no_grad():
            for batch in tqdm(iterator, postfix='evaluation', disable=self.args.no_progress):
                loss, display_loss = self._get_loss(model, batch)
                total_loss += display_loss
                total_weight += self._get_batch_weight(batch)

        loss = total_loss / (total_weight + 1e-6)
        ppl = math.exp(min(loss, 100))
        return loss, ppl

    def _get_eval_iterator(self, task, batch_size):
        return task.dataset.get_iterator(batch_size=batch_size,
                                         num_workers=self.args.data_loader_threads,
                                         cuda=self.args.cuda)

    def _get_loss(self, model, batch) -> (Tensor, float):
        """
        Calculate the loss for a given batch.
        :return The loss tensor and the display loss for training/eval statistics.
            The display loss should usually not include L2 normalization and label smoothing
        """
        raise NotImplementedError

    def _get_batch_weight(self, batch):
        return len(batch)

    def _get_training_metrics(self, batch, step_time: datetime.timedelta):
        raise NotImplementedError

    def _get_train_iterator(self, train_data: TrainData):
        return train_data.dataset.get_iterator(batch_sampler=train_data.sampler,
                                               num_workers=self.args.data_loader_threads,
                                               cuda=self.args.cuda)

    def train(self, train_data: TrainData, eval_task: Task = None):
        if self.args.fp16:
            train_data.model.half()
        if self.args.cuda:
            train_data.model.cuda()

        logger.debug('{:,d} parameters to train'.format(sum(torch.numel(p) for p in train_data.optimizer.params)))

        if eval_task is not None:
            valid_loss, valid_ppl = self.evaluate(train_data.model, eval_task)
            logger.info('Starting validation perplexity {:g}'.format(valid_ppl))

        train_data.training_start_time = datetime.datetime.now()

        for _ in range(train_data.epoch, self.args.epochs + 1):
            self._train_epoch(train_data, eval_task)
            train_data.sampler.shuffle = train_data.epoch >= self.args.curriculum
            train_data.sampler.reset()
            train_data.epoch += 1

            eval_results = self.solve(train_data.model, eval_task)
            eval_metrics = eval_task.score_results(eval_results)
            logger.info(' | '.join(eval_metrics))

        if eval_task is not None:
            valid_loss, valid_ppl = self.evaluate(train_data.model, eval_task)
            logger.info('Final validation perplexity: {:g}'.format(valid_ppl))
            results = self.solve(train_data.model, eval_task)
            eval_metrics = eval_task.score_results(results)
            logger.info(' | '.join(eval_metrics))
        else:
            valid_ppl = 0.0
        self.save_checkpoint(train_data, valid_ppl)

    def _train_epoch(self, train_data: TrainData, eval_task: Task = None):
        train_data.model.train()
        train_data.optimizer.zero_grad()

        iterator = self._get_train_iterator(train_data)

        oom_count = 0
        total_weight = 0
        total_loss = 0
        grad_norm = 0
        with tqdm(total=len(train_data.sampler), initial=train_data.sampler.index + 1,
                  disable=self.args.no_progress) as pbar:
            for index, batch in enumerate(iterator, train_data.sampler.index + 1):
                start_time = datetime.datetime.now()
                try:
                    loss, display_loss = self._get_loss(train_data.model, batch)
                    train_data.optimizer.backward(loss)
                except RuntimeError as e:
                    if 'out of memory' in str(e) or 'get_temporary_buffer' in str(e):
                        logger.warning('Ran out of memory in step {}'.format(train_data.sampler.index))
                        self._reset_state(train_data)
                        if self.args.cuda:
                            torch.cuda.empty_cache()
                        oom_count += 1
                        pbar.update()
                        continue
                    else:
                        raise e
                time_for_batch = datetime.datetime.now() - start_time

                specific_metrics = self._get_training_metrics(batch, time_for_batch)
                total_weight += self._get_batch_weight(batch)
                total_loss += display_loss
                perplexity = math.exp(total_loss / (total_weight + 1e-6))

                # TODO: factor 0.95?
                if total_weight >= self.args.batch_size_update:
                    try:
                        grad_norm = self._learning_step(train_data, total_weight)
                        train_data.num_updates += 1
                        train_data.lr_scheduler.step_update(train_data.num_updates)
                    except OverflowError as e:
                        logger.warning('Overflow detected ' + str(e))
                        train_data.optimizer.zero_grad()
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            logger.warning('Ran out of memory in step {}'.format(train_data.sampler.index))
                            self._reset_state(train_data)
                            if self.args.cuda:
                                torch.cuda.empty_cache()
                            oom_count += 1
                        else:
                            raise e
                    total_weight = 0
                    total_loss = 0

                    train_data.optimizer.zero_grad()

                    if self.args.save_every > 0 and (train_data.num_updates + 1) % self.args.save_every == 0:
                        if eval_task is not None:
                            eval_loss, eval_ppl = self.evaluate(train_data.model, eval_task)
                            logger.info('Validation perplexity: {:g}'.format(eval_ppl))
                            train_data.model.train()
                        else:
                            eval_ppl = 0.0

                        self.save_checkpoint(train_data, eval_ppl)

                progress_metrics = ['Epoch: {:2d}'.format(train_data.epoch),
                                    'Updates: {:5d}'.format(train_data.num_updates),
                                    'ppl: {:6.2f}'.format(perplexity),
                                    'lr: {:.4e}'.format(train_data.optimizer.get_lr()),
                                    'gnorm: {:.2f}'.format(grad_norm)]
                pbar.set_postfix_str(' | '.join(progress_metrics + specific_metrics))
                if index == 0 or (index + 1) % self.args.log_interval == 0:
                    elapsed_time = datetime.datetime.now() - train_data.training_start_time
                    log_metrics = ['{:5d}/{:5d}'.format(index + 1, len(iterator)),
                                   '{} elapsed'.format(str(elapsed_time).split('.')[0])]
                    log_str = " | ".join(log_metrics + progress_metrics + specific_metrics)
                    logger.info(log_str)
                pbar.update()

    def _learning_step(self, train_data, total_weight):
        if self.args.normalize_gradient:
            train_data.optimizer.multiply_grads(1. / total_weight)

        grad_norm = train_data.optimizer.clip_grad_norm(self.args.max_grad_norm)

        train_data.optimizer.step()
        return grad_norm

    def _reset_state(self, train_data: TrainData):
        def reset_state(m):
            if hasattr(m, 'reset_state'):
                m.reset_state()

        train_data.model.apply(reset_state)
        train_data.optimizer.zero_grad()

    def save_checkpoint(self, train_data: TrainData, valid_ppl, destination=None):
        logger.info('Saving checkpoint')
        if destination is None:
            destination = self.args.save_model
        os.makedirs(destination, exist_ok=True)
        checkpoint = self.state_dict()
        checkpoint['train_data'] = train_data.state_dict()
        checkpoint['args'] = self.args

        ep = float(train_data.epoch) - 1. + ((float(train_data.sampler.index) + 1.) / len(train_data.sampler))

        file_name = '{}_ppl_{:.2f}_e{:.2f}.pt'.format(self.args.model, valid_ppl, ep)
        file_name = os.path.join(destination, file_name)

        logger.info('Writing to {}'.format(file_name))
        torch.save(checkpoint, file_name)

        # check the save directory here
        existing_save_files = checkpoint_paths(destination)
        for save_file in existing_save_files[self.args.keep_save_files:]:
            logger.info('Deleting old save file {}'.format(save_file))
            os.remove(save_file)

    def load_checkpoint(self, checkpoint, for_training=False, reset_optim=False):
        self.load_args(checkpoint['args'])
        self.load_state_dict(checkpoint)

        if for_training:
            train_data = self.load_data(checkpoint['args'])
            train_data.load_state_dict(checkpoint, reset_optim)
            return train_data
        else:
            model = self._build_model(checkpoint['args'])
            model.load_state_dict(checkpoint['train_data']['model'])
            return model

    def state_dict(self):
        return {}

    def load_args(self, args):
        pass

    def load_state_dict(self, state_dict):
        pass


def checkpoint_paths(path, pattern=r'(.*)_ppl_(\d+\.\d+)\_e(\d+\.\d+)\.pt'):
    """Retrieves all checkpoints found in `path` directory.
    Checkpoints are identified by matching filename to the specified pattern. If
    the pattern contains groups, the result will be sorted by the second group in
    ascending order.
    """
    pt_regexp = re.compile(pattern)
    files = list(filter(lambda x: x is not None, map(pt_regexp.fullmatch, os.listdir(path))))

    # sort py perplexity (ascending)
    files = sorted(files, key=lambda m: float(m.group(2)))
    return [os.path.join(path, f.group(0)) for f in files]
