import datetime
import logging
import math
import os
import re

import torch
import torch.autograd
import torch.utils.data
from torch.utils.data import BatchSampler, SequentialSampler
from tqdm import tqdm

from nmtg import optim, models
from nmtg.data.samplers import StatefulBatchSampler, StatefulRandomSampler
from nmtg.meters import TimeMeter, AverageMeter, StopwatchMeter
from nmtg.models import get_model_type, build_model
from nmtg.optim import lr_scheduler

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer solves a Task using a Model"""

    @classmethod
    def add_inference_options(cls, parser, argv=None):
        parser.add_argument('-data_loader_threads', type=int, default=1,
                            help='Number of threads for data loading')
        parser.add_argument('-batch_size', type=int, default=32,
                            help='Batch size for evaluation')

    @classmethod
    def add_training_options(cls, parser, argv=None):
        cls.add_inference_options(parser)
        parser.add_argument('-update_method',
                            choices=lr_scheduler.get_lr_scheduler_names(),
                            help='Type of update rule to use')
        parser.add_argument('-optimizer',
                            choices=optim.get_optimizer_names(),
                            help='Optimization method')
        parser.add_argument('-model', choices=models.get_model_names(), required=True,
                            help='Model type')
        args, _ = parser.parse_known_args(argv)

        model_class = get_model_type(args.model)
        model_class.add_options(parser)
        optimizer_class = optim.get_optimizer_type(args.optimizer)
        optimizer_class.add_options(parser)
        lr_scheduler_class = optim.lr_scheduler.get_lr_scheduler_type(args.update_method)
        lr_scheduler_class.add_options(parser)

        parser.add_argument('-curriculum', type=int, default=0,
                            help='For this many epochs, order the minibatches based '
                                 'on source sequence length. Sometimes setting this to 1 will '
                                 'increase convergence speed.')
        parser.add_argument('-save_every', type=int, default=-1,
                            help='Save model after n training steps.')
        parser.add_argument('-log_interval', type=int, default=100,
                            help='Print stats at this interval.')
        parser.add_argument('-epochs', type=int, default=13,
                            help='Number of training epochs')
        parser.add_argument('-max_grad_norm', type=float, default=0,
                            help="If the norm of the gradient vector exceeds this, "
                                 "renormalize it to have the norm equal to max_grad_norm")
        parser.add_argument('-save_model', default='model',
                            help='Model filename (the model will be saved as '
                                 '<save_model>_epochN_PPL.pt where PPL is the '
                                 'validation perplexity')
        parser.add_argument('-keep_save_files', type=int, default=5,
                            help='Keep this many save files.')
        parser.add_argument('-memory_efficient_fp16', action='store_true',
                            help='Use alternative, more memory efficient implementation of fp16 training.')

    def __init__(self, args, for_training=True, checkpoint=None):
        self.args = args
        self.for_training = for_training
        if not for_training and checkpoint is None:
            raise ValueError('When setting up for inference, a trained model is required')

        if checkpoint is not None:
            self.upgrade_checkpoint(checkpoint)
            self._load_data(checkpoint)
        else:
            self._build_data()

        self._build_model(checkpoint['args'] if not for_training else args)

        if args.cuda:
            self.model.cuda()
        if args.fp16:
            self.model.half()

        if for_training:
            self._build_optimizer()

        if checkpoint is not None:
            self.model.load_state_dict(checkpoint['model'])

    def _build_data(self):
        self.training_steps = 0
        self.training_time = TimeMeter()

    def _load_data(self, checkpoint):
        self.training_steps = checkpoint['num_updates']
        self.training_time = TimeMeter()
        self.training_time.reset(checkpoint['training_time'], self.training_steps)

    def _save_data(self, checkpoint):
        checkpoint['num_updates'] = self.training_steps
        checkpoint['training_time'] = self.training_time.elapsed_time

    def _build_model(self, model_args):
        self.model = build_model(self.args.model, model_args)

    def _build_optimizer(self):
        assert all(x.requires_grad for x in self.model.parameters())
        logger.info('{:,d} parameters to train'.format(sum(p.numel() for p in self.model.parameters())))
        self.optimizer = optim.build_optimizer(self.args.optimizer, self.args, self.model.parameters())
        self.scheduler = lr_scheduler.build_lr_scheduler(self.args.update_method, self.args, self.optimizer)

    def _load_optimizer_state_dict(self, checkpoint):
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['lr_scheduler'])

    def _save_optimizer_sate_dict(self, checkpoint):
        checkpoint['optimizer'] = self.optimizer.state_dict()
        checkpoint['lr_scheduler'] = self.scheduler.state_dict()

    def _get_iterator(self, dataset, sampler):
        return dataset.get_iterator(batch_sampler=sampler,
                                    num_workers=self.args.data_loader_threads,
                                    cuda=self.args.cuda)

    def evaluate(self, eval_task):
        eval_dataset = self._get_eval_dataset(eval_task)
        eval_sampler = self._get_eval_sampler(eval_dataset)
        eval_iterator = self._get_iterator(eval_dataset, eval_sampler)

        self.model.eval()
        metrics = self._get_eval_metrics()

        with torch.no_grad():
            for batch in tqdm(eval_iterator, desc='evaluation', disable=self.args.no_progress):
                self._eval_pass(eval_task, batch, metrics)
        return metrics

    def _get_eval_dataset(self, task):
        raise NotImplementedError

    def _get_eval_sampler(self, dataset):
        return BatchSampler(SequentialSampler(dataset), self.args.batch_size, False)

    def _get_eval_metrics(self):
        return {}

    def format_eval_metrics(self, metrics):
        return []

    def _eval_pass(self, task, batch, metrics):
        raise NotImplementedError

    def solve(self, test_task):
        raise NotImplementedError

    def _get_train_dataset(self):
        raise NotImplementedError

    def _get_train_sampler(self, dataset):
        return StatefulBatchSampler(StatefulRandomSampler(dataset), self.args.batch_size, self.args.curriculum == 0)

    def _forward_backward_pass(self, batch, metrics):
        raise NotImplementedError

    def train(self, eval_task=None, checkpoint=None):
        train_dataset = self._get_train_dataset()
        train_sampler = self._get_train_sampler(train_dataset)

        if checkpoint is not None:
            epoch = checkpoint['epoch']
            train_sampler.load_state_dict(checkpoint['sampler'])
            self._load_optimizer_state_dict(checkpoint)
        else:
            epoch = 1

        metrics = self._get_training_metrics()

        if eval_task is not None:
            eval_metrics = self.evaluate(eval_task)
            logger.info(' | '.join(self.format_eval_metrics(eval_metrics)))
            # test_results = self.solve(eval_task)
            # test_metrics = eval_task.score_results(test_results)
            # logger.info(' | '.join(test_metrics))

        self.training_time.start()

        for _ in range(epoch, self.args.epochs + 1):
            self._train_epoch(epoch, train_dataset, train_sampler, metrics, eval_task)
            epoch += 1

        if eval_task is not None:
            eval_metrics = self.evaluate(eval_task)
            logger.info(' | '.join(self.format_eval_metrics(eval_metrics)))
            test_results = self.solve(eval_task)
            test_metrics = eval_task.score_results(test_results)
            logger.info(' | '.join(test_metrics))
        else:
            eval_metrics = {}
        self._save_checkpoint(epoch, train_sampler, eval_metrics.get('nll', None))

    def _train_epoch(self, epoch, dataset, sampler, metrics, eval_task=None):
        self.model.train()
        iterator = self._get_iterator(dataset, sampler)

        metrics['it_wall'].start()
        with tqdm(total=len(sampler), initial=sampler.index + 1, disable=self.args.no_progress) as pbar:
            for index, batch in enumerate(iterator, sampler.index + 1):
                metrics['fwbw_wall'].start()
                try:
                    self._forward_backward_pass(batch, metrics)
                except RuntimeError as e:
                    if 'out of memory' in str(e) or 'get_temporary_buffer' in str(e):
                        logger.warning('Ran out of memory in step {}'.format(index))
                        logger.warning(str(e))
                        self._deal_with_oom(metrics)
                        del batch
                        if self.args.cuda:
                            torch.cuda.empty_cache()
                        metrics['oom'].update()
                        pbar.update()
                        continue
                    else:
                        raise e
                metrics['fwbw_wall'].stop()

                if self._do_training_step(metrics, batch):
                    metrics['train_wall'].start()
                    try:
                        self.training_steps += 1
                        self._learning_step(metrics)
                    except OverflowError as e:
                        logger.warning('Overflow detected ' + str(e))
                        self._deal_with_oom(metrics)
                    except ValueError:
                        logger.critical('NaN loss detected in step {}'.format(index))
                        breakpoint()
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            logger.warning('Ran out of memory in step {}'.format(sampler.index))
                            self._deal_with_oom(metrics)
                            if self.args.cuda:
                                torch.cuda.empty_cache()
                            metrics['oom'].update()
                        else:
                            raise e
                    metrics['train_wall'].stop()

                    if self.args.save_every > 0 and (self.training_steps + 1) % self.args.save_every == 0:
                        if eval_task is not None:
                            eval_metrics = self.evaluate(eval_task)
                            logger.info(' | '.join(self.format_eval_metrics(eval_metrics)))
                            test_results = self.solve(eval_task)
                            test_metrics = eval_task.score_results(test_results)
                            logger.info(' | '.join(test_metrics))
                            self.model.train()
                        else:
                            eval_metrics = {}
                        self._save_checkpoint(epoch, sampler, eval_metrics.get('nll', None))

                formatted = self._format_train_metrics(metrics)
                formatted = ['Epoch {:2d}'.format(epoch)] + formatted
                pbar.set_postfix_str(' | '.join(formatted))
                if index == 0 or (index + 1) % self.args.log_interval == 0:
                    elapsed_time = self.training_time.elapsed_time
                    elapsed_time = str(datetime.timedelta(seconds=elapsed_time)).split('.')[0]
                    log_metrics = ['{:5d}/{:5d}'.format(index + 1, len(iterator)),
                                   '{} elapsed'.format(elapsed_time)]
                    log_str = " | ".join(log_metrics + formatted)
                    logger.info(log_str)
                    self._reset_training_metrics(metrics)

                pbar.update()
                metrics['it_wall'].update()

        sampler.shuffle = epoch >= self.args.curriculum
        sampler.reset()
        self._reset_training_metrics(metrics)

    def _do_training_step(self, metrics, batch):
        return True

    def _learning_step(self, metrics):
        grad_norm = self.optimizer.clip_grad_norm(self.args.max_grad_norm)

        if math.isnan(grad_norm):
            raise ValueError('NaN gradient norm encountered')

        self.optimizer.step()
        metrics['gnorm'].update(grad_norm)
        self.scheduler.step_update(self.training_steps)
        self.optimizer.zero_grad()

    def _get_training_metrics(self):
        meters = {
            'gnorm': AverageMeter(),
            'oom': AverageMeter(),
            'fwbw_wall': StopwatchMeter(),
            'train_wall': StopwatchMeter(),
            'it_wall': TimeMeter()}
        return meters

    def _format_train_metrics(self, metrics):
        formatted = [
            'Updates {:5d}'.format(self.training_steps),
            'lr {:.4e}'.format(self.optimizer.get_lr()),
            'gnorm {:.2f}'.format(metrics['gnorm'].val),
            'ooms {:d}'.format(metrics['oom'].sum),
            'fw/bw {:.0f}ms'.format(metrics['fwbw_wall'].avg * 1000),
            'train {:.0f}ms'.format(metrics['train_wall'].avg * 1000)]
        return formatted

    def _reset_training_metrics(self, metrics):
        metrics['fwbw_wall'].reset()
        metrics['train_wall'].reset()
        metrics['gnorm'].reset()
        metrics['it_wall'].reset()
        metrics['it_wall'].start()

    def _get_checkpoint(self, epoch, train_sampler):
        checkpoint = {'sampler': train_sampler.state_dict(),
                      'epoch': epoch,
                      'model': self.model.state_dict(),
                      'args': self.args}
        self._save_data(checkpoint)
        self._save_optimizer_sate_dict(checkpoint)
        return checkpoint

    def _deal_with_oom(self, metrics):
        self.optimizer.zero_grad()

    def _save_checkpoint(self, epoch, train_sampler, valid_nll=None, destination=None):
        logger.info('Saving checkpoint')
        if destination is None:
            destination = self.args.save_model
        os.makedirs(destination, exist_ok=True)
        checkpoint = self._get_checkpoint(epoch, train_sampler)

        if valid_nll is None:
            valid_ppl = 1.0
        else:
            valid_ppl = math.exp(valid_nll.avg)
        ep = float(epoch) - 1. + ((float(train_sampler.index) + 1.) / len(train_sampler))

        file_name = '{}_ppl_{:.2f}_e{:.2f}.pt'.format(self.args.model, valid_ppl, ep)
        file_name = os.path.join(destination, file_name)

        logger.info('Writing to {}'.format(file_name))
        torch.save(checkpoint, file_name)

        # check the save directory here
        existing_save_files = checkpoint_paths(destination)
        for save_file in existing_save_files[self.args.keep_save_files:]:
            logger.info('Deleting old save file {}'.format(save_file))
            os.remove(save_file)

    @classmethod
    def upgrade_checkpoint(cls, checkpoint):
        """Update the checkpoint to the newest version"""
        args = checkpoint['args']
        model_class = get_model_type(args.model)
        model_class.upgrade_args(args)


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
