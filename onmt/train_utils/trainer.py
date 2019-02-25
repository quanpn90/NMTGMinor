from __future__ import division

import sys, tempfile
import os, re
import onmt
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time, datetime
import random
import numpy as np
from onmt.multiprocessing.multiprocessing_wrapper import MultiprocessingRunner
from onmt.ModelConstructor import init_model_parameters
from onmt.utils import checkpoint_paths
from onmt.Meters import AverageMeter, TimeMeter
from onmt.Stats import Logger


class BaseTrainer(object):

    def __init__(self, model, loss_function, train_data, valid_data, dicts, opt):

        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.dicts = dicts
        self.opt = opt
        self.cuda = (len(opt.gpus) >= 1)

        self.loss_function = loss_function
        self.start_time = 0
        self.fp16 = opt.fp16

    def run(self, *args, **kwargs):

        raise NotImplementedError

    def eval(self, data):

        raise NotImplementedError

    def _get_grads(self):
        grads = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                raise RuntimeError('Model parameter did not receive gradient: ' + name + '. '
                                                                                         'Use the param in the forward pass or set requires_grad=False.' +
                                   ' If you are using Stochastic model + fp16 - try to increase the number of minibatches' +
                                   ' each update to avoid uninitialized gradients.')
            grads.append(p.grad.data)
        return grads

    def _get_flat_grads(self, out=None):
        grads = self._get_grads()
        if out is None:
            grads_size = sum(g.numel() for g in grads)
            out = grads[0].new(grads_size).zero_()
        offset = 0
        for g in grads:
            numel = g.numel()
            out[offset:offset + numel].copy_(g.view(-1))
            offset += numel
        return out[:offset]

    def reset_state(self):

        from onmt.modules.StaticDropout import StaticDropout
        def reset_state(m):
            if type(m) == StaticDropout:
                m.reset_state()

        self.model.apply(reset_state)

    def _get_grad_norm(self, module):

        total_norm = 0

        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                raise RuntimeError('Model parameter did not receive gradient: ' + name + '. '
                                                                                         'Use the param in the forward pass or set requires_grad=False.' +
                                   ' If you are using Stochastic model + fp16 - try to increase the number of minibatches' +
                                   ' each update to avoid uninitialized gradients.')
            total_norm += p.grad.data.norm(2).item()

        total_norm = total_norm ** 0.5
        return total_norm


class XETrainer(BaseTrainer):

    def __init__(self, model, loss_function, train_data, valid_data, dicts, opt, set_param=True):
        super().__init__(model, loss_function, train_data, valid_data, dicts, opt)
        self.optim = onmt.Optim(opt)

        if self.cuda:
            torch.cuda.set_device(self.opt.gpus[0])
            torch.manual_seed(self.opt.seed)
            self.loss_function = self.loss_function.cuda()
            self.model = self.model.cuda()

        if set_param:
            self.optim.set_parameters(self.model.parameters())

        self.logger = Logger(self.optim)
        self.meters = self.logger.meters

    def save(self, epoch, valid_ppl, batch_order=None, iteration=-1):

        opt = self.opt
        model = self.model
        dicts = self.dicts

        model_state_dict = self.model.state_dict()
        optim_state_dict = self.optim.state_dict()

        #  drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'dicts': dicts,
            'opt': opt,
            'epoch': epoch,
            'iteration': iteration,
            'batch_order': batch_order,
            'optim': optim_state_dict
        }

        file_name = '%s_ppl_%.2f_e%.2f.pt' % (opt.save_model, valid_ppl, epoch)
        print('Writing to %s' % file_name)
        torch.save(checkpoint, file_name)

        # check the save directory here
        checkpoint_dir = os.path.dirname(opt.save_model)
        existed_save_files = checkpoint_paths(checkpoint_dir)
        for save_file in existed_save_files[opt.keep_save_files:]:
            print (" * Deleting old save file %s ...." % save_file)
            os.remove(save_file)

    def eval(self, data):
        total_loss = 0
        total_words = 0

        self.model.eval()
        """ New semantics of PyTorch: save space by not creating gradients """
        with torch.no_grad():
            for i in range(len(data)):
                batch = data.next()[0]
                batch.cuda()

                """ outputs can be either 
                        hidden states from decoder or
                        prob distribution from decoder generator
                """
                outputs = self.model(batch)
                targets = batch.get('target_output')

                loss_output = self.loss_function(outputs, targets, generator=self.model.generator, backward=False)

                loss_data = loss_output['nll']
                # ~
                total_loss += loss_data
                total_words += batch.tgt_size

        self.model.train()
        return total_loss / total_words

    def train_epoch(self, epoch, resume=False, batch_order=None, iteration=0):

        opt = self.opt
        train_data = self.train_data

        # Clear the gradients of the model
        self.model.zero_grad()

        if opt.extra_shuffle and epoch > opt.curriculum:
            train_data.shuffle()

        # Shuffle mini batch order.

        if resume:
            train_data.batch_order = batch_order
            train_data.set_index(iteration)
            print("Resuming from iteration: %d" % iteration)
        else:
            batch_order = train_data.create_order()
            iteration = 0

        total_loss, total_words = 0, 0
        report_loss, report_tgt_words = 0, 0
        report_src_words = 0
        start = time.time()
        n_samples = len(train_data)

        counter = 0
        grad_norm = 0
        num_accumulated_words = 0
        num_accumulated_sents = 0

        for i in range(iteration, n_samples):

            curriculum = (epoch < opt.curriculum)

            samples = train_data.next(curriculum=curriculum)

            oom = False
            try:
                # ~ batch = self.to_variable(samples[0])
                batch = samples[0]
                batch.cuda()

                outputs = self.model(batch)

                targets = batch.get('target_output')

                batch_size = batch.size

                tgt_mask = batch.get('tgt_mask')

                loss_output = self.loss_function(outputs, targets, generator=self.model.generator,
                                                 backward=True, tgt_mask=tgt_mask)

                ## take the negative likelihood
                loss_data = loss_output['nll']
                l2 = loss_output['l2']

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory on GPU , skipping batch')
                    oom = True
                    torch.cuda.empty_cache()
                else:
                    raise e

            if not oom:
                src_size = batch.src_size
                tgt_size = batch.tgt_size

                counter = counter + 1
                num_accumulated_words += tgt_size
                num_accumulated_sents += batch_size

                # We only update the parameters after getting gradients from n mini-batches
                # simulating the multi-gpu situation
                # ~ if counter == opt.virtual_gpu:
                # ~ if counter >= opt.batch_size_update:

                if num_accumulated_words >= opt.batch_size_update * 0.95:
                    grad_denom = 1
                    if self.opt.normalize_gradient:
                        grad_denom = num_accumulated_words
                    # Update the parameters.
                    grad_norm = self.optim.step(grad_denom=grad_denom)
                    self.meters['gnorm'].update(grad_norm)
                    self.model.zero_grad()
                    counter = 0
                    num_accumulated_words = 0
                    num_accumulated_sents = 0
                    num_updates = self.optim._step
                    if opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every:
                        valid_loss = self.eval(self.valid_data)
                        valid_ppl = math.exp(min(valid_loss, 100))
                        print('Validation perplexity: %g' % valid_ppl)

                        ep = float(epoch) - 1. + ((float(i) + 1.) / n_samples)

                        self.save(ep, valid_ppl, batch_order=batch_order, iteration=i)

                num_words = tgt_size
                # report_loss += loss_data
                # report_tgt_words += num_words
                # report_src_words += src_size
                # total_loss += loss_data
                # total_words += num_words
                self.meters['report_loss'].update(loss_data)
                self.meters['report_tgt_words'].update(num_words)
                self.meters['report_src_words'].update(src_size)
                self.meters['total_loss'].update(loss_data)
                self.meters['total_words'].update(num_words)
                self.meters['l2'].update(l2, batch.size)

                optim = self.optim

                if i == 0 or (i % opt.log_interval == -1 % opt.log_interval):

                    data_size = len(train_data)
                    self.logger.log(epoch, i, data_size)
                    self.meters['report_loss'].reset()
                    self.meters['report_tgt_words'].reset()
                    self.logger.reset_time()
                    self.meters['l2'].reset()

        total_loss = self.meters['total_loss'].sum
        total_words = self.meters['total_words'].sum
        return total_loss / total_words

    def run(self, save_file=None):

        opt = self.opt
        model = self.model

        # Try to load the save_file
        checkpoint = None
        if save_file:
            checkpoint = torch.load(save_file, map_location=lambda storage, loc: storage)

        if checkpoint is not None:
            print('Loading model and optim from checkpoint at %s' % save_file)
            self.model.load_state_dict(checkpoint['model'])

            if not opt.reset_optim:

                self.optim.load_state_dict(checkpoint['optim'])
                batch_order = checkpoint['batch_order']
                iteration = checkpoint['iteration'] + 1
                opt.start_epoch = int(math.floor(float(checkpoint['epoch'] + 1)))
                resume = True
            else:
                batch_order = None
                iteration = 0
                resume = False

            del checkpoint['model']
            del checkpoint['optim']
            del checkpoint
        else:
            batch_order = None
            iteration = 0
            print('Initializing model parameters')
            init_model_parameters(model, opt)
            resume = False

        valid_loss = self.eval(self.valid_data)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g' % valid_ppl)

        self.start_time = time.time()

        for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
            print('')

            #  (1) train for one epoch on the training set
            train_loss = self.train_epoch(epoch, resume=resume,
                                          batch_order=batch_order,
                                          iteration=iteration)
            train_ppl = math.exp(min(train_loss, 100))
            print('Train perplexity: %g' % train_ppl)

            #  (2) evaluate on the validation set
            valid_loss = self.eval(self.valid_data)
            valid_ppl = math.exp(min(valid_loss, 100))
            print('Validation perplexity: %g' % valid_ppl)

            self.save(epoch, valid_ppl)
            batch_order = None
            iteration = None
            resume = False





