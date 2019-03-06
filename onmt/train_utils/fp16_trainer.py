from __future__ import division

import sys, tempfile, os
import onmt
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time, datetime
from collections import defaultdict
import random 
import numpy as np
from onmt.multiprocessing.multiprocessing_wrapper import MultiprocessingRunner
from onmt.ModelConstructor import init_model_parameters
from onmt.train_utils.trainer import BaseTrainer, XETrainer
from onmt.utils import checkpoint_paths
from onmt.Stats import Logger

    

class DynamicLossScaler:

    def __init__(self, init_scale=2**7, scale_factor=2., scale_window=2000):
        self.loss_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self._iter = 0
        self._last_overflow_iter = -1
    
    # we will pass the iter of optim to here
    def update_scale(self, overflow):
        
        self._iter += 1
        if overflow:
            self.loss_scale /= self.scale_factor
            self._last_overflow_iter = self._iter
        elif (self._iter - self._last_overflow_iter) % self.scale_window == 0:
            self.loss_scale *= self.scale_factor
        

    @staticmethod
    def has_overflow(grad_norm):
        # detect inf and nan
        if grad_norm == float('inf') or grad_norm != grad_norm:
            return True
        return False


class FP16XETrainer(XETrainer):

    def __init__(self, model, loss_function, train_data, valid_data, dicts, opt):
        super().__init__(model, loss_function, train_data, valid_data, dicts, opt, set_param=False)
        self.optim = onmt.Optim(opt)
        self.scaler = DynamicLossScaler(opt.fp16_loss_scale)
        
        if self.cuda:
           torch.cuda.set_device(self.opt.gpus[0])
           torch.manual_seed(self.opt.seed)

           # Important:
           # Loss function needs to be in fp32
           self.loss_function = self.loss_function.cuda()

        self.logger = Logger(self.optim)
        self.meters = self.logger.meters

    # fp32 utility (gradients and optim)
    def convert_fp32(self, model_state=None, optim_state=None):

        if model_state is not None:
            self.model.load_state_dict(model_state)

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optim.set_parameters(params)

        if optim_state is not None:
            self.optim.load_state_dict(optim_state)

        print(self.optim.optimizer)
        
    
    def convert_fp16(self, model_state=None, optim_state=None):
        
        if model_state is not None:
            self.model.load_state_dict(model_state)

        self.model = self.model.half().cuda()
        params = [p for p in self.model.parameters() if p.requires_grad]
        total_param_size = sum(p.data.numel() for p in params)
        
        self.fp32_params = params[0].new(0).float().new(total_param_size)
        
        # now we transfer the params from fp16 over fp32
        offset = 0
        for p in params:
            numel = p.data.numel()
            self.fp32_params[offset:offset+numel].copy_(p.data.view(-1))
            offset += numel
        
        self.fp32_params = torch.nn.Parameter(self.fp32_params)
        self.fp32_params.grad = self.fp32_params.data.new(total_param_size).zero_()
        # we optimize on the fp32 params
        self.optim.set_parameters([self.fp32_params])
        
        if optim_state is not None:
            self.optim.load_state_dict(optim_state)
        
        print(self.optim.optimizer)
        
    def eval(self, data):
        
        total_loss = 0
        total_words = 0
                
        torch.cuda.empty_cache()
        self.model.eval()
        """ New semantics of PyTorch: not creating gradients in this mode """
        with torch.no_grad():
            for i in range(len(data)):
                
                oom = False
                try:
                    samples = data.next()
                    
                    batch = samples[0]
                    batch.cuda()

                    """ outputs can be either 
                            hidden states from decoder or
                            prob distribution from decoder generator
                    """
                    outputs = self.model(batch)
                    targets = batch.get('target_output')
                    tgt_mask = batch.get('tgt_mask')
                    
                    loss_output = self.loss_function(outputs, targets, generator=self.model.generator,
                                                     tgt_mask=tgt_mask, backward=False)
                    
                    loss_data = loss_output['nll']
                
                    del loss_output
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e) or 'get_temporary_buffer' in str(e) :
                        oom = True
                        torch.cuda.empty_cache()
                    else:
                        raise e        
                
                if not oom :
                    total_loss += loss_data
                    total_words += batch.tgt_size

        self.model.train()
        return total_loss / (total_words + 1e-6)
        
    
        
    def train_epoch(self, epoch, resume=False, batch_order=None, iteration=0):

        self.meters['total_loss'].reset()
        self.meters['total_words'].reset()
        opt = self.opt
        train_data = self.train_data
        
        # Clear the gradients of the model
        self.model.zero_grad()
        self.optim.zero_grad() 

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
        num_accumulated_words = 0
        num_accumulated_sents = 0
        oom_count = 0
        grad_norm = 0
        
        for i in range(iteration, n_samples):

            curriculum = (epoch < opt.curriculum)
            
            samples = train_data.next(curriculum=curriculum)
                        
            oom = False
            try:
                # ~ torch.cuda.empty_cache()
                # ~ batch = self.to_variable(samples[0])
                batch = samples[0]
                batch.cuda()
            
                outputs = self.model(batch)
                    
                targets = batch.get('target_output')
                tgt_inputs = batch.get('target_input')
                
                batch_size = batch.size
                
                tgt_mask = batch.get('tgt_mask')
                tgt_size = batch.tgt_size
                                
                ## Scale UP the loss so that the gradients are not cutoff
                normalizer = 1.0 / self.scaler.loss_scale

                params = defaultdict(lambda: 0.0)
                params['l2'] = self.opt.l2_coeff
                
                loss_output = self.loss_function(outputs, targets, generator=self.model.generator, 
                                                             backward=True, tgt_mask=tgt_mask, normalizer=normalizer,
                                                             params=params)
                
                ## take the negative likelihood                                             
                loss_data = loss_output['nll']

                for key in loss_output:
                    if key in self.meters:
                        self.meters[key].update(loss_output[key], batch.size)
                
                del loss_output['loss']
                del loss_output
                
                
            except RuntimeError as e:
                if 'out of memory' in str(e) or 'get_temporary_buffer' in str(e) :
                    oom = True
                    self.reset_state()
                    torch.cuda.empty_cache()
                    oom_count += 1
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
                normalizer = num_accumulated_words if opt.normalize_gradient else 1
                if num_accumulated_words >= opt.batch_size_update * 0.95:
                    # Update the parameters.
                    
                    # First we have to copy the grads from fp16 to fp32
                    self._get_flat_grads(out=self.fp32_params.grad)
                    
                    
                    normalizer = normalizer * self.scaler.loss_scale 
                    # rescale and clip grads
                    self.fp32_params.grad.data.div_(normalizer)

                    grad_norm = torch.norm(self.fp32_params.grad.data).item()
                    
                    
                    overflow = DynamicLossScaler.has_overflow(grad_norm)
                    self.scaler.update_scale(overflow)


                    
                    if overflow:
                        if self.scaler.loss_scale <= 1e-4:
                            raise Exception((
                                'Minimum loss scale reached ({}). Your loss is probably exploding. '
                                'Try lowering the learning rate, using gradient clipping or '
                                'increasing the batch size.'
                            ).format(1e-4))
                        print('setting loss scale to: ' + str(self.scaler.loss_scale))
                        self.model.zero_grad()
                        self.optim.zero_grad()
                        num_accumulated_words = 0
                        num_accumulated_sents = 0
                        loss_data = 0
                    
                    else:
                        try:
                            self.meters['gnorm'].update(grad_norm)
                            max_norm = 5.0
                            if grad_norm > max_norm > 0:
                                clip_coef = max_norm / (grad_norm + 1e-6)
                                self.fp32_params.grad.data.mul_(clip_coef)

                            self.optim.step(grad_denom=1) # update the parameters in fp32 
                            
                            # copying the parameters back to fp16
                            offset = 0
                            for p in self.model.parameters():
                                if not p.requires_grad:
                                    continue
                                numel = p.data.numel()
                                p.data.copy_(self.fp32_params.data[offset:offset+numel].view_as(p.data))
                                offset += numel
                        except RuntimeError as e:
                            if 'out of memory' in str(e):
                                torch.cuda.empty_cache()
                                oom_count += 1
                            else:
                                raise e

                        self.model.zero_grad()
                        self.optim.zero_grad()
                        counter = 0
                        num_accumulated_words = 0
                        num_accumulated_sents = 0
                        num_updates = self.optim._step
                        if opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every :
                            valid_loss = self.eval(self.valid_data)
                            valid_ppl = math.exp(min(valid_loss, 100))
                            print('Validation perplexity: %g' % valid_ppl)
                            
                            ep = float(epoch) - 1. + ((float(i) + 1.) / n_samples)
                            
                            self.save(ep, valid_ppl, batch_order=batch_order, iteration=i)
                

                num_words = tgt_size
                self.meters['report_loss'].update(loss_data)
                self.meters['report_tgt_words'].update(num_words)
                self.meters['report_src_words'].update(src_size)
                self.meters['total_loss'].update(loss_data)
                self.meters['total_words'].update(num_words)

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
        optim = self.optim
        self.model = self.model
        
        # Try to load the save_file
        checkpoint = None
        if save_file:
            checkpoint = torch.load(save_file)

       	checkpoint_dir = os.path.dirname(opt.save_model)
        existed_save_files = checkpoint_paths(checkpoint_dir)
        print(existed_save_files)

        
        if checkpoint is not None:

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

            print('Loading model and optim from checkpoint at %s' % save_file)
            self.convert_fp16(checkpoint['model'], checkpoint['optim'])

            # self.convert_fp16(checkpoint['model'], checkpoint['optim'])
            # batch_order = checkpoint['batch_order']
            # iteration = checkpoint['iteration'] + 1
            # opt.start_epoch = int(math.floor(float(checkpoint['epoch'] + 1)))
            # resume=True
            #
            del checkpoint['model']
            del checkpoint['optim']
            del checkpoint
            
        else:
            batch_order = None
            iteration = 0
            print('Initializing model parameters')
            init_model_parameters(model, opt)
            resume=False

            if self.opt.fp16:
                self.convert_fp16()
            else:
                self.convert_fp32()
        
        
        
        valid_loss = self.eval(self.valid_data)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g' % valid_ppl)
        #~ 
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
            
            # only save at the end of epoch when the option to save "every iterations" is disabled
            if self.opt.save_every <= 0: 
                self.save(epoch, valid_ppl)
            batch_order = None
            iteration = None
            resume = False
        
        
    
    
    
