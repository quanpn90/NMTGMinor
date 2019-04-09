from __future__ import division

import sys, tempfile
import onmt
import onmt.Markdown
import onmt.modules
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
from onmt.train_utils.trainer import BaseTrainer, XETrainer
    

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
        super().__init__(model, loss_function, train_data, valid_data, dicts, opt)
        self.optim = onmt.Optim(opt)
        self.scaler = DynamicLossScaler(opt.fp16_loss_scale)
        self.fp16 = True
        
        if self.cuda:
           torch.cuda.set_device(self.opt.gpus[0])
           torch.manual_seed(self.opt.seed)
           #~ print(torch.cuda.get_device_capability(0)[0])
           
           # Important:
           # Loss function needs to be in fp32
           self.loss_function = self.loss_function.cuda()
           self.model = self.model.cuda().half()
           
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
        self.fp32_params.grad = self.fp32_params.data.new(total_param_size)
        # we optimize on the fp32 params
        self.optim.set_parameters([self.fp32_params])

        
    def eval(self, data):
        total_loss = 0
        total_words = 0
                
        batch_order = data.create_order(random=False)
        self.model.eval()
        """ New semantics of PyTorch: save space by not creating gradients """
        with torch.no_grad():
            for i in range(len(data)):
                    
                batch = data.next()[0]

                batch.cuda(fp16=self.fp16)
                
                """ outputs can be either 
                        hidden states from decoder or
                        prob distribution from decoder generator
                """
                outputs = self.model(batch)
                targets = batch.get('target_output')
                tgt_mask = targets.ne(onmt.Constants.PAD)
                outputs['tgt_mask'] = tgt_mask
                
                # loss_data, grad_outputs = self.loss_function(outputs, targets, generator=self.model.generator, backward=False)
                # if self.opt.ctc_loss != 0:
                #     _, loss_data, grad_outputs = self.loss_function(outputs, encoder, targets,
                #                                                     generator=self.model.generator, backward=False,
                #                                                     source_mask=src_mask, target_mask=tgt_mask)
                # else:
                #     _, loss_data, grad_outputs = self.loss_function(outputs, targets, generator=self.model.generator[0],
                #                                                     backward=False, mask=tgt_mask)

                loss_dict = self.loss_function(outputs, targets, model=self.model,
                                                                 backward=False)
                loss_data = loss_dict['data']
                
                total_loss += loss_data
                total_words += batch.tgt_size

        self.model.train()
        return total_loss / total_words
        
    def train_epoch(self, epoch, resume=False, batch_order=None, iteration=0):
        
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
        nSamples = len(train_data)
        
        counter = 0
        num_accumulated_words = 0
        num_accumulated_sents = 0
        oom_count = 0
        
        for i in range(iteration, nSamples):

            curriculum = (epoch < opt.curriculum)
            
            batch = train_data.next(curriculum=curriculum)[0]
            batch.cuda(fp16=self.fp16)

            oom = False
            try:

                outputs = self.model(batch)
                    
                targets = batch.get('target_output')
                tgt_inputs = batch.get('target_input')
                
                batch_size = batch.size
                
                tgt_mask = targets.data.ne(onmt.Constants.PAD)
                outputs['tgt_mask'] = tgt_mask
                                
                # Scale UP the loss so that the gradients are not cutoff
                normalizer = 1.0 / self.scaler.loss_scale

                # if self.opt.ctc_loss != 0:
                #
                #     _, loss_data, grad_outputs = self.loss_function(outputs, encoder, targets,
                #                                                     generator=self.model.generator, backward=True,
                #                                                     source_mask=src_mask, target_mask=tgt_mask, normalizer=normalizer)
                # else:
                #     _, loss_data, grad_outputs = self.loss_function(outputs, targets, generator=self.model.generator[0],
                #                                                     backward=True, mask=tgt_mask, normalizer=normalizer)
                loss_dict = self.loss_function(outputs, targets, model=self.model,
                                                                 backward=True, normalizer=normalizer)
                loss_data = loss_dict['data']
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    #~ print('| WARNING: ran out of memory on GPU , skipping batch')
                    oom = True
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
                        loss_data = 0
                    
                    else:
                        self.optim.step(grad_denom=1)

                        # re-copy the parameters from fp32 to fp16
                        offset = 0
                        for p in self.model.parameters():
                            if not p.requires_grad:
                                continue
                            numel = p.data.numel()
                            p.data.copy_(self.fp32_params.data[offset:offset+numel].view_as(p.data))
                            offset += numel
                        
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
                            
                            ep = float(epoch) - 1. + ((float(i) + 1.) / nSamples)
                            
                            self.save(ep, valid_ppl, batch_order=batch_order, iteration=i)
                

                num_words = tgt_size
                report_loss += loss_data
                report_tgt_words += num_words
                report_src_words += src_size
                total_loss += loss_data
                total_words += num_words
                
                optim = self.optim

                if i == 0 or (i % opt.log_interval == -1 % opt.log_interval):
                    print(("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; lr: %.7f ; num updates: %7d " +
                           "%5.0f src tok/s; %5.0f tgt tok/s; lscale %0.2f; oom %d; %s elapsed") %
                          (epoch, i+1, len(train_data),
                           math.exp(report_loss / report_tgt_words),
                           optim.getLearningRate(),
                           optim._step,
                           report_src_words/(time.time()-start),
                           report_tgt_words/(time.time()-start),
                           self.scaler.loss_scale,
                           oom_count, 
                           str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))

                    report_loss, report_tgt_words = 0, 0
                    report_src_words = 0
                    start = time.time()
            
            
        return total_loss / total_words
    

    def run(self, save_file=None):
        
        opt = self.opt
        model = self.model
        optim = self.optim
        
        # Try to load the save_file
        checkpoint = None
        if save_file:
            checkpoint = torch.load(save_file)
        
        
        if checkpoint is not None:
            print('Loading model and optim from checkpoint at %s' % save_file)
            self.model.load_state_dict(checkpoint['model'])
            
            if not opt.reset_optim:
                self.optim.load_state_dict(checkpoint['optim'])
                batch_order = checkpoint['batch_order']
                iteration = checkpoint['iteration'] + 1
                opt.start_epoch = int(math.floor(float(checkpoint['epoch'] + 1)))
                resume=True  
            else:
                batch_order = None
                iteration = 0
                resume=False
                
            
            del checkpoint['model']
            del checkpoint['optim']
            del checkpoint
        else:
            batch_order = None
            iteration = 0
            print('Initializing model parameters')
            init_model_parameters(model, opt)
            resume=False
        
        
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
            
            
            self.save(epoch, valid_ppl)
            batch_order = None
            iteration = None
            resume = False
        
        
    
    
    
