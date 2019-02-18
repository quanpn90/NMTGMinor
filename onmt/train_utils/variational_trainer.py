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
from onmt.Stats import Logger
from statistics import mean, stdev

    
    
from onmt.Meters import AverageMeter, TimeMeter

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


class VariationalTrainerFP16(XETrainer):

    def __init__(self, model, loss_function, trainData, validData, dicts, opt):
        super().__init__(model, loss_function, trainData, validData, dicts, opt, set_param=False)
        self.optim = onmt.Optim(opt)
        self.scaler = DynamicLossScaler(opt.fp16_loss_scale, scale_window=2000)
        self.n_samples = 1
        
        if self.cuda:
           torch.cuda.set_device(self.opt.gpus[0])
           torch.manual_seed(self.opt.seed)
           
           #~ print(torch.cuda.get_device_capability(0)[0])
           
           # Important:
           # Loss function needs to be in fp32
           self.loss_function = self.loss_function.cuda()
           self.model = self.model.cuda()
        
        # prepare some meters
        self.meters = dict()
        self.meters["total_loss"] = AverageMeter()
        self.meters["total_words"] = AverageMeter()
        self.meters["report_loss"] = AverageMeter()
        self.meters["report_tgt_words"] = AverageMeter()
        self.meters["report_src_words"] = AverageMeter()
        self.meters["kl"] = AverageMeter()
        self.meters["kl_prior"] = AverageMeter()
        self.meters["gnorm"] = AverageMeter()
        self.meters["oom"] = AverageMeter() 
        self.meters["total_sloss"] = AverageMeter()
        self.meters["baseline"] = AverageMeter()
        self.meters["R"] = AverageMeter()
        self.meters["ce"] = AverageMeter()
        self.meters["q_entropy"] = AverageMeter()
        self.meters["q_mean"] = AverageMeter()
        self.meters["q_var"] = AverageMeter()

        self.logger = Logger(self.optim, self.meters, scaler=self.scaler)
    
    # fp16 utility
    def convert_fp16(self, model_state=None, optim_state=None):
        
        if model_state is not None:
            self.model.load_state_dict(model_state)

        self.model = self.model.half()
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


    # fp32 utility (gradients and optim)
    def convert_fp32(self, model_state=None, optim_state=None):

        if model_state is not None:
            self.model.load_state_dict(model_state)

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optim.set_parameters(params)

        if optim_state is not None:
            self.optim.load_state_dict(optim_state)

        print(self.optim.optimizer)


    def eval(self, data):
        
        total_loss = 0
        total_words = 0
        total_batch_size = 0
        total_p_entropy = 0
        total_kl = 0
                
        batch_order = data.create_order(random=False)
        torch.cuda.empty_cache()
        self.model.eval()

        ppls = dict()
        total_losses = dict()
        """ New semantics of PyTorch: not creating gradients in this mode """

        with torch.no_grad():
            for n in range(self.n_samples):
                total_losses[n] = 0
                for i in range(len(data)):
                    
                    samples = data.next()
                        
                    batch = samples[0]
                    batch.cuda()
                    
                    
                    """ outputs can be either 
                            hidden states from decoder or
                            prob distribution from decoder generator
                        during Evaluation we sample from the prior distribution
                        and don't use sampling
                    """


                    outputs = self.model(batch, dist='prior', sampling=False)
                    targets = batch.get('target_output')
                    
                    loss_output = self.loss_function(outputs, targets, generator=self.model.generator, backward=False)
                    
                    loss_data = loss_output['nll']
                    kl = loss_output['kl']

                    if n == 0:
                        total_p_entropy += loss_output['p_entropy']
                        total_batch_size += batch.size
                        total_words += batch.tgt_size
                        total_kl += kl
                
                    del loss_output
                    total_losses[n] += loss_data


                    

        p_entropy = total_p_entropy / (total_batch_size)
        kl = total_kl / (total_batch_size)
        print("Prior entropy: %.3f " % p_entropy)
        print("Validation KL divergence: %.3f " % kl)
        self.model.train()

        # take the average
        total_loss = sum(total_losses.values()) / float(len(total_losses))
        ppls = list()


        if self.n_samples > 1:
            for k in total_losses:
                valid_loss = total_losses[k] / (total_words + 1e-6)
                ppl = math.exp(min(valid_loss, 100))
                print("perplexity for sampling %d : %.3f " % (k, ppl))
                ppls.append(ppl)
            mean_ = mean(ppls)
            std_ = stdev(ppls)
            print("Mean and std: %.3f, %.3f" % (mean_, std_))
            
        else:
            ppl = total_loss / (total_words + 1e-6)
            ppl = math.exp(min(ppl, 100))
            print("Perplexity Using Prior Mean : %.3f " % (ppl))

        return total_loss / (total_words + 1e-6)
        
    
        
    def train_epoch(self, epoch, resume=False, batchOrder=None, iteration=0):
        
        opt = self.opt
        trainData = self.trainData
        
        # Clear the gradients of the model
        self.model.zero_grad()
        self.optim.zero_grad() 

        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # Shuffle mini batch order.
        if resume:
            trainData.batchOrder = batchOrder
            trainData.set_index(iteration)
            print("Resuming from iteration: %d" % iteration)
        else:
            batchOrder = trainData.create_order()
            iteration = 0


        self.logger.reset()
        nSamples = len(trainData)
        
        counter = 0
        num_accumulated_words = 0
        num_accumulated_sents = 0
        
        for i in range(iteration, nSamples):

            curriculum = (epoch < opt.curriculum)
            
            samples = trainData.next(curriculum=curriculum)
                        
            oom = False
            try:
                # ~ torch.cuda.empty_cache()
                # ~ batch = self.to_variable(samples[0])
                batch = samples[0]
                batch.cuda()
                
                sampling = not opt.var_not_sampling
                dist = opt.var_sample_from
                outputs = self.model(batch, dist=dist, sampling=sampling)
                    
                targets = batch.get('target_output')
                tgt_inputs = batch.get('target_input')
                
                batch_size = batch.size
                
                tgt_mask = batch.get('tgt_mask')
                tgt_size = batch.tgt_size
                                
                ## Scale UP the loss so that the gradients are not cutoff

                if self.opt.fp16:
                    normalizer = 1.0 / self.scaler.loss_scale 
                else:
                    normalizer = 1.0
                
                warmup_steps = self.optim.warmup_steps
                # alpha = 1 / (warmup_steps * (warmup_steps ** -1.5))


                # from bowman et al, 2016:
                # gradually increasing the coefficient for kl divergence loss
                # so that the model does not ignore the latent variable later on
                # (actually current unused in the loss function) to be implemented
                if opt.var_annealing_kl:
                    max_steps = self.optim.warmup_steps * 2
                    min_steps = self.optim.warmup_steps / 2

                    
                    alpha = max(min((self.optim._step - min_steps) / max_steps, 1.0), 0.0)
                    kl_lambda = alpha * opt.var_kl_lambda
                    # else:
                    #     kl_lambda = alpha * opt.var_kl_lambda
                else:
                    kl_lambda = opt.var_kl_lambda   

                loss_output = self.loss_function(outputs, targets, generator=self.model.generator, 
                                                             backward=True, tgt_mask=tgt_mask, normalizer=normalizer,
                                                             kl_lambda=kl_lambda)
                
                ## take the negative likelihood                                             
                loss_data = loss_output['nll']
                kl = loss_output['kl']
                kl_prior = loss_output['kl_prior']
                baseline = loss_output['baseline']
                R = loss_output['R']
                ce = loss_output['ce']
                q_entropy = loss_output['q_entropy']
                q_z = outputs['q_z']


                
                del loss_output['loss']
                del loss_output
                
                
            except RuntimeError as e:
                if 'out of memory' in str(e) or 'get_temporary_buffer' in str(e) :
                    oom = True
                    self.reset_state()
                    torch.cuda.empty_cache()
                    self.meters['oom'].update(1)
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
                if num_accumulated_words >= opt.batch_size_update :
                    # Update the parameters.
                    
                    if self.opt.fp16:

                        # First we have to copy the grads from fp16 to fp32
                        self._get_flat_grads(out=self.fp32_params.grad)
                        
                        normalizer = normalizer * self.scaler.loss_scale 
                        # rescale and clip grads
                        self.fp32_params.grad.data.div_(normalizer)

                        grad_norm = torch.norm(self.fp32_params.grad.data).item()
                        
                        
                        
                        overflow = DynamicLossScaler.has_overflow(grad_norm)
                        self.scaler.update_scale(overflow)
                    else:
                        overflow = False

                    
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
                        grad_norm = 0
                        self.meters['gnorm'].reset()
                    
                    else:
                        try:
                            # max_norm = self.opt.max_grad_norm 
                            # if grad_norm > max_norm > 0:
                            #     clip_coef = max_norm / (grad_norm + 1e-6)
                            #     self.fp32_params.grad.data.mul_(clip_coef)

                            if self.opt.fp16:
                                grad_denom = 1
                            else:
                                grad_denom = normalizer
                            grad_norm = self.optim.step(grad_denom=grad_denom) # update the parameters in fp32 
                            self.meters['gnorm'].update(grad_norm)

                            if self.opt.fp16:
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
                                self.meters["oom"].update(1)
                            else:
                                raise e


                        self.model.zero_grad()
                        self.optim.zero_grad()
                        counter = 0
                        num_accumulated_words = 0
                        num_accumulated_sents = 0
                        num_updates = self.optim._step
                        if opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every :
                            valid_loss = self.eval(self.validData)
                            valid_ppl = math.exp(min(valid_loss, 100))
                            print('Validation perplexity: %g' % valid_ppl)
                            
                            ep = float(epoch) - 1. + ((float(i) + 1.) / nSamples)
                            
                            self.save(ep, valid_ppl, batchOrder=batchOrder, iteration=i)
                

                num_words = tgt_size
                self.meters['report_loss'].update(loss_data)
                self.meters['report_tgt_words'].update(num_words)
                self.meters['report_src_words'].update(src_size)
                self.meters['total_loss'].update(loss_data)
                self.meters['total_words'].update(num_words)
                self.meters['kl'].update(kl, batch_size)
                self.meters['kl_prior'].update(kl_prior, batch_size)
                self.meters['baseline'].update(baseline, batch_size)
                self.meters['R'].update(R, batch_size)
                self.meters['ce'].update(ce, batch_size) # this is sentence level loss
                self.meters['q_entropy'].update(q_entropy, batch_size)

                if isinstance(q_z, (list,)):
                    self.meters['q_mean'].update(q_z[0].loc.sum().item(), batch_size)
                    self.meters['q_var'].update(q_z[0].scale.sum().item(), batch_size)
                else:
                    self.meters['q_mean'].update(q_z.loc.sum().item(), batch_size)
                    self.meters['q_var'].update(q_z.scale.sum().item(), batch_size)
                
                optim = self.optim
                
                if i == 0 or (i % opt.log_interval == -1 % opt.log_interval):

                    data_size = len(trainData)
                    self.logger.log(epoch, i, data_size)
                    
                    # self.logger.reset_meter("report_loss")
                    # self.logger.reset_meter("report_tgt_words")
                    # self.logger.reset_meter("report_src_words")
                    
        
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
        
        if checkpoint is not None:
            print('Loading model and optim from checkpoint at %s' % save_file)
            self.convert_fp16(checkpoint['model'], checkpoint['optim'])
            batchOrder = checkpoint['batchOrder']
            iteration = checkpoint['iteration'] + 1
            opt.start_epoch = int(math.floor(float(checkpoint['epoch'] + 1)))
            resume=True  
            
            del checkpoint['model']
            del checkpoint['optim']
            del checkpoint
            
        else:
            batchOrder = None
            iteration = 0
            print('Initializing model parameters')
            init_model_parameters(model, opt)
            resume=False

            if self.opt.fp16:
                self.convert_fp16()
            else:
                self.convert_fp32()

        if opt.var_load_pretrained:
            print("Loading pretrained from %s " % opt.var_load_pretrained)

            pretrained_cp = torch.load(opt.var_load_pretrained, map_location=lambda storage, loc: storage)

            transformer_model_weights = pretrained_cp['model']
            self.model.load_transformer_weights(transformer_model_weights)
            
            print("Done")
        
        
        valid_loss = self.eval(self.validData)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g' % valid_ppl)
        #~ 
        self.start_time = time.time()
        
        for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
            print('')

            #  (1) train for one epoch on the training set
            train_loss = self.train_epoch(epoch, resume=resume,
                                                 batchOrder=batchOrder,
                                                 iteration=iteration)
            train_ppl = math.exp(min(train_loss, 100))
            print('Train perplexity: %g' % train_ppl)

            #  (2) evaluate on the validation set
            valid_loss = self.eval(self.validData)
            valid_ppl = math.exp(min(valid_loss, 100))
            print('Validation perplexity: %g' % valid_ppl)
            
            # only save at the end of epoch when the option to save "every iterations" is disabled
            if self.opt.save_every <= 0: 
                self.save(epoch, valid_ppl)
            batchOrder = None
            iteration = None
            resume = False
        
        
    
    
    
