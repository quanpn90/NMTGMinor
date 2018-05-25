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

class BaseTrainer(object):
    
    def __init__(self, model, loss_function, trainData, validData, dataset, opt):
        
        self.model = model
        self.trainData = trainData
        self.validData = validData
        self.dicts = dataset['dicts']
        self.dataset = dataset
        self.opt = opt
        self.cuda = (len(opt.gpus) >= 1)
        
        self.loss_function = loss_function
        self.start_time = 0
        
        
        
    def run(self, *args,**kwargs):
        
        raise NotImplementedError    
    
    def eval(self, data):
        
        raise NotImplementedError
        
    def to_variable(self, data):
        
        for i, t in enumerate(data):
            if self.cuda:
                data[i] = Variable(data[i].cuda())
            else:
                data[i] = Variable(data[i])

        return data
            


class XETrainer(BaseTrainer):

    def __init__(self, model, loss_function, trainData, validData, dataset, opt):
        super().__init__(model, loss_function, trainData, validData, dataset, opt)
        self.optim = onmt.Optim(opt)
        
        if self.cuda:
           torch.cuda.set_device(self.opt.gpus[0])
           torch.manual_seed(self.opt.seed)
           self.loss_function = self.loss_function.cuda()
           self.model = self.model.cuda()
        
        self.optim.set_parameters(self.model.parameters())

    def save(self, epoch, valid_ppl, batchOrder=None, iteration=-1):
        
        opt, dataset = self.opt, self.dataset
        model = self.model
        

        model_state_dict = self.model.state_dict()
        optim_state_dict = self.optim.state_dict()
                
        #  drop a checkpoint
        checkpoint = {
                'model': model_state_dict,
                'dicts': dataset['dicts'],
                'opt': opt,
                'epoch': epoch,
                'iteration' : iteration,
                'batchOrder' : batchOrder,
                'optim': optim_state_dict
        }
        
        file_name = '%s_ppl_%.2f_e%.2f.pt' % (opt.save_model, valid_ppl, epoch)
        print('Writing to %s' % file_name)
        torch.save(checkpoint, file_name)
        
    def eval(self, data):
        total_loss = 0
        total_words = 0
                
        batch_order = data.create_order(random=False)
        self.model.eval()
        """ New semantics of PyTorch: save space by not creating gradients """
        with torch.no_grad():
            for i in range(len(data)):
                    
                samples = data.next()
                
                batch = self.to_variable(samples[0])
                
                """ outputs can be either 
                        hidden states from decoder or
                        prob distribution from decoder generator
                """
                outputs = self.model(batch)
                targets = batch[1][1:]
                
                loss_data, grad_outputs = self.loss_function(outputs, targets, generator=self.model.generator, backward=False)
                
#~ 
                total_loss += loss_data
                total_words += targets.data.ne(onmt.Constants.PAD).sum().item()

        self.model.train()
        return total_loss / total_words
        
    def train_epoch(self, epoch, resume=False, batchOrder=None, iteration=0):
        
        opt = self.opt
        trainData = self.trainData
        
        # Clear the gradients of the model
        # self.runner.zero_grad()
        self.model.zero_grad()

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
        # if batchOrder is not None:
            # batchOrder = trainData.create_order()
        # else:
            # trainData.batchOrder = batchOrder
            
        # if iteration is not None and iteration > -1:
            # trainData.set_index(iteration)
            # print("Resuming from iteration: %d" % iteration)

        total_loss, total_words = 0, 0
        report_loss, report_tgt_words = 0, 0
        report_src_words = 0
        start = time.time()
        nSamples = len(trainData)
        dataset = self.dataset
        
        counter = 0
        num_accumulated_words = 0
        num_accumulated_sents = 0
        
        for i in range(iteration, nSamples):

            curriculum = (epoch < opt.curriculum)
            
            samples = trainData.next(curriculum=curriculum)
                        
            batch = self.to_variable(samples[0])
            
            oom = False
            try:
            
                outputs = self.model(batch)
                    
                targets = batch[1][1:]
                tgt_inputs = batch[1][:-1]
                
                batch_size = targets.size(1)
                
                tgt_mask = targets.data.ne(onmt.Constants.PAD)
                tgt_size = tgt_mask.sum()
                
                tgt_mask = torch.autograd.Variable(tgt_mask)
                #~ tgt_mask = None
                normalizer = 1
                
                if self.opt.normalize_gradient:
                    normalizer = tgt_size
                
                loss_data, grad_outputs = self.loss_function(outputs, targets, generator=self.model.generator, 
                                                             backward=True, mask=tgt_mask, normalizer=normalizer)
                
                #~ outputs.backward(grad_outputs)
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory on GPU , skipping batch')
                    oom = True
                    torch.cuda.empty_cache()
                else:
                    raise e        
                
            if not oom:
                src_size = batch[0].data.ne(onmt.Constants.PAD).sum().item()
                tgt_size = targets.data.ne(onmt.Constants.PAD).sum().item()
                
                
                counter = counter + 1 
                num_accumulated_words += tgt_size
                num_accumulated_sents += batch_size
                
                # We only update the parameters after getting gradients from n mini-batches
                # simulating the multi-gpu situation
                #~ if counter == opt.virtual_gpu:
                #~ if counter >= opt.batch_size_update:
                if num_accumulated_words >= opt.batch_size_update * 0.95:
                    # Update the parameters.
                    self.optim.step(grad_denom=1)
                    self.model.zero_grad()
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
                report_loss += loss_data
                report_tgt_words += num_words
                report_src_words += src_size
                total_loss += loss_data
                total_words += num_words
                
                optim = self.optim
                
                
                
                if i == 0 or (i % opt.log_interval == -1 % opt.log_interval):
                    print(("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; lr: %.7f ; num updates: %7d " +
                           "%5.0f src tok/s; %5.0f tgt tok/s; %s elapsed") %
                          (epoch, i+1, len(trainData),
                           math.exp(report_loss / report_tgt_words),
                           optim.getLearningRate(),
                           optim._step,
                           report_src_words/(time.time()-start),
                           report_tgt_words/(time.time()-start),
                           str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))

                    report_loss, report_tgt_words = 0, 0
                    report_src_words = 0
                    start = time.time()
            
            
        return total_loss / total_words
    
    
    
    def run(self, save_file=None):
        
        opt = self.opt
        model = self.model
        dataset = self.dataset
        optim = self.optim
        
        # Try to load the save_file
        checkpoint = None
        if save_file:
            checkpoint = torch.load(save_file)
        
        
        if checkpoint is not None:
            print('Loading model and optim from checkpoint at %s' % save_file)
            self.model.load_state_dict(checkpoint['model'])
            
            if opt.reset_optim == False:
                self.optim.load_state_dict(checkpoint['optim'])
                batchOrder = checkpoint['batchOrder']
                iteration = checkpoint['iteration'] + 1
                opt.start_epoch = int(math.floor(float(checkpoint['epoch'] + 1)))
                resume=True  
            else:
                batchOrder = None
                iteration = 0
                resume=False
                
            
            del checkpoint['model']
            del checkpoint['optim']
            del checkpoint
        else:
            batchOrder = None
            iteration = 0
            print('Initializing model parameters')
            init_model_parameters(model, opt)
            resume=False
        
        
        valid_loss = self.eval(self.validData)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g' % valid_ppl)
        
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
            
            
            self.save(epoch, valid_ppl)
            batchOrder = None
            iteration = None
            resume = False
        
        
    
    
    
