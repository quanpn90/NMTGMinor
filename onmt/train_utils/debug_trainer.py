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
from onmt.train_utils.trainer import BaseTrainer, XETrainer

import torch.nn.functional as F

memory = torch.cuda.memory_cached()

print(memory)

class DebugTrainer(BaseTrainer):

    def __init__(self, model, loss_function, trainData, validData, dicts, opt, set_param=True):
        super().__init__(model, loss_function, trainData, validData, dicts, opt)
        self.optim = onmt.Optim(opt)
        
        if self.cuda:
           torch.cuda.set_device(self.opt.gpus[0])
           torch.manual_seed(self.opt.seed)
           self.loss_function = self.loss_function.cuda()
           self.model = self.model.cuda()
        
        if set_param:
            self.optim.set_parameters(self.model.parameters())

   
        
    def train_epoch(self, epoch, resume=False, batchOrder=None, iteration=0):
        
        opt = self.opt
        trainData = self.trainData
        
        # Clear the gradients of the model
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

        total_loss, total_words = 0, 0
        report_loss, report_tgt_words = 0, 0
        report_src_words = 0
        start = time.time()
        nSamples = len(trainData)
        
        counter = 0
        grad_norm = 0
        num_accumulated_words = 0
        num_accumulated_sents = 0

        torch.cuda.empty_cache()    
        begin_memory = torch.cuda.memory_cached()
        memory_consumed = begin_memory
        memory_consumed = memory_consumed / ( 1024 ** 3)
        

        print("Overhead memory: %.3f GB " % memory_consumed)
        
        for i in range(iteration, nSamples):

            curriculum = (epoch < opt.curriculum)
            
            samples = trainData.next(curriculum=curriculum)

            oom = False
            # try:
            batch = samples[0]
            batch.cuda()
            
            
                    
            targets = batch.get('target_output')
                
            batch_size = batch.size
            src_size = batch.src_size
            tgt_size = batch.tgt_size
            
            tgt_mask = batch.get('tgt_mask')
            tgt_size = batch.tgt_size
                
            normalizer = 1
            

            torch.cuda.empty_cache()    
            begin_memory = torch.cuda.memory_cached()

            outputs = self.model(batch)
            



            # loss_output = self.loss_function(outputs, targets, generator=self.model.generator, 
                                                         # backward=False, tgt_mask=tgt_mask)
            
            # loss_data = loss_output['nll']

            # dist = self.model.generator(outputs['hiddens'])
            hiddens = outputs['hiddens'].transpose(0, 1) # T x B x C -> B x T x C
            dist = F.linear(hiddens, self.model.decoder.word_lut.weight)
            dist = F.log_softmax(dist, dim=-1)

            tensor_size = outputs['hiddens'].size(0) * outputs['hiddens'].size(1)

            after_memory =  torch.cuda.memory_cached()

                            
            memory_consumed = after_memory - begin_memory
            memory_consumed = memory_consumed / ( 1024 ** 3)

            print(("Epoch %2d, %5d/%5d; ; batch size: %d ; batch total size %d; num words: %d; memory_consumed %.3f GB; peak mem %.3f GB;  " +
                       " %s elapsed") %
                      (epoch, i+1, len(trainData),
                       batch.size,
                       tensor_size,
                       tgt_size,
                       memory_consumed,
                       after_memory / ( 1024 ** 3) ,
                       str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))    

            del dist
            
            # except RuntimeError as e:
            #     if 'out of memory' in str(e):
            #         print('| WARNING: ran out of memory on GPU , skipping batch')
            #         oom = True
            #         torch.cuda.empty_cache()
            #     else:
            #         raise e        
                
            # if not oom:
            
                
                
            #     counter = counter + 1 
            #     num_accumulated_words += tgt_size
            #     num_accumulated_sents += batch_size
                
            #     # We only update the parameters after getting gradients from n mini-batches
            #     # simulating the multi-gpu situation
            #     #~ if counter == opt.virtual_gpu:
            #     #~ if counter >= opt.batch_size_update:
                
            #     if num_accumulated_words >= opt.batch_size_update * 0.95:
            #         grad_denom = 1
            #         if self.opt.normalize_gradient:
            #             grad_denom = num_accumulated_words
            #         # Update the parameters.
            #         grad_norm = self.optim.step(grad_denom=grad_denom)
            #         self.model.zero_grad()
            #         counter = 0
            #         num_accumulated_words = 0
            #         num_accumulated_sents = 0
            #         num_updates = self.optim._step
                

            # num_words = tgt_size
            # # report_loss += loss_data
            # report_tgt_words += num_words
            # report_src_words += src_size
            # # total_loss += loss_data
            # total_words += num_words
            
            # optim = self.optim
                
                
                
            # if i == 0 or (i % 10 == 0):
            #     print(("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; lr: %.7f ; num updates: %7d " +
            #            "%5.0f src tok/s; %5.0f tgt tok/s; %s elapsed") %
            #           (epoch, i+1, len(trainData),
            #            math.exp(report_loss / report_tgt_words),
            #            optim.getLearningRate(),
            #            optim._step,
            #            report_src_words/(time.time()-start),
            #            report_tgt_words/(time.time()-start),
            #            str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))    

            #     report_loss, report_tgt_words = 0, 0
            #     report_src_words = 0
            #     start = time.time()

        return 0
        # return total_loss / total_words
    
    
    
    def run(self, save_file=None):
        
        opt = self.opt
        model = self.model
        optim = self.optim
        batchOrder = None
        iteration = 0
        print('Initializing model parameters')
        init_model_parameters(model, opt)
        resume=False
        
       
        
        self.start_time = time.time()

        print('')
        epoch = 0
        train_loss = self.train_epoch(epoch, resume=resume, batchOrder=batchOrder, iteration=iteration)
            
            
        
        
    
    
    
