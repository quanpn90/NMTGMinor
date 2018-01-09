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

class BaseTrainer(object):
    
    def __init__(self, model, loss_function, trainData, validData, dataset, optim, opt):
        
        self.model = model
        self.trainData = trainData
        self.validData = validData
        self.dicts = dataset['dicts']
        self.dataset = dataset
        self.optim = optim 
        self.opt = opt
        
        self.loss_function = loss_function
        self.start_time = 0
        
    def run(self, *args,**kwargs):
        
        raise NotImplementedError    
    
    def eval(self, data):
        
        raise NotImplementedError
        
       

class XETrainer(BaseTrainer):

    #~ def __init__(self, model, loss_function, trainSets, validSets, dataset, optim, opt):
        
        #~ super(BaseTrainer, self).__init__(model, loss_function, trainSets, validSets, dataset, optim,opt)
        #~ BaseTrainer.__init__(self, model, loss_function, trainSets, validSets, dataset, optim, opt)
    def save(self, epoch, valid_ppl, batchOrder=None, iteration=-1):
        
        model, opt, dataset = self.model, self.opt, self.dataset
        optim = self.optim
        
        model_state_dict = (model.module.state_dict() if len(opt.gpus) > 1
                else model.state_dict())
                
        #  drop a checkpoint
        checkpoint = {
                'model': model_state_dict,
                'dicts': dataset['dicts'],
                'opt': opt,
                'epoch': epoch,
                'iteration' : iteration,
                'batchOrder' : batchOrder,
                'optim': optim
        }
        
        file_name = '%s_ppl_%.2f_e%.2f.pt' % (opt.save_model, valid_ppl, epoch)
        print('Writing to %s' % file_name)
        torch.save(checkpoint, file_name)
        
    def eval(self, data):
        total_loss = 0
        total_words = 0
        
        self.model.eval()
        """ New semantics of PyTorch: save space by not creating gradients """
        with torch.no_grad():
            for i in range(len(data)):
                batch = data[i]
                
                """ outputs can be either 
                        hidden states from decoder or
                        prob distribution from decoder generator
                """
                outputs = self.model(batch)
                # exclude <s> from targets
                targets = batch[1][1:]
                
                loss_data, _ = self.loss_function(outputs, targets, backward=False)

                total_loss += loss_data
                total_words += targets.data.ne(onmt.Constants.PAD).sum()

        self.model.train()
        return total_loss / total_words
        
    def train_epoch(self, epoch, batchOrder=None):
        
        opt = self.opt
        trainData = self.trainData
        model = self.model
        optim = self.optim
        
        # Clear the gradients of the model
        model.zero_grad()

        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # Shuffle mini batch order.
        if not batchOrder:
            batchOrder = torch.randperm(len(trainData))

        total_loss, total_words = 0, 0
        report_loss, report_tgt_words = 0, 0
        report_src_words = 0
        start = time.time()
        nSamples = len(trainData)
        dataset = self.dataset
        
        counter = 0
        
        for i in range(nSamples):

            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            # Exclude original indices.
            batch = trainData[batchIdx]


            hiddens = model(batch)
            # Exclude <s> from targets.
            targets = batch[1][1:]
            
            # Compute loss with the loss function
            loss_data, grad_hiddens = self.loss_function(hiddens, targets, backward=True)
            
            # We only compute the gradients of loss w.r.t to the hiddens in the last step
            # So we continue to backward from there
            hiddens.backward(grad_hiddens)
            
            counter = counter + 1
            
            # We only update the parameters after getting gradients from n mini-batches
            # simulating the multi-gpu situation
            if counter == opt.virtual_gpu:
                # Update the parameters.
                optim.step()
                model.zero_grad()
                counter = 0
                if opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every :
                    valid_loss = self.eval(self.validData)
                    valid_ppl = math.exp(min(valid_loss, 100))
                    print('Validation perplexity: %g' % valid_ppl)
                    
                    ep = float(epoch) - 1. + ((float(i) + 1.) / nSamples)
                    
                    self.save(ep, valid_ppl, batchOrder=batchOrder, iteration=i)
            

            num_words = targets.data.ne(onmt.Constants.PAD).sum()
            report_loss += loss_data
            report_tgt_words += num_words
            report_src_words += batch[0].data.ne(onmt.Constants.PAD).sum()
            total_loss += loss_data
            total_words += num_words
            
            num_updates = optim._step
            
            
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
    
    
    
    def run(self):
        valid_loss = self.eval(self.validData)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g' % valid_ppl)
        
        opt = self.opt
        model = self.model
        dataset = self.dataset
        optim = self.optim
        
        self.start_time = time.time()
        
        for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
            print('')

            #  (1) train for one epoch on the training set
            train_loss = self.train_epoch(epoch)
            train_ppl = math.exp(min(train_loss, 100))
            print('Train perplexity: %g' % train_ppl)

            #  (2) evaluate on the validation set
            valid_loss = self.eval(self.validData)
            valid_ppl = math.exp(min(valid_loss, 100))
            print('Validation perplexity: %g' % valid_ppl)
            
            
            self.save(epoch, valid_ppl)
            

        
        
        
    
    
    
