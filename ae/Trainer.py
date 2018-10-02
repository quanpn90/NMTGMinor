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
from onmt.train_utils.trainer import BaseTrainer



class AETrainer(BaseTrainer):

    def __init__(self, autoencoder, model, loss_function, trainData, validData, dicts, opt):
        super().__init__(model, loss_function, trainData, validData, dicts, opt)
        self.optim = onmt.Optim(opt)
        self.autoencoder = autoencoder

        if self.cuda:
            torch.cuda.set_device(self.opt.gpus[0])
            torch.manual_seed(self.opt.seed)
            self.loss_function = self.loss_function.cuda()
            self.model = self.model.cuda()
            self.autoencoder = self.autoencoder.cuda()

        self.optim.set_parameters(self.autoencoder.parameters())

    def save(self, epoch, valid_ppl, batchOrder=None, iteration=-1):

        opt = self.opt
        autoencoder = self.autoencoder

        autoencoder_state_dict = self.autoencoder.state_dict()
        optim_state_dict = self.optim.state_dict()

        #  drop a checkpoint
        checkpoint = {
            'autoencoder': autoencoder_state_dict,
            'opt': opt,
            'epoch': epoch,
            'iteration': iteration,
            'batchOrder': batchOrder,
            'optim': optim_state_dict
        }

        file_name = '%s_ppl_%.2f_e%.2f.pt' % (opt.save_model, valid_ppl, epoch)
        print('Writing to %s' % file_name)
        torch.save(checkpoint, file_name)

        # check te save directory here

    def eval(self, data):
        total_loss = 0
        total_words = 0

        batch_order = data.create_order(random=False)
        self.model.eval()
        self.autoencoder.eval()
        """ New semantics of PyTorch: save space by not creating gradients """
        with torch.no_grad():
            for i in range(len(data)):
                samples = data.next()

                batch = self.to_variable(samples[0])

                """ outputs can be either 
                        hidden states from decoder or
                        prob distribution from decoder generator
                """

                targets,outputs = self.autoencoder(batch)

                loss_data = self.loss_function(outputs, targets)

                # ~
                total_loss += loss_data
                total_words += outputs.size(0)

        self.autoencoder.train()
        return total_loss / total_words

    def train_epoch(self, epoch, resume=False, batchOrder=None, iteration=0):

        opt = self.opt
        trainData = self.trainData

        # Clear the gradients of the model
        # self.runner.zero_grad()
        self.autoencoder.zero_grad()
        self.model.eval()

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
        num_accumulated_words = 0
        num_accumulated_sents = 0

        for i in range(iteration, nSamples):

            curriculum = (epoch < opt.curriculum)

            samples = trainData.next(curriculum=curriculum)

            batch = self.to_variable(samples[0])

            oom = False
            try:

                batch_size = batch[1][1:].size(1)

                # print("Input size:",batch[0].size())
                targets,outputs = self.autoencoder(batch)

                loss_data= self.loss_function(outputs, targets.data)
                loss_data.backward()


            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory on GPU , skipping batch')
                    oom = True
                    torch.cuda.empty_cache()
                else:
                    raise e

            if not oom:

                counter = counter + 1
                num_accumulated_words += targets.size(0)
                num_accumulated_sents += batch_size


                if num_accumulated_words >= opt.batch_size_update * 0.95:
                    grad_denom = 1
                    if self.opt.normalize_gradient:
                        grad_denom = num_accumulated_words
                    # Update the parameters.
                    self.optim.step(grad_denom=grad_denom)
                    self.autoencoder.zero_grad()
                    counter = 0
                    num_accumulated_words = 0
                    num_accumulated_sents = 0
                    num_updates = self.optim._step
                    if opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every:
                        valid_loss = self.eval(self.validData)
                        print('Validation perplexity: %g' % valid_loss)

                        ep = float(epoch) - 1. + ((float(i) + 1.) / nSamples)

                        self.save(ep, valid_loss, batchOrder=batchOrder, iteration=i)

                num_words = targets.size(0)
                report_loss += loss_data
                report_tgt_words += num_words
                total_loss += loss_data
                total_words += num_words

                optim = self.optim

                if i == 0 or (i % opt.log_interval == -1 % opt.log_interval):
                    print(("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; lr: %.7f ; num updates: %7d " +
                           "%5.0f src tok/s; %s elapsed") %
                          (epoch, i + 1, len(trainData),
                           math.exp(report_loss / report_tgt_words),
                           optim.getLearningRate(),
                           optim._step,
                           report_tgt_words / (time.time() - start),
                           str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))

                    report_loss, report_tgt_words = 0, 0
                    report_src_words = 0
                    start = time.time()

        return total_loss / total_words

    def run(self, save_file=None):

        opt = self.opt
        model = self.model
        autoencoder = self.autoencoder
        optim = self.optim

        # Try to load the save_file
        batchOrder = None
        iteration = 0
        print('Initializing model parameters')
        init_model_parameters(autoencoder, opt)
        resume = False

        valid_loss = self.eval(self.validData)
        print('Validation loss: %g' % valid_loss)

        self.start_time = time.time()

        for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
            print('')

            #  (1) train for one epoch on the training set
            train_loss = self.train_epoch(epoch, resume=resume,
                                          batchOrder=batchOrder,
                                          iteration=iteration)
            print('Train loss: %g' % train_loss)

            #  (2) evaluate on the validation set
            valid_loss = self.eval(self.validData)
            print('Validation perplexity: %g' % valid_loss)

            self.save(epoch, valid_loss)
            batchOrder = None
            iteration = None
            resume = False





