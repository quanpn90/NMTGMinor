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
from onmt.train_utils.trainer import BaseTrainer



class AETrainer(BaseTrainer):

    def __init__(self, autoencoder, model, loss_function, trainData, validData, dicts, opt):
        super().__init__(model, loss_function, trainData, validData, dicts, opt)
        self.optim = onmt.Optim(opt)
        self.autoencoder = autoencoder
        if(opt.auto_encoder_type is None):
            self.auto_encoder_type = "Baseline"
        else:
            self.auto_encoder_type = opt.auto_encoder_type

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

                batch = data.next()[0]
                if (self.cuda):
                    batch.cuda()

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
        train_data = self.train_data

        # Clear the gradients of the model
        # self.runner.zero_grad()
        self.autoencoder.zero_grad()
        self.model.eval()

        if opt.extra_shuffle and epoch > opt.curriculum:
            train_data.shuffle()

        # Shuffle mini batch order.

        if resume:
            train_data.batchOrder = batchOrder
            train_data.set_index(iteration)
            print("Resuming from iteration: %d" % iteration)
        else:
            batchOrder = train_data.create_order()
            iteration = 0

        total_loss, total_words = 0, 0
        report_loss, report_mu,report_sig,report_el,report_mse, report_kl, report_tgt_words = 0, 0,0,0,0,0,0
        report_src_words = 0
        start = time.time()
        nSamples = len(train_data)

        counter = 0
        num_accumulated_words = 0
        num_accumulated_sents = 0

        for i in range(iteration, nSamples):

            curriculum = (epoch < opt.curriculum)

            batch = train_data.next(curriculum=curriculum)[0]
            if (self.cuda):
                batch.cuda()

            oom = False
            try:

                batch_size = batch.size

                # print("Input size:",batch[0].size())
                targets,outputs = self.autoencoder(batch)

                loss_data= self.loss_function(outputs, targets.data)
                if(self.auto_encoder_type == "Variational"):
                    m = self.autoencoder.variational_layer.mean
                    std = self.autoencoder.variational_layer.std
                    m = m.mul(m)
                    one = torch.ones(m.size())
                    if(m.is_cuda):
                        one = one.cuda()
                    var_loss = ((m+std-std.log()-one)*0.5).sum()
                    report_mse += loss_data.item()
                    report_kl += var_loss.item()
                    report_mu += m.sum().item()
                    report_sig += std.sum().item()
                    report_el += m.numel()
                    loss_data = loss_data + var_loss
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
                    self.model.zero_grad()
                    counter = 0
                    num_accumulated_words = 0
                    num_accumulated_sents = 0
                    num_updates = self.optim._step

                    if opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every:
                        valid_loss = self.eval(self.valid_data)
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
                    print(("Epoch %2d, %5d/%5d; ; loss: %6.2f (%6.2f, %6.2f) ; var: mu %6.2f sig: %6.2f; lr: %.7f ; num updates: %7d " +
                           "%5.0f src tok/s; %s elapsed") %
                          (epoch, i + 1, len(train_data),
                           report_loss / report_tgt_words,report_mse/report_tgt_words,report_kl/report_tgt_words,
                           report_mu / max(1,report_el), report_sig / max(1,report_el),
                           optim.getLearningRate(),
                           optim._step,
                           report_tgt_words / (time.time() - start),
                           str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))
                    report_loss, report_tgt_words ,report_mse,report_kl= 0, 0, 0,0
                    report_mu,report_sig,report_el = 0,0,0
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
        autoencoder.init_model_parameters()
        resume = False

        valid_loss = self.eval(self.valid_data)
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
            valid_loss = self.eval(self.valid_data)
            print('Validation perplexity: %g' % valid_loss)

            self.save(epoch, valid_loss)
            batchOrder = None
            iteration = None
            resume = False





