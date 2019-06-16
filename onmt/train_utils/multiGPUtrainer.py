from __future__ import division

import onmt
import torch
import math
import time, datetime
from onmt.multiprocessing.multiprocessing_wrapper import MultiprocessingRunner
from onmt.train_utils.trainer import BaseTrainer
from onmt.ModelConstructor import init_model_parameters


class MultiGPUXETrainer(BaseTrainer):

    def __init__(self, model, loss_function, trainData, validData, dataset, opt):
        super().__init__(model, loss_function, trainData, validData, dataset, opt)
        
        print('Initializing model parameters')
        init_model_parameters(model, opt)
        
        # create a multi-gpu runner here 
        self.runner = MultiprocessingRunner(opt, model, loss_function, device_ids=opt.gpus)

    def save(self, epoch, valid_ppl, batchOrder=None, iteration=-1):
        
        opt, dataset = self.opt, self.dataset

        #~ model_state_dict, optim_state_dict = self.runner.state_dict()
                
        #  drop a checkpoint
        checkpoint = {
                #~ 'model': model_state_dict,
                'dicts': dataset['dicts'],
                'opt': opt,
                'epoch': epoch,
                'iteration' : iteration,
                'batchOrder' : batchOrder,
                #~ 'optim': optim_state_dict
        }
        
        file_name = '%s_ppl_%.2f_e%.2f.pt' % (opt.save_model, valid_ppl, epoch)
        print('Writing to %s' % file_name)
        
        self.runner.save_checkpoint(checkpoint, file_name)
        #~ torch.save(checkpoint, file_name)
        
    def eval(self, data):
        total_loss = 0
        total_words = 0
                
        batch_order = data.create_order(random=False)
        """ New semantics of PyTorch: save space by not creating gradients """
        with torch.no_grad():
            for i in range(len(data)):
                    
                samples = data.next()
                
                """ outputs can be either 
                        hidden states from decoder or
                        prob distribution from decoder generator
                """
                
                logging_outputs = self.runner.step(samples, eval=True)
                
                loss_data = logging_outputs['loss']
                ooms = logging_outputs['oom']
                
#~ 
                total_loss += loss_data
                total_words += logging_outputs['tgt_size']

        #~ self.model.train()
        return total_loss / total_words
        
    def train_epoch(self, epoch, batchOrder=None, iteration=None):
        
        opt = self.opt
        trainData = self.trainData
        
        # Clear the gradients of the model
        self.runner.zero_grad()

        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # Shuffle mini batch order.
        if not batchOrder:
            batchOrder = trainData.create_order()
        else:
            trainData.batchOrder = batchOrder
            
        if iteration is not None and iteration > -1:
            trainData.set_index(iteration)

        total_loss, total_words = 0, 0
        report_loss, report_tgt_words = 0, 0
        report_src_words = 0
        start = time.time()
        nSamples = len(trainData)
        dataset = self.dataset
        
        counter = 0
        num_accumulated_words = 0
        
        for i in range(nSamples):

            #~ batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            # Exclude original indices.
            #~ batch = trainData[batchIdx]
            curriculum = (epoch < opt.curriculum)
            
            samples = trainData.next(curriculum=curriculum)
            
            logging_outputs = self.runner.step(samples)
            loss_data = logging_outputs['loss']
            ooms = logging_outputs['oom']
            
            counter = counter + 1 
            num_accumulated_words += logging_outputs['tgt_size']
            
            # We only update the parameters after getting gradients from n mini-batches
            # simulating the multi-gpu situation
            if counter == opt.virtual_gpu:
                # Update the parameters.
                grad_denom = 1
                if self.opt.normalize_gradient:
                    grad_denom = num_accumulated_words
                
                self.runner.update_parameters(grad_denom=grad_denom)
                self.runner.zero_grad()
                counter = 0
                num_accumulated_words = 0
                if opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every :
                    valid_loss = self.eval(self.validData)
                    valid_ppl = math.exp(min(valid_loss, 100))
                    print('Validation perplexity: %g' % valid_ppl)
                    
                    ep = float(epoch) - 1. + ((float(i) + 1.) / nSamples)
                    
                    self.save(ep, valid_ppl, batchOrder=batchOrder, iteration=i)
            

            num_words = logging_outputs['tgt_size']
            report_loss += loss_data
            report_tgt_words += num_words
            report_src_words += logging_outputs['src_size']
            total_loss += loss_data
            total_words += num_words
            
            optim = self.runner.get_optim()
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
    
    
    
    def run(self, save_file=None):
        
        
        opt = self.opt
        model = self.model
        dataset = self.dataset
        
        # Try to load the save_file
        checkpoint = None
        if save_file:
            checkpoint = torch.load(save_file)        
        if checkpoint is not None:
            print('Loading model and optim from checkpoint at %s' % save_file) 
            batchOrder = checkpoint['batchOrder']
            iteration = checkpoint['iteration'] + 1
            opt.start_epoch = int(math.floor(float(checkpoint['epoch'] + 1)))
        else:
            batchOrder = None
            iteration = None
            
        
        valid_loss = self.eval(self.validData)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g' % valid_ppl)
        
        self.start_time = time.time()
        
        for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
            print('')

            #  (1) train for one epoch on the training set
            train_loss = self.train_epoch(epoch, batchOrder=batchOrder,
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
