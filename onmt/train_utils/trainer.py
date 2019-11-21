from __future__ import division

import onmt
import onmt.Markdown
import onmt.modules
import torch
from torch.autograd import Variable
import math
import time, datetime
import os
from onmt.ModelConstructor import init_model_parameters
from onmt.utils import checkpoint_paths, normalize_gradients
from apex import amp


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

        self.additional_data = []

    def add_additional_data(self,d,ratio):
        self.additional_data = d
        if ratio == "-1" :
            self.additional_data_ratio = [1]*(len(self.additional_data + 1))
        else:
            self.additional_data_ratio = [int(s) for s in ratio.split(";")]
            assert(len(self.additional_data_ratio) == len(self.additional_data) + 1)

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

    def _get_grads(self):
        grads = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                raise RuntimeError('Model parameter did not receive gradient: ' + name + '. '
                                   'Use the param in the forward pass or set requires_grad=False.' +
                                   ' If you are using Stochastic model + fp16 - try to increase the number of minibatches' +
                                   ' each update to avoid uninitialized gradients.' )
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
            out[offset:offset+numel].copy_(g.view(-1))
            offset += numel
        return out[:offset]


class XETrainer(BaseTrainer):

    def __init__(self, model, loss_function, train_data, valid_data, dicts, opt, setup_optimizer=True):
        super().__init__(model, loss_function, train_data, valid_data, dicts, opt)

        if self.cuda:
            torch.cuda.set_device(self.opt.gpus[0])
            torch.manual_seed(self.opt.seed)
            self.loss_function = self.loss_function.cuda()
            self.model = self.model.cuda()

        if setup_optimizer:

            self.optim = onmt.Optim(opt)
            self.optim.set_parameters(self.model.parameters())

            opt_level = "O0" if not self.opt.fp16 else "O2"
            print("Optimization level: %s" % opt_level)
            self.model, self.optim.optimizer = amp.initialize(self.model,
                                                                   self.optim.optimizer,
                                                                   opt_level=opt_level,
                                                                   keep_batchnorm_fp32=False, loss_scale="dynamic",
                                                                   verbosity=0)
        # An ugly hack to switch between align right and align left
        if hasattr(self.model, 'relative'):
            if self.model.relative:
                self.train_data.src_align_right = True
                self.train_data.tgt_align_right = False
                self.valid_data.src_align_right = True
                self.valid_data.tgt_align_right = False

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
                'iteration' : iteration,
                'batch_order' : batch_order,
                'optim': optim_state_dict,
                'additional_batch_order' : getattr(self, 'additional_batch_order', None),
                'additional_data_iteration' : getattr(self, 'additional_data_iteration', None),
                'amp': amp.state_dict()
        }
        
        file_name = '%s_ppl_%.6f_e%.2f.pt' % (opt.save_model, valid_ppl, epoch)
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
                
        batch_order = data.create_order(random=False)
        self.model.eval()
        self.model.reset_states()
        """ PyTorch semantics: save space by not creating gradients """
        with torch.no_grad():
            for i in range(len(data)):

                batch = data.next()[0]

                if self.cuda:
                    batch.cuda(fp16=self.opt.fp16)
                
                """ outputs can be either 
                        hidden states from decoder or
                        prob distribution from decoder generator
                """
                targets = batch.get('target_output')
                tgt_mask = targets.ne(onmt.Constants.PAD)
                outputs = self.model(batch, target_masking=tgt_mask)

                outputs['tgt_mask'] = tgt_mask

                loss_dict = self.loss_function(outputs, targets, model=self.model)

                loss_data = loss_dict['data']

                total_loss += loss_data
                total_words += batch.tgt_size

        self.model.train()
        return total_loss / total_words
        
    def train_epoch(self, epoch, resume=False, batch_order=None, iteration=0):
        
        opt = self.opt
        train_data = self.train_data
        
        # Clear the gradients of the model
        # self.runner.zero_grad()
        self.model.zero_grad()
        self.model.reset_states()

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
        denom = 3584
        nan = False
        
        for i in range(iteration, n_samples):

            curriculum = (epoch < opt.curriculum)

            batches = [train_data.next(curriculum=curriculum)[0]]

            if(len(self.additional_data) > 0 and
                i % self.additional_data_ratio[0] == 0):
                for j in range(len(self.additional_data)):
                    for k in range(self.additional_data_ratio[j+1]):
                        if self.additional_data_iteration[j] == len(self.additional_data[j]):
                            self.additional_data_iteration[j] = 0
                            self.additional_data[j].shuffle()
                            self.additional_batch_order[j] = self.additional_data[j].create_order()

                        batches.append(self.additional_data[j].next()[0])
                        self.additional_data_iteration[j] += 1

            for b in range(len(batches)):
                batch = batches[b]
                if self.cuda:
                    batch.cuda(fp16=self.opt.fp16)
            
                oom = False
                try:
                    # outputs is a dictionary containing keys/values necessary for loss function
                    # can be flexibly controlled within models for easier extensibility
                    targets = batch.get('target_output')
                    tgt_mask = targets.data.ne(onmt.Constants.PAD)
                    outputs = self.model(batch, target_masking=tgt_mask, zero_encoder=opt.zero_encoder)

                    batch_size = batch.size

                    outputs['tgt_mask'] = tgt_mask

                    loss_dict = self.loss_function(outputs, targets, model=self.model)
                    loss_data = loss_dict['data']
                    loss = loss_dict['loss'].div(denom)  # a little trick to avoid gradient overflow with fp16

                    optimizer = self.optim.optimizer

                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('| WARNING: ran out of memory on GPU , skipping batch')
                        oom = True
                        torch.cuda.empty_cache()
                        loss = 0
                    else:
                        raise e

                if loss != loss:
                    # catching NAN problem
                    oom = True
                    self.model.zero_grad()
                    self.optim.zero_grad()
                    num_accumulated_words = 0
                    num_accumulated_sents = 0
                
                if not oom:
                    src_size = batch.src_size
                    tgt_size = batch.tgt_size
                
                    counter = counter + 1
                    num_accumulated_words += tgt_size
                    num_accumulated_sents += batch_size
                
                    #   We only update the parameters after getting gradients from n mini-batches
                    # simulating the multi-gpu situation
                    # if counter == opt.virtual_gpu:
                    # if counter >= opt.batch_size_update:
                
                    if num_accumulated_words >= opt.batch_size_update * 0.95:
                        grad_denom = 1 / denom
                        if self.opt.normalize_gradient:
                            grad_denom = num_accumulated_words / denom
                        normalize_gradients(amp.master_params(optimizer), grad_denom)
                        # Update the parameters.
                        self.optim.step(grad_denom=grad_denom)
                        self.optim.zero_grad()
                        self.model.zero_grad()
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
                    report_loss += loss_data
                    report_tgt_words += num_words
                    report_src_words += src_size
                    total_loss += loss_data
                    total_words += num_words
                    optim = self.optim

                    if b == 0 and (i == 0 or (i % opt.log_interval == -1 % opt.log_interval)):
                        print(("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; lr: %.7f ; num updates: %7d " +
                           "%5.0f src tok/s; %5.0f tgt tok/s; %s elapsed") %
                          (epoch, i+1, len(train_data),
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

    # def run(self, save_file=None):
    def run(self, checkpoint=None):

        opt = self.opt
        model = self.model
        optim = self.optim
        
        # Try to load the save_file
        # checkpoint = None
        # if save_file:
        #     checkpoint = torch.load(save_file, map_location=lambda storage, loc: storage)
        
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint['model'])
            
            if not opt.reset_optim:
                self.optim.load_state_dict(checkpoint['optim'])
                if 'amp' in checkpoint:
                    amp.load_state_dict(checkpoint['amp'])
                if 'batch_order' in checkpoint:
                    batch_order = checkpoint['batch_order']
                    iteration = checkpoint['iteration'] + 1
                else:
                    batch_order = None
                    iteration = 0
                opt.start_epoch = int(math.floor(float(checkpoint['epoch'] + 1)))

                resume=True
                if len(self.additional_data) > 0:
                    if 'additional_batch_order' in checkpoint:
                        self.additional_batch_order = checkpoint['additional_batch_order']
                        self.additional_data_iteration = checkpoint['additional_data_iteration']
                    else:
                        self.init_additional_data()
            else:
                batch_order = None
                iteration = 0
                resume=False
                self.init_additional_data()

            del checkpoint['model']
            del checkpoint['optim']
            del checkpoint
        else:
            batch_order = None
            iteration = 0
            print('Initializing model parameters')
            init_model_parameters(model, opt)
            resume=False
            self.init_additional_data()

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

    def init_additional_data(self):
        self.additional_batch_order = []
        self.additional_data_iteration = []
        for i in range(len(self.additional_data)):
            self.additional_data_iteration.append(0)
            self.additional_data[i].shuffle()
            self.additional_batch_order.append(self.additional_data[i].create_order())


