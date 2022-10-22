from __future__ import division

import datetime
import gc
import inspect_model
import math
import os
import re
import time
import torch
from apex import amp

import onmt
import onmt.markdown
import onmt.modules
from onmt.data.data_iterator import DataIterator
from onmt.data.dataset import rewrap
from onmt.model_factory import build_model, build_language_model, optimize_model
from onmt.model_factory import init_model_parameters
from onmt.train_utils.stats import Logger
from onmt.utils import checkpoint_paths, normalize_gradients
from .trainer import BaseTrainer


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def generate_data_iterator(dataset, seed, num_workers=1, epoch=1., buffer_size=0):
    # check if dataset is a list:
    if isinstance(dataset, list):
        # this is a multidataset
        data_iterator = MultiDataIterator(dataset, seed=seed, num_workers=num_workers,
                                          epoch=epoch, buffer_size=buffer_size)
    else:

        data_iterator = DataIterator(dataset, dataset.collater, dataset.batches, seed=seed,
                                     num_workers=num_workers, epoch=epoch, buffer_size=buffer_size)

    return data_iterator


class SpeechAETrainer(BaseTrainer):
    def __init__(self, model, loss_function, train_data, valid_data, dicts, opt, setup_optimizer=True):
        super().__init__(model, loss_function, train_data, valid_data, dicts, opt)

        self.n_gpus = len(self.opt.gpus)

        if self.cuda:
            torch.cuda.set_device(self.opt.gpus[0])
            if self.opt.seed >= 0:
                torch.manual_seed(self.opt.seed)
            self.loss_function = self.loss_function.cuda()
            self.model = self.model.cuda()

        if setup_optimizer:

            self.optim = onmt.Optim(opt)
            self.optim.set_parameters(self.model.parameters())

            if not self.opt.fp16:
                opt_level = "O0"
                keep_batchnorm_fp32 = False
            elif self.opt.fp16_mixed:
                opt_level = "O1"
                keep_batchnorm_fp32 = None
            else:
                opt_level = "O2"
                keep_batchnorm_fp32 = False

            if self.cuda:
                self.model, self.optim.optimizer = amp.initialize(self.model,
                                                                  self.optim.optimizer,
                                                                  opt_level=opt_level,
                                                                  keep_batchnorm_fp32=keep_batchnorm_fp32,
                                                                  loss_scale="dynamic",
                                                                  verbosity=1 if self.opt.verbose else 0)

    def warm_up(self):
        """
        Warmup the memory allocator, by attempting to fit the largest batch
        :return:
        """
        print("Tacotron_warmup")
        if self.opt.memory_profiling:
            from pytorch_memlab import MemReporter
            reporter = MemReporter()

        batch = self.train_data[0].get_largest_batch() if isinstance(self.train_data, list) \
            else self.train_data.get_largest_batch()
        opt = self.opt

        if self.cuda:
            batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)

        self.model.train()
        self.model.zero_grad()
        oom = False

        if self.opt.memory_profiling:
            print("Input size: ")
            print(batch.size, batch.src_size, batch.tgt_size)

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        try:
            targets = batch.get('target_output')
            tgt_mask = None
            outputs = self.model(batch)
            gate_padded = batch.get('gate_padded')

            if self.opt.n_frames_per_step > 1:
                slice = torch.arange(self.opt.n_frames_per_step - 1, gate_padded.size(1), self.opt.n_frames_per_step)
                gate_padded = gate_padded[:, slice]

            src_org = batch.get('source_org')
            src_org = src_org.narrow(2, 1, src_org.size(2) - 1)
            target = [src_org.permute(1,2,0).contiguous(), gate_padded]
            loss = self.loss_function(outputs, target)
            # loss_dict = self.loss_function(outputs, targets, model=self.model)
            loss = loss # a little trick to avoid gradient overflow with fp16
            full_loss = loss

            optimizer = self.optim.optimizer

            if self.opt.memory_profiling:
                reporter.report(verbose=True)



            if self.cuda:
                with amp.scale_loss(full_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.div_(batch.tgt_size).backward()

            if self.opt.memory_profiling:
                print('========= after backward =========')
                reporter.report(verbose=True)

            self.model.zero_grad()
            self.optim.zero_grad()
        except RuntimeError as e:
            if 'out of memory' in str(e):
                oom = True
            else:
                raise e

        if oom:
            print("* Warning: out-of-memory in warming up. This is due to the largest batch is too big for the GPU.")
        else:
            print("* Warming up successuflly.")

        if self.opt.memory_profiling:
            if hasattr(torch.cuda, 'memory_summary'):
                print(torch.cuda.memory_summary())
            exit()

    def save(self, epoch, valid_ppl, itr=None):

        opt = self.opt
        model = self.model
        dicts = self.dicts

        model_state_dict = self.model.state_dict()
        optim_state_dict = self.optim.state_dict()

        if itr:
            itr_state_dict = itr.state_dict()
        else:
            itr_state_dict = None

        #  drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'dicts': dicts,
            'opt': opt,
            'epoch': epoch,
            'itr': itr_state_dict,
            'optim': optim_state_dict,
            'amp': amp.state_dict()
        }

        file_name = '%s_ppl_%.6f_e%.2f.pt' % (opt.save_model, valid_ppl, epoch)
        print('Writing to %s' % file_name)
        torch.save(checkpoint, file_name)

        # check the save directory here
        checkpoint_dir = os.path.dirname(opt.save_model)
        existed_save_files = checkpoint_paths(checkpoint_dir)
        for save_file in existed_save_files[opt.keep_save_files:]:
            print(" * Deleting old save file %s ...." % save_file)
            os.remove(save_file)

    def run(self, checkpoint=None):

        opt = self.opt
        model = self.model
        optim = self.optim

        if checkpoint is not None:
            self.model.load_state_dict(checkpoint['model'])
            prec_opt = checkpoint['opt'] if 'opt' in checkpoint else None

            if not opt.reset_optim:
                print("* Loading optimizer states ... ")
                self.optim.load_state_dict(checkpoint['optim'])
                if prec_opt is not None and hasattr(prec_opt, "fp16_mixed"):
                    # Only load amp information if the mode is the same
                    # Maybe its better to change between optimization mode?
                    if opt.fp16_mixed == prec_opt.fp16_mixed and opt.fp16 == prec_opt.fp16:
                        if 'amp' in checkpoint:
                            amp.load_state_dict(checkpoint['amp'])

                # Only load the progress when we use the same optimizer
                if 'itr' in checkpoint:
                    itr_progress = checkpoint['itr']
                else:
                    itr_progress = None

                resume = True
                start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 1
                if start_epoch is None:
                    start_epoch = 1
            else:
                itr_progress = None
                resume = False
                start_epoch = 1

            del checkpoint['model']
            del checkpoint['optim']
            del checkpoint
        else:
            itr_progress = None
            print('Initializing model parameters')
            init_model_parameters(model, opt)
            resume = False
            start_epoch = 1

        if opt.load_encoder_from:
            self.load_encoder_weight(opt.load_encoder_from)

        if opt.load_decoder_from:
            self.load_decoder_weight(opt.load_decoder_from)

        # if we are on a GPU: warm up the memory allocator
        if self.cuda:
            self.warm_up()

            valid_loss = self.eval(self.valid_data)

            print('Validation loss: %g' % valid_loss)

        self.start_time = time.time()

        for epoch in range(start_epoch, start_epoch + opt.epochs):
            print('')

            #  (1) train for one epoch on the training set
            train_loss = self.train_epoch(epoch, resume=resume, itr_progress=itr_progress)
            print('Train loss: %g' % train_loss)

            #  (2) evaluate on the validation set
            valid_loss = self.eval(self.valid_data)
            print('Validation loss: %g' % valid_loss)

            self.save(epoch, valid_loss)
            itr_progress = None
            resume = False

    def eval(self, data):
        total_loss = 0
        total_tgt_frames = 0
        total_sent = 0
        opt = self.opt

        self.model.eval()
        self.loss_function.eval()
        # self.model.reset_states()

        # the data iterator creates an epoch iterator
        data_iterator = generate_data_iterator(data, seed=self.opt.seed,
                                               num_workers=opt.num_workers, epoch=1, buffer_size=opt.buffer_size)
        epoch_iterator = data_iterator.next_epoch_itr(False, pin_memory=False)

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        """ PyTorch semantics: save space by not creating gradients """

        data_size = len(epoch_iterator)
        i = 0

        with torch.no_grad():
            # for i in range(len()):
            while not data_iterator.end_of_epoch():
                # batch = data.next()[0]
                batch = next(epoch_iterator)
                if isinstance(batch, list):
                    batch = batch[0]
                batch = rewrap(batch)

                if self.cuda:
                    batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)

                """ outputs can be either 
                        hidden states from decoder or
                        prob distribution from decoder generator
                """

                outputs = self.model(batch)

                gate_padded = batch.get('gate_padded')

                if self.opt.n_frames_per_step > 1:
                    slice = torch.arange(self.opt.n_frames_per_step - 1, gate_padded.size(1), self.opt.n_frames_per_step)
                    gate_padded = gate_padded[:, slice]

                src_org = batch.get('source_org')
                src_org = src_org.narrow(2, 1, src_org.size(2) - 1)
                target = [src_org.permute(1, 2, 0).contiguous(), gate_padded]
                loss = self.loss_function(outputs, target)
                loss_data = loss.data.item()



                total_loss += loss_data
                total_tgt_frames += batch.src_size
                total_sent += batch.size
                i = i + 1

        self.model.train()
        self.loss_function.train()
        return total_loss / data_size * 100

    def train_epoch(self, epoch, resume=False, itr_progress=None):

        global rec_ppl
        opt = self.opt
        train_data = self.train_data
        streaming = opt.streaming

        self.model.train()
        self.loss_function.train()
        # Clear the gradients of the model
        # self.runner.zero_grad()
        self.model.zero_grad()


        dataset = train_data
        data_iterator = generate_data_iterator(dataset, seed=self.opt.seed, num_workers=opt.num_workers,
                                               epoch=epoch, buffer_size=opt.buffer_size)

        if resume:
            data_iterator.load_state_dict(itr_progress)

        epoch_iterator = data_iterator.next_epoch_itr(not streaming, pin_memory=opt.pin_memory)

        total_loss, total_frames = 0, 0

        report_loss, report_tgt_frames,report_sent = 0, 0, 0



        start = time.time()
        n_samples = len(epoch_iterator)

        counter = 0

        num_accumulated_sents = 0
        grad_scaler = -1

        nan = False
        nan_counter = 0



        i = data_iterator.iterations_in_epoch if not isinstance(train_data, list) else epoch_iterator.n_yielded

        while not data_iterator.end_of_epoch():

            curriculum = (epoch < opt.curriculum)

            # this batch generator is not very clean atm
            batch = next(epoch_iterator)
            if isinstance(batch, list) and self.n_gpus == 1:
                batch = batch[0]
            batch = rewrap(batch)
            if grad_scaler == -1:
                grad_scaler = 1  # if self.opt.update_frequency > 1 else batch.tgt_size

            if self.cuda:
                batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)

            oom = False
            try:
                # outputs is a dictionary containing keys/values necessary for loss function
                # can be flexibly controlled within models for easier extensibility
            #    targets = batch.get('target_output')
              #  tgt_mask = targets.ne(onmt.constants.PAD)
                outputs = self.model(batch)

                gate_padded = batch.get('gate_padded')

                if self.opt.n_frames_per_step > 1:
                    slice = torch.arange(0, gate_padded.size(1), self.opt.n_frames_per_step)
                    gate_padded = gate_padded[:, slice]

                src_org = batch.get('source_org')
                src_org = src_org.narrow(2, 1, src_org.size(2) - 1)

                target = [src_org.permute(1, 2, 0).contiguous(), gate_padded]
                loss = self.loss_function(outputs, target)


                batch_size = batch.size
                loss_data = loss.data.item()
                 # a little trick to avoid gradient overflow with fp16
                full_loss = loss

                optimizer = self.optim.optimizer

                # When the batch size is large, each gradient step is very easy to explode on fp16
                # Normalizing the loss to grad scaler ensures this will not happen
                full_loss.div_(grad_scaler)

                if self.cuda:
                    with amp.scale_loss(full_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    full_loss.backward()

                del outputs

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory on GPU , skipping batch')
                    oom = True
                    torch.cuda.empty_cache()
                    loss = 0
                    if opt.streaming:  # reset stream in this case ...
                        streaming_state = self.model.init_stream()
                else:
                    raise e

            if loss != loss:
                # catching NAN problem
                oom = True
                self.model.zero_grad()
                self.optim.zero_grad()
                nan_counter = nan_counter + 1
                print("Warning!!! Loss is Nan")
                if nan_counter >= 15:
                    raise ValueError("Training stopped because of multiple NaN occurence. "
                                     "For ASR, using the Relative Transformer is more stable and recommended.")
            else:
                nan_counter = 0

            if not oom:
                src_size = batch.src_size


                counter = counter + 1


                #   We only update the parameters after getting gradients from n mini-batches
                update_flag = False
                if counter >= opt.update_frequency > 0:
                    update_flag = True
                elif i == n_samples:  # update for the last minibatch
                    update_flag = True

                if update_flag:
                    # accumulated gradient case, in this case the update frequency
                    if (counter == 1 and self.opt.update_frequency != 1) or counter > 1:
                        grad_denom = 1 / grad_scaler
                        # if self.opt.normalize_gradient:
                        #     grad_denom = num_accumulated_words * grad_denom
                    else:
                        grad_denom = 1.0
                    # When we accumulate the gradients, each gradient is already normalized by a constant grad_scaler
                    normalize_gradients(amp.master_params(optimizer), grad_denom)
                    # Update the parameters.
                    if self.opt.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.opt.max_grad_norm)
                    self.optim.step()
                    self.optim.zero_grad()
                    self.model.zero_grad()
                    counter = 0
                    # num_accumulated_words = 0

                    grad_scaler = -1
                    num_updates = self.optim._step
                    if opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every:
                        valid_loss = self.eval(self.valid_data)
                        valid_ppl = math.exp(min(valid_loss, 100))
                        print('Validation perplexity: %g' % valid_ppl)

                        ep = float(epoch) - 1. + ((float(i) + 1.) / n_samples)

                        self.save(ep, valid_ppl, itr=data_iterator)


                report_loss += loss_data
                # report_tgt_words += num_words
                num_accumulated_sents += batch_size
                report_sent += batch_size
                total_frames+= src_size
                report_tgt_frames += src_size
                total_loss += loss_data

                optim = self.optim
                # batch_efficiency = total_non_pads / total_tokens




                if i == 0 or (i % opt.log_interval == -1 % opt.log_interval):
                    log_string = ("Epoch %2d, %5d/%5d; ; loss : %6.2f ; " %
                                  (epoch, i + 1, len(data_iterator),
                                   report_loss ))




                    log_string += ("lr: %.7f ; updates: %7d; " %
                                   (optim.getLearningRate(),
                                    optim._step))
                    #
                    log_string += ("%5.0f src tok/s " %
                                   (report_tgt_frames / (time.time() - start)))

                    log_string += ("%s elapsed" %
                                   str(datetime.timedelta(seconds=int(time.time() - self.start_time))))

                    print(log_string)

                    report_loss = 0
                    report_tgt_frames = 0
                    report_sent = 0
                    start = time.time()

                i = i + 1

        return total_loss / n_samples * 100


class XETrainer(BaseTrainer):

    def __init__(self, model, loss_function, train_data, valid_data, dicts, opt, setup_optimizer=True):
        super().__init__(model, loss_function, train_data, valid_data, dicts, opt)

        if opt.lfv_multilingual or opt.lid_loss:
            from onmt.models.speech_recognizer.lid_loss import CrossEntropyLIDLoss
            lid_loss = CrossEntropyLIDLoss(opt.n_languages, opt.label_smoothing, opt.fast_xentropy)
            self.loss_function.add_loss_function(lid_loss, 'lid_loss')

        self.n_gpus = len(self.opt.gpus)

        if self.cuda:
            torch.cuda.set_device(self.opt.gpus[0])
            if self.opt.seed >= 0:
                torch.manual_seed(self.opt.seed)
            self.loss_function = self.loss_function.cuda()
            self.model = self.model.cuda()

        if setup_optimizer:

            self.optim = onmt.Optim(opt)
            self.optim.set_parameters(self.model.parameters())

            if not self.opt.fp16:
                opt_level = "O0"
                keep_batchnorm_fp32 = False
            elif self.opt.fp16_mixed:
                opt_level = "O1"
                keep_batchnorm_fp32 = None
            else:
                opt_level = "O2"
                keep_batchnorm_fp32 = False

            if self.cuda:
                # print(234)
                self.model, self.optim.optimizer = amp.initialize(self.model,
                                                                  self.optim.optimizer,
                                                                  opt_level=opt_level,
                                                                  keep_batchnorm_fp32=keep_batchnorm_fp32,
                                                                  loss_scale="dynamic",
                                                                  verbosity=1 if self.opt.verbose else 0)
        # An ugly hack to switch between align right and align left
        if hasattr(self.model, 'relative'):
            if self.model.relative:
                self.train_data.src_align_right = True
                self.train_data.tgt_align_right = False
                self.valid_data.src_align_right = True
                self.valid_data.tgt_align_right = False
                self.valid_data.tgt_align_right = False

    def save(self, epoch, valid_ppl, itr=None):

        opt = self.opt
        model = self.model
        dicts = self.dicts

        model_state_dict = self.model.state_dict()
        optim_state_dict = self.optim.state_dict()

        if itr:
            itr_state_dict = itr.state_dict()
        else:
            itr_state_dict = None

        #  drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'dicts': dicts,
            'opt': opt,
            'epoch': epoch,
            'itr': itr_state_dict,
            'optim': optim_state_dict,
            'amp': amp.state_dict()
        }

        file_name = '%s_ppl_%.6f_e%.2f.pt' % (opt.save_model, valid_ppl, epoch)
        print('Writing to %s' % file_name)
        torch.save(checkpoint, file_name)

        # check the save directory here
        checkpoint_dir = os.path.dirname(opt.save_model)
        existed_save_files = checkpoint_paths(checkpoint_dir)
        for save_file in existed_save_files[opt.keep_save_files:]:
            print(" * Deleting old save file %s ...." % save_file)
            os.remove(save_file)

    def eval(self, data):
        total_loss = 0
        total_words = 0
        opt = self.opt

        self.model.eval()
        self.loss_function.eval()
        self.model.reset_states()

        # the data iterator creates an epoch iterator
        data_iterator = generate_data_iterator(data, seed=self.opt.seed,
                                               num_workers=opt.num_workers, epoch=1, buffer_size=opt.buffer_size)
        epoch_iterator = data_iterator.next_epoch_itr(False, pin_memory=False)

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        """ PyTorch semantics: save space by not creating gradients """

        data_size = len(epoch_iterator)
        i = 0

        with torch.no_grad():
            # for i in range(len()):
            while not data_iterator.end_of_epoch():
                # batch = data.next()[0]
                batch = next(epoch_iterator)
                if isinstance(batch, list):
                    batch = batch[0]
                batch = rewrap(batch)

                if self.cuda:
                    batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)

                """ outputs can be either 
                        hidden states from decoder or
                        prob distribution from decoder generator
                """
                targets = batch.get('target_output')
                tgt_mask = targets.ne(onmt.constants.PAD)
                outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
                                     mirror=opt.mirror_loss, streaming_state=streaming_state, nce=opt.nce)

                if opt.streaming:
                    streaming_state = outputs['streaming_state']

                outputs['tgt_mask'] = tgt_mask

                loss_dict = self.loss_function(outputs, targets, model=self.model, eval=True)

                loss_data = loss_dict['data']

                total_loss += loss_data
                total_words += batch.tgt_size
                i = i + 1

        self.model.train()
        self.loss_function.train()
        return total_loss / total_words

    def train_epoch(self, epoch, resume=False, itr_progress=None):

        global rec_ppl
        opt = self.opt
        train_data = self.train_data
        streaming = opt.streaming

        self.model.train()
        self.loss_function.train()
        # Clear the gradients of the model
        # self.runner.zero_grad()
        self.model.zero_grad()
        self.model.reset_states()

        dataset = train_data
        data_iterator = generate_data_iterator(dataset, seed=self.opt.seed, num_workers=opt.num_workers,
                                               epoch=epoch, buffer_size=opt.buffer_size)

        if resume:
            data_iterator.load_state_dict(itr_progress)

        epoch_iterator = data_iterator.next_epoch_itr(not streaming, pin_memory=opt.pin_memory)

        total_tokens, total_loss, total_words = 0, 0, 0
        total_non_pads = 0
        report_loss, report_tgt_words = 0, 0
        report_src_words = 0
        report_rec_loss, report_rev_loss, report_mirror_loss = 0, 0, 0
        start = time.time()
        n_samples = len(epoch_iterator)

        counter = 0
        num_accumulated_words = 0
        num_accumulated_sents = 0
        grad_scaler = -1

        nan = False
        nan_counter = 0

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        i = data_iterator.iterations_in_epoch if not isinstance(train_data, list) else epoch_iterator.n_yielded

        while not data_iterator.end_of_epoch():

            curriculum = (epoch < opt.curriculum)

            # this batch generator is not very clean atm
            batch = next(epoch_iterator)
            if isinstance(batch, list) and self.n_gpus == 1:
                batch = batch[0]
            batch = rewrap(batch)
            if grad_scaler == -1:
                grad_scaler = 1  # if self.opt.update_frequency > 1 else batch.tgt_size

            if self.cuda:
                batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)

            # if opt.streaming:
            #     if train_data.is_new_stream():
            #         streaming_state = self.model.init_stream()
            # else:
            #     streaming_state = None

            oom = False
            try:
                # outputs is a dictionary containing keys/values necessary for loss function
                # can be flexibly controlled within models for easier extensibility
                targets = batch.get('target_output')
                tgt_mask = targets.ne(onmt.constants.PAD)

                outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
                                     zero_encoder=opt.zero_encoder,
                                     mirror=opt.mirror_loss, streaming_state=streaming_state,
                                     nce=opt.nce)
                # print("time " + str(time.time() - start_time_t))
                batch_size = batch.size

                outputs['tgt_mask'] = tgt_mask

                loss_dict = self.loss_function(outputs, targets, model=self.model)
                loss_data = loss_dict['data']
                loss = loss_dict['loss']  # a little trick to avoid gradient overflow with fp16
                full_loss = loss

                if opt.mirror_loss:
                    rev_loss = loss_dict['rev_loss']
                    rev_loss_data = loss_dict['rev_loss_data']
                    mirror_loss = loss_dict['mirror_loss']
                    full_loss = full_loss + rev_loss + mirror_loss
                    mirror_loss_data = loss_dict['mirror_loss'].item()
                else:
                    rev_loss_data = None
                    mirror_loss_data = 0

                # reconstruction loss
                if opt.reconstruct:
                    rec_loss = loss_dict['rec_loss']
                    rec_loss = rec_loss
                    full_loss = full_loss + rec_loss
                    rec_loss_data = loss_dict['rec_loss_data']
                else:
                    rec_loss_data = None

                if opt.lfv_multilingual or opt.lid_loss:
                    lid_logits = outputs['lid_logits']
                    lid_labels = batch.get('target_lang')
                    lid_loss_function = self.loss_function.get_loss_function('lid_loss')
                    lid_loss = lid_loss_function([lid_logits.unsqueeze(0)]  , lid_labels)
                    full_loss = full_loss + lid_loss

                optimizer = self.optim.optimizer

                # When the batch size is large, each gradient step is very easy to explode on fp16
                # Normalizing the loss to grad scaler ensures this will not happen
                full_loss.div_(grad_scaler)

                if self.cuda:
                    with amp.scale_loss(full_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    full_loss.backward()

                del outputs

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory on GPU , skipping batch')
                    oom = True
                    torch.cuda.empty_cache()
                    loss = 0
                    if opt.streaming:  # reset stream in this case ...
                        streaming_state = self.model.init_stream()
                else:
                    raise e

            if loss != loss:
                # catching NAN problem
                oom = True
                self.model.zero_grad()
                self.optim.zero_grad()
                num_accumulated_words = 0
                num_accumulated_sents = 0
                nan_counter = nan_counter + 1
                print("Warning!!! Loss is Nan")
                if nan_counter >= 15:
                    raise ValueError("Training stopped because of multiple NaN occurence. "
                                     "For ASR, using the Relative Transformer is more stable and recommended.")
            else:
                nan_counter = 0

            if not oom:
                src_size = batch.src_size
                tgt_size = batch.tgt_size

                counter = counter + 1
                num_accumulated_words += tgt_size
                num_accumulated_sents += batch_size

                #   We only update the parameters after getting gradients from n mini-batches
                update_flag = False
                if counter >= opt.update_frequency > 0:
                    update_flag = True
                elif 0 < opt.batch_size_update <= num_accumulated_words:
                    update_flag = True
                elif i == n_samples:  # update for the last minibatch
                    update_flag = True

                if update_flag:
                    # accumulated gradient case, in this case the update frequency
                    if (counter == 1 and self.opt.update_frequency != 1) or counter > 1:
                        grad_denom = 1 / grad_scaler
                        if self.opt.normalize_gradient:
                            grad_denom = num_accumulated_words * grad_denom
                    else:
                        grad_denom = 1
                    # When we accumulate the gradients, each gradient is already normalized by a constant grad_scaler
                    normalize_gradients(amp.master_params(optimizer), grad_denom)
                    # Update the parameters.
                    if self.opt.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.opt.max_grad_norm)
                    self.optim.step()
                    self.optim.zero_grad()
                    self.model.zero_grad()
                    counter = 0
                    num_accumulated_words = 0
                    num_accumulated_sents = 0
                    grad_scaler = -1
                    num_updates = self.optim._step
                    if opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every:
                        valid_loss = self.eval(self.valid_data)
                        valid_ppl = math.exp(min(valid_loss, 100))
                        print('Validation perplexity: %g' % valid_ppl)

                        ep = float(epoch) - 1. + ((float(i) + 1.) / n_samples)

                        self.save(ep, valid_ppl, itr=data_iterator)

                num_words = tgt_size
                report_loss += loss_data
                report_tgt_words += num_words
                report_src_words += src_size
                total_loss += loss_data
                total_words += num_words
                total_tokens += batch.get('target_output').nelement()
                total_non_pads += batch.get('target_output').ne(onmt.constants.PAD).sum().item()
                optim = self.optim
                batch_efficiency = total_non_pads / total_tokens

                if opt.reconstruct:
                    report_rec_loss += rec_loss_data

                if opt.mirror_loss:
                    report_rev_loss += rev_loss_data
                    report_mirror_loss += mirror_loss_data

                if i == 0 or (i % opt.log_interval == -1 % opt.log_interval):
                    log_string = ("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; " %
                                  (epoch, i + 1, len(data_iterator),
                                   math.exp(report_loss / report_tgt_words)))

                    if opt.reconstruct:
                        rec_ppl = math.exp(report_rec_loss / report_src_words.item())
                        log_string += (" rec_ppl: %6.2f ; " % rec_ppl)

                    if opt.mirror_loss:
                        rev_ppl = math.exp(report_rev_loss / report_tgt_words)
                        log_string += (" rev_ppl: %6.2f ; " % rev_ppl)
                        # mirror loss per word
                        log_string += (" mir_loss: %6.2f ; " % (report_mirror_loss / report_tgt_words))

                    log_string += ("lr: %.7f ; updates: %7d; " %
                                   (optim.getLearningRate(),
                                    optim._step))

                    log_string += ("%5.0f src tok/s; %5.0f tgt tok/s; " %
                                   (report_src_words / (time.time() - start),
                                    report_tgt_words / (time.time() - start)))

                    log_string += ("%s elapsed" %
                                   str(datetime.timedelta(seconds=int(time.time() - self.start_time))))

                    print(log_string)

                    report_loss = 0
                    report_tgt_words, report_src_words = 0, 0
                    report_rec_loss, report_rev_loss, report_mirror_loss = 0, 0, 0
                    start = time.time()

                i = i + 1

        return total_loss / total_words

    # def run(self, save_file=None):
    def run(self, checkpoint=None):

        opt = self.opt
        model = self.model
        optim = self.optim

        if checkpoint is not None:
            self.model.load_state_dict(checkpoint['model'])
            prec_opt = checkpoint['opt'] if 'opt' in checkpoint else None

            if not opt.reset_optim:
                print("* Loading optimizer states ... ")
                self.optim.load_state_dict(checkpoint['optim'])
                if prec_opt is not None and hasattr(prec_opt, "fp16_mixed"):
                    # Only load amp information if the mode is the same
                    # Maybe its better to change between optimization mode?
                    if opt.fp16_mixed == prec_opt.fp16_mixed and opt.fp16 == prec_opt.fp16:
                        if 'amp' in checkpoint:
                            amp.load_state_dict(checkpoint['amp'])

                # Only load the progress when we use the same optimizer
                if 'itr' in checkpoint:
                    itr_progress = checkpoint['itr']
                else:
                    itr_progress = None

                resume = True
                start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 1
                if start_epoch is None:
                    start_epoch = 1
            else:
                itr_progress = None
                resume = False
                start_epoch = 1

            del checkpoint['model']
            del checkpoint['optim']
            del checkpoint
        else:
            itr_progress = None
            print('Initializing model parameters')
            init_model_parameters(model, opt)
            resume = False
            start_epoch = 1

        if opt.load_encoder_from:
            self.load_encoder_weight(opt.load_encoder_from)

        if opt.load_decoder_from:
            self.load_decoder_weight(opt.load_decoder_from)

        # if we are on a GPU: warm up the memory allocator
        self.start_time = time.time()


        if self.cuda:
            self.warm_up()

            valid_loss = self.eval(self.valid_data)
            valid_ppl = math.exp(min(valid_loss, 100))

            print('Validation perplexity: %g' % valid_ppl)

            # valid_loss = self.train_epoch(0)
            # valid_ppl = math.exp(min(valid_loss, 100))
            #
            # print('Validation perplexity: %g' % valid_ppl)


        for epoch in range(start_epoch, start_epoch + opt.epochs):
            print('')

            #  (1) train for one epoch on the training set
            train_loss = self.train_epoch(epoch, resume=resume, itr_progress=itr_progress)
            train_ppl = math.exp(min(train_loss, 100))
            print('Train perplexity: %g' % train_ppl)

            #  (2) evaluate on the validation set
            valid_loss = self.eval(self.valid_data)
            valid_ppl = math.exp(min(valid_loss, 100))
            print('Validation perplexity: %g' % valid_ppl)

            self.save(epoch, valid_ppl)
            itr_progress = None
            resume = False
