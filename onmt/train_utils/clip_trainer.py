from __future__ import division

import datetime
import gc
import math
import os
import re
import time
import torch
import copy
import sys
import contextlib

import onmt
import onmt.markdown
import onmt.modules
from onmt.data.data_iterator import DataIterator
from onmt.data.multidata_iterator import MultiDataIterator
from onmt.data.dataset import rewrap
from onmt.model_factory import build_model, build_language_model, optimize_model
from onmt.model_factory import init_model_parameters
from onmt.modules.loss import NMTLossFunc, NMTAndCTCLossFunc
from onmt.train_utils.stats import Logger
from onmt.utils import checkpoint_paths, normalize_gradients, clip_grad_norm
from onmt.model_factory import build_model, optimize_model, init_model_parameters
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP_model
from torch.cuda.amp import autocast
import warnings
from onmt.constants import add_tokenidx
import dill

from .mp_trainer import prepare_sample, is_list, generate_data_iterator, zero_tensor, all_reduce_and_rescale_tensors
from .mp_trainer import Trainer

# ignore the pytorch -> numpy conversion warnings
warnings.filterwarnings("ignore", category=UserWarning)


class ClipTrainer(Trainer):

    def __init__(self, device, dicts, opt, constants=None, setup_optimizer=True):
        """
        :param model:
        :param device: int (GPU id)
        :param loss_function:
        :param train_data:
        :param valid_data:
        :param dicts:
        :param opt:
        """
        self.device = device
        opt.node_rank = 0
        opt.nodes = 1

        self.world_size = len(opt.gpus)
        self.constants = dill.loads(constants) if constants is not None else None

        # in the case of single node distributed, it should equal self.device
        self.rank = self.device

        # make a group to later use with self.all_reduce
        self.group = dist.group.WORLD

        self.print("[INFO] Training Options:", opt)
        if self.world_size > 1:
            dist.init_process_group(backend='nccl', init_method='env://', world_size=self.world_size, rank=self.rank)

        self.model = None

        self.dicts = dicts
        self.opt = opt
        self.cuda = (len(opt.gpus) >= 1 and opt.gpus[0] >= 0)

        if self.cuda:
            torch.cuda.set_device(self.device)

        assert self.cuda, "[ERROR] Training is only available on GPUs."

        self.start_time = 0

        torch.manual_seed(self.opt.seed)

        # note: we must start creating models after ccreating the processes
        # for some reason passing a pre-created model to a process creates a "pickle" error

        if self.is_main():
            print("[INFO] Building models .... ", flush=True)
            print("Languages: ", dicts['langs'], flush=True)
        model = build_model(opt, dicts, False, self.constants)

        """ Building the loss function """

        from onmt.modules.loss import CLIPCrossEntropyLoss
        loss_function = CLIPCrossEntropyLoss(temperature=1.0)

        # distributed is required to convert BatchNorm to SyncBatchNorm for DDP
        optimize_model(model, distributed=(self.world_size > 1))

        init_model_parameters(model, opt)
        self.model = model
        self.loss_function = loss_function
        self.grad_scaler = torch.cuda.amp.GradScaler()

        if opt.load_from:
            checkpoint = torch.load(opt.load_from, map_location=lambda storage, loc: storage)

            try:
                self.model.load_state_dict(checkpoint['model'])
            except RuntimeError as e:
                self.model.load_state_dict(checkpoint['model'], strict=True)

            # if 'scaler' in checkpoint and checkpoint['scaler'] is not None:
            #     self.grad_scaler.load_state_dict(checkpoint['scaler'])

        if self.cuda:
            self.loss_function = self.loss_function.cuda(device=self.device)
            self.model = self.model.cuda(device=self.device)
            if opt.ctc_loss > 0.0:
                self.ctc_loss_function = self.ctc_loss_function.cuda(device=self.device)
            if opt.load_pretrained_classifier:
                self.classifier = self.classifier.cuda(device=self.device)

        if setup_optimizer:

            self.optim = onmt.Optim(opt)
            self.optim.set_parameters(self.model.parameters())

            if self.is_main():
                print("[INFO] Optimizer: ", self.optim.optimizer)

            if opt.load_from and not opt.reset_optim:
                if 'optim' in checkpoint and checkpoint['optim'] is not None and not opt.reset_optim:
                    self.optim.load_state_dict(checkpoint['optim'])

            if opt.starting_step > 0:
                print("[INFO] Optimizer starting from state %d " % opt.starting_step)
                self.optim.set_starting_step(opt.starting_step)

        if self.world_size > 1:
            find_unused_parameters = opt.find_unused_parameters

            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank],
                                                                   output_device=self.rank,
                                                                   find_unused_parameters=find_unused_parameters)

        if self.is_main():
            nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("[INFO] Total number of trainable paramaters: %d" % nparams)
            nparams = sum(p.numel() for p in model.parameters())
            print("[INFO] Total number of paramaters: %d" % nparams)

            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.is_main = True
            else:
                self.model.is_main = True

        print("[INFO] Process %d ready." % self.rank, flush=True)

    def save(self, epoch, valid_ppl, itr=None):

        opt = self.opt
        model = self.model
        dicts = self.dicts

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
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
            'scaler': self.grad_scaler.state_dict()
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

    #
    def eval(self, data):

        self.print("[INFO] Running cross-entropy evaluation...", flush=True)
        opt = self.opt

        rank = self.rank
        world_size = self.world_size
        # the data iterator creates an epoch iterator
        data_iterator = generate_data_iterator(data, rank, world_size, seed=self.opt.seed,
                                               num_workers=1, epoch=1, buffer_size=opt.buffer_size, split_even=False,
                                               dataset_ids=opt.valid_sets)
        epoch_iterator = data_iterator.next_epoch_itr(False, pin_memory=False)

        data_size = len(data_iterator)
        i = 0

        self.model.eval()
        self.loss_function.eval()

        total_loss = zero_tensor()
        total_words = zero_tensor()

        with torch.no_grad():
            # while not data_iterator.end_of_epoch():
            while i < len(epoch_iterator):
                samples = next(epoch_iterator)

                def maybe_no_sync():
                    if isinstance(self.model, DDP_model):
                        return self.model.no_sync()
                    else:
                        return contextlib.ExitStack()  # dummy contextmanager

                if samples:
                    with maybe_no_sync():
                        with autocast(enabled=opt.fp16):
                            batch = prepare_sample(samples, device=self.device)
                            targets = batch.get('target_output')
                            tgt_mask = targets.ne(onmt.constants.PAD)

                            output_dict = self.model(batch)
                            batch_size = batch.size

                            acoustic_features = output_dict['acoustic_features']
                            text_features = output_dict['text_features']

                            loss = self.loss_function(acoustic_features, text_features)

                            loss_data = loss.item()

                    total_loss.add_(loss_data)
                    total_words.add_(batch_size * 2)

                    i = i + 1
        # allreduce the total loss and total words from other processes
        self.all_reduce(total_loss, op=dist.ReduceOp.SUM, group=self.group)
        self.all_reduce(total_words, op=dist.ReduceOp.SUM, group=self.group)

        self.model.train()
        self.loss_function.train()

        return total_loss.item() / total_words.item()

    def train_epoch(self, train_data, valid_data, epoch, resume=False, itr_progress=None):

        streaming = False

        opt = self.opt
        grad_norm = -1

        # Clear the gradients of the model
        self.optim.zero_grad(set_to_none=opt.true_zero_grad)

        # note: for Training split_even=True
        dataset = train_data
        data_iterator = generate_data_iterator(dataset, self.rank, self.world_size,
                                               seed=self.opt.seed, num_workers=opt.num_workers,
                                               epoch=epoch, buffer_size=opt.buffer_size, split_even=True,
                                               dataset_ids=opt.train_sets)

        epoch_iterator = data_iterator.next_epoch_itr(not streaming, pin_memory=opt.pin_memory)

        total_tokens, total_loss, total_words = zero_tensor(), zero_tensor(), zero_tensor()
        total_non_pads = zero_tensor()
        report_loss, report_tgt_words = zero_tensor(), zero_tensor()
        report_src_words = zero_tensor()
        report_sents = zero_tensor()

        start = time.time()
        n_samples = len(data_iterator)

        counter = 0
        num_accumulated_words = zero_tensor()
        num_accumulated_sents = zero_tensor()
        report_contrastive_loss = zero_tensor()

        i = data_iterator.iterations_in_epoch if not is_list(train_data) else epoch_iterator.n_yielded
        i = i * self.world_size

        while not data_iterator.end_of_epoch():

            # curriculum = (epoch < opt.curriculum)

            samples = next(epoch_iterator)

            batch = prepare_sample(samples, device=self.device)
            targets = batch.get('target_output')

            # TODO: dealing with oom during distributed training
            oom = zero_tensor()
            counter = counter + 1
            reduce = True if counter >= opt.update_frequency or i == (n_samples - 1) else False

            try:
                def maybe_no_sync():
                    if not reduce and isinstance(self.model, DDP_model):
                        return self.model.no_sync()
                    else:
                        # when we dont reach the updating step, we do not need to synchronize the gradients
                        # thus disabling the backward grad sync to improve speed
                        return contextlib.ExitStack()  # dummy contextmanager

                with maybe_no_sync():
                    with autocast(enabled=opt.fp16):
                        output_dict = self.model(batch)
                        acoustic_features = output_dict['acoustic_features']
                        text_features = output_dict['text_features']

                        batch_size = batch.size
                        # outputs is a dictionary containing keys/values necessary for loss function
                        # can be flexibly controlled within models for easier extensibility

                        loss = self.loss_function(acoustic_features, text_features)
                        loss_data = loss.item()
                        full_loss = loss

                        optimizer = self.optim.optimizer

                    # TODO for adversarial:
                    grad_list = [p for p in self.model.parameters() if p.requires_grad]
                    model_input = None
                    vanilla_logits = None

                    # grad scaler has to be done outside of the autocast
                    self.grad_scaler.scale(full_loss).backward()

                    # del outputs
            #
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('[WARNING]: ran out of memory on GPU %d' % self.rank, flush=True)
                    print('Input size at OOM position:',
                          batch.get('source').size() if batch.get('source') is not None else None,
                          batch.get('target').size() if batch.get('target') is not None else None)

                    # continue
                    raise e
                    # recovering mechanism doesn't work at the moment
                    # loss = 0
                    # for p in self.model.parameters():
                    #     if p.grad is not None:
                    #         del p.grad  # free some memory
                    #     loss = loss + p.sum() * 0

                    # torch.cuda.empty_cache()
                    #
                    # if opt.streaming:  # reset stream in this case ...
                    #     streaming_state = self.model.init_stream()
                    #
                    #
                    # # backward to actually free the graph
                    # # self.grad_scaler.scale(loss).backward()
                    # oom.add_(1)

                raise e

            # connecting the oom signal from different gpus
            # self.all_reduce(oom, op=dist.ReduceOp.SUM, group=self.group)
            # # if OOM: all gpus reset grad and reset counter
            # # or maybe all-reduce grad?
            # if oom.item() > 0:
            #     # reset counter
            #     self.model.zero_grad()
            #     self.optim.zero_grad()
            #     counter = 0
            #     oom.zero_()

            batch_size = batch.size

            src_size = batch.src_size
            tgt_size = batch.tgt_size
            num_accumulated_words.add_(tgt_size)
            num_accumulated_sents.add_(batch_size)

            # We only update the parameters after getting gradients from n mini-batches
            update_flag = reduce

            if update_flag:

                # accumulated gradient case, in this case the update frequency
                self.all_reduce(num_accumulated_words, op=dist.ReduceOp.SUM, group=self.group)
                self.all_reduce(num_accumulated_sents, op=dist.ReduceOp.SUM, group=self.group)

                grad_denom = 1.0

                self.grad_scaler.unscale_(self.optim.optimizer)

                if self.opt.normalize_gradient:
                    grad_denom = num_accumulated_words.item() * grad_denom
                else:
                    grad_denom = 1

                # When we accumulate the gradients, each gradient is already normalized by a constant grad_scaler
                if grad_denom != 1:
                    normalize_gradients(self.model.parameters(), grad_denom)

                # Update the pagrameters.
                grad_norm = clip_grad_norm(self.model.parameters(), self.opt.max_grad_norm)

                self.optim.step(scaler=self.grad_scaler)
                self.grad_scaler.update()
                self.optim.zero_grad(set_to_none=opt.true_zero_grad)
                counter = 0
                num_accumulated_words.zero_()
                num_accumulated_sents.zero_()

                num_updates = self.optim._step
                if (opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every) \
                        or (num_updates >= opt.max_step):
                    valid_loss = self.eval(valid_data)
                    valid_ppl = math.exp(min(valid_loss, 100))

                    if self.is_main():
                        print('Validation perplexity: %g' % valid_ppl)
                        ep = float(epoch) - 1. + ((float(i) + 1.) / n_samples)
                        value = valid_ppl
                        self.save(ep, value,
                                  itr=data_iterator)

                    if num_updates >= opt.max_step:
                        print('[INFO] Max-training-step reached.')
                        exit(0)

            num_words = tgt_size
            report_loss.add_(loss_data)
            report_tgt_words.add_(batch_size * 2)
            report_src_words.add_(src_size)
            total_loss.add_(loss_data)
            total_words.add_(num_words)
            report_sents.add_(1)
            # total_tokens += batch.get('target_output').nelement()
            # total_non_pads += batch.get('target_output').ne(onmt.constants.PAD).sum().item()
            # batch_efficiency = total_non_pads / total_tokens

            # control the index a little bit to ensure the log is always printed
            if i == 0 or ((i + 1) % opt.log_interval < self.world_size):

                self.all_reduce(report_loss, op=dist.ReduceOp.SUM, group=self.group)
                self.all_reduce(report_tgt_words, op=dist.ReduceOp.SUM, group=self.group)
                self.all_reduce(report_src_words, op=dist.ReduceOp.SUM, group=self.group)

                if self.is_main():
                    log_string = ("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; grad_norm: %6.4f " %
                                  (epoch, i + 1, len(data_iterator),
                                   math.exp(report_loss.item() / report_tgt_words.item()),
                                   grad_norm))

                    log_string += ("lr: %.7f ; updates: %7d; " %
                                   (self.optim.get_learning_rate(),
                                    self.optim._step))

                    log_string += ("%5.0f src tok/s; %5.0f tgt tok/s; " %
                                   (report_src_words.item() / (time.time() - start),
                                    report_tgt_words.item() / (time.time() - start)))

                    log_string += ("%s elapsed" %
                                   str(datetime.timedelta(seconds=int(time.time() - self.start_time))))

                    self.print(log_string, flush=True)
                #
                report_loss.zero_()
                report_tgt_words.zero_()
                report_src_words.zero_()
                # report_sents.zero_()
                start = time.time()

            # increase i by world size
            i = i + self.world_size

        return total_loss / total_words

    def run(self, train_data=None, valid_data=None, checkpoint=None):
        opt = self.opt

        if checkpoint is not None:

            # TODO: have loading checkpoints for each process
            prec_opt = checkpoint['opt'] if 'opt' in checkpoint else None

            if not opt.reset_optim:

                itr_progress = None

                resume = True
                start_epoch = math.floor(checkpoint['epoch']) + 1 if 'epoch' in checkpoint else 1
                if start_epoch is None:
                    start_epoch = 1
            else:
                itr_progress = None
                resume = False
                start_epoch = 1

            # optim_state_dict = checkpoint['optim']
            # # del checkpoint['optim']
            del checkpoint
        else:
            itr_progress = None

            resume = False
            start_epoch = 1

        if opt.load_encoder_from:
            self.load_encoder_weight(opt.load_encoder_from)
        #
        if opt.load_decoder_from:
            self.load_decoder_weight(opt.load_decoder_from)

        if opt.estimate_fisher_information:
            self.start_time = time.time()
            self.estimate_fisher(train_data)
            return

        if opt.run_validation_before_training or opt.max_step <= 0:
            valid_loss, valid_accuracy = self.eval(valid_data)
            valid_ppl = math.exp(min(valid_loss, 100))

            if self.is_main():
                print('[INFO] Validation perplexity: %g' % valid_ppl, flush=True)
                # percent is never used in plural :)
                print('[INFO] Validation accuracy: %g percent' % (100 * valid_accuracy))

            if opt.max_step <= 0:
                if self.is_main():
                    self.save(0, valid_ppl if opt.save_metrics in ['ppl', 'perplexity'] else 1 - valid_accuracy)

                return

        self.start_time = time.time()

        for epoch in range(start_epoch, start_epoch + opt.epochs):
            self.print('')

            #  (1) train for one epoch on the training set
            train_loss = self.train_epoch(train_data, valid_data, epoch,
                                          resume=resume, itr_progress=itr_progress)
            train_ppl = math.exp(min(train_loss, 100))
            self.print('[INFO] Train perplexity: %g' % train_ppl)

            #  (2) evaluate on the validation set
            valid_loss = self.eval(valid_data)
            valid_ppl = math.exp(min(valid_loss, 100))

            if self.is_main():
                print('[INFO] Validation perplexity: %g' % valid_ppl)
                print('[INFO] Validation accuracy: %g percent' % (100 * valid_accuracy))

                value = valid_ppl
                self.save(epoch, value)

            itr_progress = None
            resume = False
