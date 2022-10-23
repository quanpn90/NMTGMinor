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
import numpy as np

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

# ignore the pytorch -> numpy conversion warnings
warnings.filterwarnings("ignore", category=UserWarning)

import quadprog

from .mp_trainer import prepare_sample, generate_data_iterator, zero_tensor, Trainer


def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp:
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp:
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).to(gradient.device).view(-1, 1))

def is_factorized_param(p):

    if p.endswith("r_i") or p.endswith("s_i"):
        return True

    if p.endswith("rm_i") or p.endswith("rm_o"):
        return True

    if p.endswith("sm_i") or p.endswith("sm_o"):
        return True

    if p.endswith("r_o") or p.endswith("s_o"):
        return True

    if p.endswith("r_p") or p.endswith("s_p"):
        return True

    if p.endswith("rm_p") or p.endswith("sm_p"):
        return True

    if p.endswith("r_q") or p.endswith("s_q") or p.endswith("r_kv") or p.endswith("s_kv"):
        return True

    if p.endswith("rm_q") or p.endswith("sm_q") or p.endswith("rm_kv") or p.endswith("sm_kv"):
        return True

    return False




class GEMTrainer(Trainer):

    def __init__(self, device, train_data, valid_data, dicts, opt, constants=None, setup_optimizer=True):
        """
        :param model:
        :param device: int (GPU id)
        :param loss_function:
        :param train_data:
        :param valid_data:
        :param dicts:
        :param opt:
        """

        super(GEMTrainer, self).__init__(device, train_data, valid_data, dicts, opt,
                                         constants=constants, setup_optimizer=setup_optimizer)

        assert  isinstance(train_data, list)
        assert  isinstance(valid_data, list)

        assert(len(opt.train_sets) > 0)
        assert(len(opt.train_set_orders) > 0)
        assert(len(opt.train_set_orders) == len(opt.train_sets)), "The number of train sets and the number of orders must match"

        self.print("[INFO] Preparing parameters for Gradient Episodic Memory")

        self.gem_params = list()
        self.gem_param_names = list()
        self.gem_param_size = list()
        self.ft_params = list()

        for n, p in self.model.named_parameters():
            if is_factorized_param(n):
                self.ft_params.append(n)
            else:
                if p.requires_grad:
                    self.gem_params.append(p)
                    self.gem_param_names.append(n)
                    self.gem_param_size.append(p.numel())

        self.print("[INFO] Done Preparing parameters.")



        # print out the stuff
        # for (gem_param, gem_param_name, gem_param_size) in zip(self.gem_params, self.gem_param_names, self.gem_param_size):
        #     print(gem_param_name, gem_param_size)
        # exit()

        self.orders = dict()

        for order, train_set in zip(opt.train_set_orders, opt.train_sets):
            if order not in self.orders:
                self.orders[order] = list()

            self.orders[order].append(train_set)

        memory_size = len(self.orders)
        self.grads = torch.Tensor(sum(self.gem_param_size), memory_size).cuda()

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
        if opt.load_pretrained_classifier:
            self.classifier.eval()

        total_loss = zero_tensor()
        total_words = zero_tensor()
        total_correct = zero_tensor()

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

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

                            if opt.load_pretrained_classifier:
                                layer_states = self.classifier.encode(batch)
                            else:
                                layer_states = None

                            outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
                                                 mirror=opt.mirror_loss, streaming_state=streaming_state, nce=opt.nce,
                                                 pretrained_layer_states=layer_states)

                            outputs['tgt_mask'] = tgt_mask
                            loss_dict = self.loss_function(outputs, targets, model=self.model, eval=True)
                            loss_data = loss_dict['data']
                            correct, total = loss_dict['correct'], loss_dict['total']

                            # if total != batch.tgt_size:
                            #     # print(batch.get('target').size())
                            #     # print(batch.get('target_output').size())
                            #     targets = batch.get('target_output')
                            #     targets_ = targets.view(-1)
                            #     non_pad_mask = torch.nonzero(targets_.ne(self.loss_function.padding_idx)).squeeze(1)
                            #     labels = targets_.index_select(0, non_pad_mask)
                            #     print(labels, labels.numel(), batch.tgt_size)

                            assert (total == batch.tgt_size), \
                                "Process %i, Minibatch %d/%d: Expected %d tokens from the batch, got %d" \
                                % (self.rank, i, data_size, batch.tgt_size, total)

                            # print(i, len(data_iterator), total, batch.tgt_size, loss_data)

                    total_loss.add_(loss_data)
                    total_words.add_(batch.tgt_size)
                    total_correct.add_(correct)
                    i = i + 1

        # allreduce the total loss and total words from other processes
        self.all_reduce(total_loss, op=dist.ReduceOp.SUM, group=self.group)
        self.all_reduce(total_words, op=dist.ReduceOp.SUM, group=self.group)
        self.all_reduce(total_correct, op=dist.ReduceOp.SUM, group=self.group)

        self.model.train()
        self.loss_function.train()
        if opt.load_pretrained_classifier:
            self.classifier.train()

        return total_loss.item() / total_words.item(), total_correct.item() / total_words.item()

    def train_epoch(self, epoch, resume=False, itr_progress=None):

        opt = self.opt
        train_data = self.train_data
        streaming = opt.streaming
        grad_norm = -1
        memory_size = len(self.orders)

        # Clear the gradients of the model
        self.optim.zero_grad(set_to_none=opt.true_zero_grad)
        # self.model.module.reset_states()

        # note: for Training split_even=True
        dataset = train_data

        data_iterators = dict()

        for order in self.orders:
            # self.orders[order] contains the list of training datasets for order
            # [0] is by default the currently (newest) added datasets

            data_iterators[order] = generate_data_iterator(dataset, self.rank, self.world_size,
                                                   seed=self.opt.seed, num_workers=opt.num_workers,
                                                   epoch=epoch, buffer_size=opt.buffer_size, split_even=True,
                                                   dataset_ids=self.orders[order])

        data_iterator = data_iterators[0]

        epoch_iterators = dict()
        for order in self.orders:
            # for the memory datasets, allow for reset_
            reset_ = order != 0
            epoch_iterators[order] = data_iterators[order].next_epoch_itr(not streaming, pin_memory=opt.pin_memory)

        epoch_iterator = epoch_iterators[0]

        total_tokens, total_loss, total_words = zero_tensor(), zero_tensor(), zero_tensor()
        total_non_pads = zero_tensor()
        report_loss, report_tgt_words = zero_tensor(), zero_tensor()
        report_ctc_loss = zero_tensor()
        report_src_words = zero_tensor()
        report_sents = zero_tensor()
        report_rec_loss, report_rev_loss, report_mirror_loss = zero_tensor(), zero_tensor(), zero_tensor()
        start = time.time()
        n_samples = len(data_iterator)

        counter = 0
        num_accumulated_words = zero_tensor()
        num_accumulated_sents = zero_tensor()
        report_contrastive_loss = zero_tensor()

        streaming_state = None

        i = data_iterator.iterations_in_epoch if not isinstance(train_data, list) else epoch_iterator.n_yielded
        i = i * self.world_size

        while not data_iterator.end_of_epoch():
            self.grads.zero_()

            # TODO: Sampling samples from the memory datasets
            for t in self.orders:
                self.optim.zero_grad(set_to_none=opt.true_zero_grad)

                if t == 0:
                    continue

                memory_data_iterator = epoch_iterators[t]
                if not memory_data_iterator.has_next():
                    # reset
                    epoch_iterators[t] = data_iterators[order].next_epoch_itr(not streaming, pin_memory=opt.pin_memory)

                memory_data_iterator = epoch_iterators[t]

                prev_samples = next(memory_data_iterator)
                batch = prepare_sample(prev_samples, device=self.device)
                targets = batch.get('target_output')
                streaming_state = None

                with autocast(enabled=opt.fp16):
                    tgt_mask = targets.ne(onmt.constants.PAD)
                    if opt.load_pretrained_classifier:
                        with torch.no_grad():
                            layer_states = self.classifier.encode(batch)
                    else:
                        layer_states = None

                    outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
                                         zero_encoder=opt.zero_encoder,
                                         mirror=opt.mirror_loss, streaming_state=streaming_state,
                                         nce=opt.nce, pretrained_layer_states=layer_states,
                                         adv_ptb_grad=opt.virtual_adversarial_training_mode > 0,
                                         checkpointing_ffn=opt.checkpointing_ffn,
                                         checkpointing_cross_attn=opt.checkpointing_cross_attn,
                                         checkpointing_self_attn=opt.checkpointing_self_attn
                                         )

                    outputs['tgt_mask'] = tgt_mask

                    loss_dict = self.loss_function(outputs, targets, model=self.model)
                    loss_data = loss_dict['data']
                    loss = loss_dict['loss']  # a little trick to avoid gradient overflow with fp16
                    full_loss = loss

                    rev_loss_data = None
                    mirror_loss_data = 0
                    rec_loss_data = None

                    correct, total = loss_dict['correct'], loss_dict['total']
                    optimizer = self.optim.optimizer

                # backward to get gradients (and synchronize between gpus)
                self.grad_scaler.scale(full_loss).backward()
                self.grad_scaler.unscale_(self.optim.optimizer)
                store_grad(self.gem_params, self.grads, self.gem_param_size, order)
                self.optim.optimizer.step(fake=True)
                # self.grad_scaler.update()
                # self.grad_scaler.step(self.optim.optimizer)
                self.grad_scaler.update()

            self.optim.zero_grad(set_to_none=opt.true_zero_grad)
            # zero model grads
            # forward and backward pass
            # synchronize the gradients and scale !!!!
            # put them in the grads
            # zero model grads

            samples = next(epoch_iterator)

            batch = prepare_sample(samples, device=self.device)
            targets = batch.get('target_output')
            streaming_state = None

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

                        tgt_mask = targets.ne(onmt.constants.PAD)
                        if opt.load_pretrained_classifier:
                            with torch.no_grad():
                                layer_states = self.classifier.encode(batch)
                        else:
                            layer_states = None

                        outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
                                             zero_encoder=opt.zero_encoder,
                                             mirror=opt.mirror_loss, streaming_state=streaming_state,
                                             nce=opt.nce, pretrained_layer_states=layer_states,
                                             adv_ptb_grad=opt.virtual_adversarial_training_mode > 0,
                                             checkpointing_ffn=opt.checkpointing_ffn,
                                             checkpointing_cross_attn=opt.checkpointing_cross_attn,
                                             checkpointing_self_attn=opt.checkpointing_self_attn
                                             )

                        batch_size = batch.size
                        # outputs is a dictionary containing keys/values necessary for loss function
                        # can be flexibly controlled within models for easier extensibility
                        outputs['tgt_mask'] = tgt_mask

                        loss_dict = self.loss_function(outputs, targets, model=self.model)
                        loss_data = loss_dict['data']
                        loss = loss_dict['loss']  # a little trick to avoid gradient overflow with fp16
                        full_loss = loss

                        if opt.ctc_loss > 0.0:
                            ctc_loss = self.ctc_loss_function(outputs, targets)
                            ctc_loss_data = ctc_loss.item()
                            full_loss = full_loss + opt.ctc_loss * ctc_loss

                        rev_loss_data = None
                        mirror_loss_data = 0
                        rec_loss_data = None

                        correct, total = loss_dict['correct'], loss_dict['total']
                        optimizer = self.optim.optimizer

                    grad_list = [p for p in self.model.parameters() if p.requires_grad]
                    model_input = None
                    vanilla_logits = None

                    # grad scaler has to be done outside of the autocast
                    self.grad_scaler.scale(full_loss).backward(inputs=grad_list)

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('[WARNING]: ran out of memory on GPU %d' % self.rank, flush=True)
                    print('Input size at OOM position:', batch.get('source').size(),
                          batch.get('target').size())
                    raise e
                    loss = 0

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

                grad_denom = 1.0

                self.grad_scaler.unscale_(self.optim.optimizer)

                if self.opt.normalize_gradient:
                    grad_denom = num_accumulated_words.item() * grad_denom

                # the gradient is scaled by world size, so in order to match the model without multiGPU
                # we rescale the model parameters w.r.t the world size
                # grad_denom = grad_denom / self.world_size

                # When we accumulate the gradients, each gradient is already normalized by a constant grad_scaler
                if grad_denom != 1:
                    normalize_gradients(self.model.parameters(), grad_denom)

                # Update the pagrameters.
                # grad_norm = clip_grad_norm(self.model.parameters(), self.opt.max_grad_norm)

                with torch.no_grad():
                    t = 0
                    store_grad(self.gem_params, self.grads, self.gem_param_size, t)
                    indx = torch.arange(1, len(self.orders), device=self.gem_params[0].device)

                    dotp = torch.mm(self.grads[:, 0].unsqueeze(0),
                                    self.grads.index_select(1, indx))

                    self.margin = 0.5
                    if (dotp < 0).sum() != 0:
                        project2cone2(self.grads[:, t].unsqueeze(1),
                                      self.grads.index_select(1, indx), self.margin)

                        overwrite_grad(self.gem_params, self.grads[:, t],
                                       self.gem_param_size)

                self.optim.step(scaler=self.grad_scaler)
                self.grad_scaler.update()
                self.optim.zero_grad(set_to_none=opt.true_zero_grad)
                counter = 0
                num_accumulated_words.zero_()
                num_accumulated_sents.zero_()

                num_updates = self.optim._step
                if (opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every) \
                        or (num_updates >= opt.max_step):
                    valid_loss, valid_accuracy = self.eval(self.valid_data)
                    valid_ppl = math.exp(min(valid_loss, 100))

                    if self.is_main():
                        print('Validation perplexity: %g' % valid_ppl)
                        print('Validation accuracy: %g percent' % (100 * valid_accuracy))
                        ep = float(epoch) - 1. + ((float(i) + 1.) / n_samples)
                        self.save(ep, valid_ppl if opt.save_metrics in ['ppl', 'perplexity'] else 1 - valid_accuracy,
                                  itr=data_iterator)

                    if num_updates >= opt.max_step:
                        print('[INFO] Max-training-step reached.')
                        exit(0)

            num_words = tgt_size
            report_loss.add_(loss_data)
            report_tgt_words.add_(num_words)
            report_src_words.add_(src_size)
            total_loss.add_(loss_data)
            total_words.add_(num_words)
            report_sents.add_(1)
            # total_tokens += batch.get('target_output').nelement()
            # total_non_pads += batch.get('target_output').ne(onmt.constants.PAD).sum().item()
            # batch_efficiency = total_non_pads / total_tokens

            if opt.reconstruct:
                report_rec_loss.add_(rec_loss_data)

            if opt.mirror_loss:
                report_rev_loss.add_(rev_loss_data)
                report_mirror_loss.add_(mirror_loss_data)

            if opt.ctc_loss > 0.0:
                report_ctc_loss.add_(ctc_loss_data)

            # control the index a little bit to ensure the log is always printed
            if i == 0 or ((i + 1) % opt.log_interval < self.world_size):

                self.all_reduce(report_loss, op=dist.ReduceOp.SUM, group=self.group)
                self.all_reduce(report_tgt_words, op=dist.ReduceOp.SUM, group=self.group)
                self.all_reduce(report_src_words, op=dist.ReduceOp.SUM, group=self.group)
                # self.all_reduce(report_sents, op=dist.ReduceOp.SUM, group=self.group)
                # self.all_reduce(report_contrastive_loss, op=dist.ReduceOp.SUM, group=self.group)

                if self.is_main():
                    log_string = ("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; grad_norm: %6.4f " %
                                  (epoch, i + 1, len(data_iterator),
                                   math.exp(report_loss.item() / report_tgt_words.item()),
                                   grad_norm))

                    if opt.mirror_loss:
                        self.all_reduce(report_rev_loss, op=dist.ReduceOp.SUM, group=self.group)
                        rev_ppl = math.exp(report_rev_loss.item() / report_tgt_words.item())
                        log_string += (" rev_ppl: %6.2f ; " % rev_ppl)
                        log_string += (" mir_loss: %6.2f ; " % (report_mirror_loss / report_tgt_words))

                    if opt.ctc_loss > 0.0:
                        # if torch.isinf(report_ctc_loss):
                        #     report_ctc_loss.zero_()
                        # self.all_reduce(report_ctc_loss, op=dist.ReduceOp.SUM, group=self.group)
                        ctc_loss = report_ctc_loss.item() / report_tgt_words.item()
                        log_string += (" ctcloss: %8.2f ; " % ctc_loss)

                    if opt.contrastive_loss_coeff > 0.0:
                        #
                        ctv_loss = report_contrastive_loss.item() / report_tgt_words.item()
                        log_string += (" ctv_loss: %8.2f ; " % ctv_loss)

                    log_string += ("lr: %.7f ; updates: %7d; " %
                                   (self.optim.get_learning_rate(),
                                    self.optim._step))

                    log_string += ("%5.0f src tok/s; %5.0f tgt tok/s; " %
                                   (report_src_words.item() / (time.time() - start),
                                    report_tgt_words.item() / (time.time() - start)))

                    log_string += ("%s elapsed" %
                                   str(datetime.timedelta(seconds=int(time.time() - self.start_time))))

                    self.print(log_string, flush=True)

                report_loss.zero_()
                report_tgt_words.zero_()
                report_src_words.zero_()
                report_rec_loss.zero_()
                report_rev_loss.zero_()
                report_mirror_loss.zero_()
                report_ctc_loss.zero_()
                # report_sents.zero_()
                if report_contrastive_loss is not None:
                    report_contrastive_loss.zero_()
                start = time.time()

            # increase i by world size
            i = i + self.world_size

        return total_loss / total_words

    def run(self, checkpoint=None):

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

        # if we are on a GPU: warm up the memory allocator
        if self.cuda:
            self.warm_up()

        if opt.estimate_fisher_information:
            self.start_time = time.time()
            self.estimate_fisher(self.train_data)
            return

        if opt.run_validation_before_training or opt.max_step <= 0:
            valid_loss, valid_accuracy = self.eval(self.valid_data)
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
            train_loss = self.train_epoch(epoch, resume=resume, itr_progress=itr_progress)
            train_ppl = math.exp(min(train_loss, 100))
            self.print('[INFO] Train perplexity: %g' % train_ppl)

            #  (2) evaluate on the validation set
            valid_loss, valid_accuracy = self.eval(self.valid_data)
            valid_ppl = math.exp(min(valid_loss, 100))

            if self.is_main():
                print('[INFO] Validation perplexity: %g' % valid_ppl)
                print('[INFO] Validation accuracy: %g percent' % (100 * valid_accuracy))
                self.save(epoch, valid_ppl if opt.save_metrics in ['ppl', 'perplexity'] else 1 - valid_accuracy)

            itr_progress = None
            resume = False







