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
from multiprocessing.managers import ListProxy as ListProxy

# ignore the pytorch -> numpy conversion warnings
warnings.filterwarnings("ignore", category=UserWarning)


def prepare_sample(batch, device=None):
    """
    Put minibatch on the corresponding GPU
    :param batch:
    :param device:
    :return:
    """
    if isinstance(batch, list):
        batch = batch[0]
    batch = rewrap(batch)
    batch.cuda(fp16=False, device=device)

    return batch

def is_list(object):
    if isinstance(object, list):
        return True

    elif isinstance(object, ListProxy):
        return True

    return False


def generate_data_iterator(dataset, rank, world_size, seed,
                           num_workers=1, epoch=1., buffer_size=0, split_even=True,
                           dataset_ids=None):
    # check if dataset is a list:
    if is_list(dataset):
        # this is a multidataset
        data_iterator = MultiDataIterator(dataset, seed=seed, num_workers=num_workers,
                                          epoch=epoch, buffer_size=buffer_size,
                                          num_shards=world_size, shard_id=rank, split_even=split_even,
                                          dataset_ids=dataset_ids)
    else:
        data_iterator = DataIterator(dataset, dataset.get_collater(), dataset.get_batches(), seed=seed,
                                     num_workers=num_workers, epoch=epoch, buffer_size=buffer_size,
                                     num_shards=world_size, shard_id=rank, split_even=split_even)

    return data_iterator


def zero_tensor(device=None):
    if device is None:
        return torch.Tensor([0]).cuda()
    else:
        return torch.Tensor([0]).to(device)


def all_reduce_and_rescale_tensors(tensors, rescale_denom=1,
                                   buffer_size=10485760):
    """All-reduce and rescale tensors in chunks of the specified size.
    Args:
        tensors: list of Tensors to all-reduce
        rescale_denom: denominator for rescaling summed Tensors
        buffer_size: all-reduce chunk size in bytes
    """
    # buffer size in bytes, determine equiv. # of elements based on data type
    buffer_t = tensors[0].new(
        math.ceil(buffer_size / tensors[0].element_size())).zero_()
    buffer = []

    def all_reduce_buffer():
        # copy tensors into buffer_t
        offset = 0
        for t in buffer:
            numel = t.numel()
            buffer_t[offset:offset + numel].copy_(t.view(-1))
            offset += numel

        # all-reduce and rescale
        torch.distributed.all_reduce(buffer_t[:offset])
        buffer_t.div_(rescale_denom)

        # copy all-reduced buffer back into tensors
        offset = 0
        for t in buffer:
            numel = t.numel()
            t.view(-1).copy_(buffer_t[offset:offset + numel])
            offset += numel

    with torch.no_grad():
        filled = 0
        for t in tensors:
            sz = t.numel() * t.element_size()
            if sz > buffer_size:
                # tensor is bigger than buffer, all-reduce and rescale directly
                torch.distributed.all_reduce(t)
                t.div_(rescale_denom)
            elif filled + sz > buffer_size:
                # buffer is full, all-reduce and replace buffer with grad
                all_reduce_buffer()
                buffer = [t]
                filled = sz
            else:
                # add tensor to buffer
                buffer.append(t)
                filled += sz

        if len(buffer) > 0:
            all_reduce_buffer()


class Trainer(object):

    # def __init__(self, device, train_data, valid_data, dicts, opt, constants=None, setup_optimizer=True):
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
        tgt_pad = dicts['tgt_pad']

        if opt.ctc_loss > 0.0:
            from onmt.speech.ctc_loss import CTC
            self.ctc_loss_function = CTC(dicts['tgt'].size(), opt.model_size, 0.0, reduce=True,
                                         padding_idx=tgt_pad, blank_idx=0)

        if opt.predict_language:
            from onmt.models.speech_recognizer.lid_loss import CrossEntropyLIDLoss
            self.lid_loss_function = CrossEntropyLIDLoss(opt.n_languages, label_smoothing=0.0)

        if opt.nce:
            from onmt.modules.nce.nce_loss import NCELoss
            loss_function = NCELoss(opt.model_size, dicts['tgt'].size(), noise_ratio=opt.nce_noise,
                                    logz=9, label_smoothing=opt.label_smoothing)
        else:
            loss_function = NMTLossFunc(opt.model_size, dicts['tgt'].size(),
                                        label_smoothing=opt.label_smoothing,
                                        mirror=opt.mirror_loss,
                                        padding_idx=tgt_pad)

        # This function replaces modules with the more optimized counterparts so that it can run faster
        # Currently exp with LayerNorm

        # distributed is required to convert BatchNorm to SyncBatchNorm for DDP
        optimize_model(model, distributed=(self.world_size > 1))

        if opt.load_pretrained_classifier:
            from onmt.model_factory import build_classifier
            self.print("Loading pretrained external classifier ...", flush=True)
            classifier_checkpoint = torch.load(opt.load_pretrained_classifier,
                                               map_location=lambda storage, loc: storage)
            classifier_opt = classifier_checkpoint['opt']
            classifier_dicts = classifier_checkpoint['dicts']
            self.classifier = build_classifier(classifier_opt, classifier_dicts)
            self.classifier.load_state_dict(classifier_checkpoint['model'])

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

            # Ensure that the distributed copies have the same initial parameters
            # Manual seed may not work the same for different GPU models.
            # if self.world_size > 1:
            #     params = [p for p in self.model.parameters()]
            #
            #     with torch.no_grad():
            #         if not self.is_main():
            #             # zero everything except for the main model
            #             for p in params:
            #                 p.zero_()
            #         else:
            #             for p in params:
            #                 p.add_(0)

            # run all_reduce to ensure that all models have exactly the same parameters
            # if self.world_size > 1:
            #     params = [p for p in self.model.parameters()]
            #     all_reduce_and_rescale_tensors(params, 1)

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

        if opt.load_fisher:
            if self.is_main():
                print("[INFO] Loading fisher information from: %s" % opt.load_fisher)
            self.fisher_info = torch.load(opt.load_fisher, map_location=lambda storage, loc: storage)
            if self.cuda:
                for n in self.fisher_info['mean']:
                    self.fisher_info['mean'][n] = self.fisher_info['mean'][n].cuda()
                for n in self.fisher_info['fisher_diag']:
                    self.fisher_info['fisher_diag'][n] = self.fisher_info['fisher_diag'][n].cuda()
        else:
            self.fisher_info = None

        print("[INFO] Process %d ready." % self.rank, flush=True)

    def is_main(self):
        return self.rank == 0

    def all_reduce(self, tensor, **kwargs):
        if self.world_size > 1:
            dist.all_reduce(tensor, **kwargs)

        return

    def print(self, *content, flush=False):
        """
        A helper function to print only on the main process
        :param flush:
        :param content:
        :return:
        """

        if self.is_main():
            print(*content, flush=flush)
        else:
            return

    def load_encoder_weight(self, checkpoint_file, wav2vec=False):

        if not wav2vec:
            print("Loading pretrained Encoder Weights from %s" % checkpoint_file, flush=True)
            checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)

            pretrained_model = build_model(checkpoint['opt'], checkpoint['dicts'], False, self.constants)
            pretrained_model.load_state_dict(checkpoint['model'])

            model = self.model.module if self.world_size > 1 else self.model

            model.load_encoder_weights(pretrained_model)

        else:
            checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
            model = self.model.module if self.world_size > 1 else self.model
            model.load_encoder_weights(checkpoint)

        return

    def load_decoder_weight(self, checkpoint_file):

        self.print("Loading pretrained models from %s" % checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        chkpoint_dict = checkpoint['dicts']

        pretrained_model = build_model(checkpoint['opt'], chkpoint_dict, False, self.constants)
        pretrained_model.load_state_dict(checkpoint['model'])

        self.print("Loading pretrained decoder weights ...")
        # first we have to remove the embeddings which probably have difference size ...
        pretrained_word_emb = pretrained_model.decoder.word_lut
        pretrained_model.decoder.word_lut = None
        pretrained_lang_emb = pretrained_model.decoder.language_embeddings
        pretrained_model.decoder.language_embeddings = None

        # actually we assume that two decoders have the same language embeddings...
        untrained_word_emb = self.model.decoder.word_lut
        self.model.decoder.word_lut = None
        untrained_lang_emb = self.model.decoder.language_embeddings
        self.model.decoder.language_embeddings = None

        decoder_state_dict = pretrained_model.decoder.state_dict()
        self.model.decoder.load_state_dict(decoder_state_dict)

        # now we load the embeddings ....
        n_copies = 0
        for token in self.dicts['tgt'].labelToIdx:

            untrained_id = self.dicts['tgt'].labelToIdx[token]

            if token in chkpoint_dict['tgt'].labelToIdx:
                pretrained_id = chkpoint_dict['tgt'].labelToIdx[token]
                untrained_word_emb.weight.data[untrained_id].copy_(pretrained_word_emb.weight.data[pretrained_id])

                self.model.generator[0].linear.bias.data[untrained_id].copy_(pretrained_model
                                                                             .generator[0].linear.bias.data[
                                                                                 pretrained_id])
                n_copies += 1

        self.print("Copied embedding for %d words" % n_copies)
        self.model.decoder.word_lut = untrained_word_emb

        # now we load the language embeddings ...
        if pretrained_lang_emb and untrained_lang_emb and 'langs' in chkpoint_dict:
            for lang in self.dicts['langs']:

                untrained_id = self.dicts['langs'][lang]
                if lang in chkpoint_dict['langs']:
                    pretrained_id = chkpoint_dict['langs'][lang]
                    untrained_lang_emb.weight.data[untrained_id].copy_(pretrained_lang_emb.weight.data[pretrained_id])

        self.model.decoder.language_embeddings = untrained_lang_emb

    def warm_up(self, train_data):
        """
        Warmup the memory allocator, by attempting to fit the largest batch
        :return:
        """

        batch = train_data[0].get_largest_batch(bsz=-1, src_size=-1, tgt_size=-1) \
            if is_list(train_data) \
            else train_data.get_largest_batch(bsz=328, src_size=319520, tgt_size=18)
        opt = self.opt

        if self.cuda:
            batch.cuda(fp16=False)

        self.model.train()
        self.loss_function.train()

        loss = 0
        for p in self.model.parameters():
            loss = loss + p.sum() * 0

        # this will create zero grads
        loss.backward()
        # self.model.zero_grad()
        oom = False

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        # try:
        with autocast(enabled=opt.fp16):
            targets = batch.get('target_output')
            tgt_mask = None
            outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
                                 zero_encoder=opt.zero_encoder,
                                 mirror=opt.mirror_loss, streaming_state=streaming_state,
                                 nce=opt.nce, checkpointing_ffn=opt.checkpointing_ffn,
                                 checkpointing_cross_attn=opt.checkpointing_cross_attn,
                                 checkpointing_self_attn=opt.checkpointing_self_attn)

            outputs['tgt_mask'] = tgt_mask

            loss_dict = self.loss_function(outputs, targets, model=self.model)
            loss_data = loss_dict['data']
            loss = loss_dict['loss']  # a little trick to avoid gradient overflow with fp16
            full_loss = loss

            if opt.ctc_loss > 0.0:
                ctc_loss = self.ctc_loss_function(outputs, targets)
                ctc_loss_data = ctc_loss.item()
                full_loss = full_loss + opt.ctc_loss * ctc_loss

            if opt.mirror_loss:
                rev_loss = loss_dict['rev_loss']
                mirror_loss = loss_dict['mirror_loss']
                full_loss = full_loss + rev_loss + mirror_loss

            if opt.predict_lang:
                lid_loss = loss_dict['lid']
                full_loss = full_loss + lid_loss
                lid_loss_data = lid_loss.item()
            else:
                lid_loss_data = 0

            # reconstruction loss
            if opt.reconstruct:
                rec_loss = loss_dict['rec_loss']
                rec_loss = rec_loss
                full_loss = full_loss + rec_loss

            if opt.lfv_multilingual:
                lid_logits = outputs['lid_logits']
                lid_labels = batch.get('target_lang')
                lid_loss_function = self.loss_function.get_loss_function('lid_loss')
                lid_loss = lid_loss_function(lid_logits, lid_labels)
                full_loss = full_loss + lid_loss

            optimizer = self.optim.optimizer

        # Warning: self-defined parameter list
        parameter_list = [p for p in self.model.parameters() if p.requires_grad]
        # Later if we need to do Adversarial Perturbation:

        self.grad_scaler.scale(full_loss).backward()

        loss = 0

        for p in parameter_list:
            loss += p.sum() * 0.0

        loss.backward()

        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()
        # self.model.zero_grad()
        # self.optim.zero_grad()
        # self.optim.step()
        # self.optim.reset()

        # except RuntimeError as e:
        #     if 'out of memory' in str(e):
        #         oom = True
        #     # else:
        #         print("[INFO] Warning: out-of-memory in warming up. "
        #               "This is due to the largest batch is too big for the GPU.",
        #               flush=True)
        #         raise e
        # else:
        self.print("[INFO] Warming up successfully.", flush=True)

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

    def train_epoch(self, train_data, valid_data, epoch, resume=False, itr_progress=None):

        opt = self.opt
        streaming = opt.streaming
        grad_norm = -1

        # Clear the gradients of the model
        self.optim.zero_grad(set_to_none=opt.true_zero_grad)
        # self.model.module.reset_states()

        # note: for Training split_even=True
        dataset = train_data
        data_iterator = generate_data_iterator(dataset, self.rank, self.world_size,
                                               seed=self.opt.seed, num_workers=opt.num_workers,
                                               epoch=epoch, buffer_size=opt.buffer_size, split_even=True,
                                               dataset_ids=opt.train_sets)

        # TODO: fix resume which is currently buggy
        if resume:
            data_iterator.load_state_dict(itr_progress)

        epoch_iterator = data_iterator.next_epoch_itr(not streaming, pin_memory=opt.pin_memory)

        total_tokens, total_loss, total_words = zero_tensor(), zero_tensor(), zero_tensor()
        total_non_pads = zero_tensor()
        report_loss, report_tgt_words = zero_tensor(), zero_tensor()
        report_ctc_loss = zero_tensor()
        report_ewc_loss = zero_tensor()
        report_ewc_count = 0
        report_src_words = zero_tensor()
        report_sents = zero_tensor()
        report_rec_loss, report_rev_loss, report_mirror_loss = zero_tensor(), zero_tensor(), zero_tensor()
        start = time.time()
        n_samples = len(data_iterator)

        counter = 0
        num_accumulated_words = zero_tensor()
        num_accumulated_sents = zero_tensor()
        report_contrastive_loss = zero_tensor()

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        ewc_importance = opt.ewc_importance

        if ewc_importance > 0:
            assert self.fisher_info is not None
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                model = self.model.module
            else:
                model = self.model

            # parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
            parameters = dict()
            for n, p in model.named_parameters():
                if n in self.fisher_info['mean'] and p.requires_grad:
                    parameters[n] = p

        i = data_iterator.iterations_in_epoch if not is_list(train_data) else epoch_iterator.n_yielded
        i = i * self.world_size

        while not data_iterator.end_of_epoch():

            # curriculum = (epoch < opt.curriculum)

            # this batch generator is not very clean atm
            # TODO: move everything to the multiGPU trainer
            samples = next(epoch_iterator)

            batch = prepare_sample(samples, device=self.device)
            targets = batch.get('target_output')

            if opt.streaming:
                if train_data.is_new_stream():
                    streaming_state = self.model.init_stream()
            else:
                streaming_state = None

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

                        if opt.mirror_loss:
                            rev_loss = loss_dict['rev_loss']
                            rev_loss_data = loss_dict['rev_loss_data']
                            mirror_loss = loss_dict['mirror_loss']
                            full_loss = full_loss + rev_loss + mirror_loss
                            mirror_loss_data = loss_dict['mirror_loss'].item()
                        else:
                            rev_loss_data = None
                            mirror_loss_data = 0

                        if opt.predict_language:
                            enc_pred_lang = outputs['enc_pred_lang']
                            enc_mask = outputs['src_mask']
                            enc_lid_loss = self.lid_loss_function(enc_pred_lang, batch.get("source_lang"), enc_mask)

                            dec_pred_lang = outputs['dec_pred_lang']
                            dec_mask = outputs['target_mask']
                            dec_lid_loss = self.lid_loss_function(dec_pred_lang, batch.get("target_lang"), dec_mask)

                            full_loss = full_loss + 0.01 * (enc_lid_loss + dec_lid_loss)

                        else:
                            enc_lid_loss = None
                            enc_lid_loss_data = None
                            dec_lid_loss = None
                            dec_lid_loss_data = None

                        # reconstruction loss
                        if opt.reconstruct:
                            rec_loss = loss_dict['rec_loss']
                            rec_loss = rec_loss
                            full_loss = full_loss + rec_loss
                            rec_loss_data = loss_dict['rec_loss_data']
                        else:
                            rec_loss_data = None

                        if opt.contrastive_loss_coeff > 0 and 'contrastive_loss' in outputs:
                            contrastive_loss = outputs['contrastive_loss']
                            full_loss = full_loss + opt.contrastive_loss_coeff * contrastive_loss
                            report_contrastive_loss.add_(contrastive_loss.item())

                        correct, total = loss_dict['correct'], loss_dict['total']
                        optimizer = self.optim.optimizer

                    #         ewc_penalty = ewc_penalty + (torch.square(parameters[n] - self.fisher_info['mean'][n]) *
                    #                                      self.fisher_info['fisher_diag'][n]).sum()
                    #     full_loss += ewc_penalty * ewc_importance

                    # TODO for adversarial:
                    grad_list = [p for p in self.model.parameters() if p.requires_grad]
                    if opt.virtual_adversarial_training_mode > 0:
                        # if we use virtual adversarial training: add the input to the list of gradient to take
                        model_input = outputs['source']
                        vanilla_logits = outputs['logprobs']
                        grad_list += [model_input]
                    else:
                        model_input = None
                        vanilla_logits = None

                    # grad scaler has to be done outside of the autocast
                    self.grad_scaler.scale(full_loss).backward()

                    # del outputs
                    if opt.virtual_adversarial_training_mode > 0:
                        # run forward pass one more time
                        # the perturbation is the gradient of the model w.r.t the input
                        perturb = model_input.grad.data.new(*model_input.size()).copy_(model_input.grad.data)

                        with autocast(enabled=opt.fp16):
                            assert model_input.grad is not None
                            outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
                                                 pretrained_layer_states=layer_states,
                                                 input_ptb=perturb)

                            full_loss = None
                            # compute loss for mode 2 3
                            # In this mode, we add noise to the input and minimise the loss given the noisy inputs
                            if opt.virtual_adversarial_training_mode in [2, 3]:
                                loss_dict = self.loss_function(outputs, targets, model=self.model)
                                full_loss = loss_dict['loss']

                            # for mode 1, 3 compute kl divergence
                            # In this mode, we minimise the kl divergence between the model output with and without noise
                            if opt.virtual_adversarial_training_mode in [1, 3]:
                                logits = outputs['logprobs']

                                with torch.no_grad():
                                    vanilla_probs = \
                                        F.softmax(vanilla_logits.float().view(-1, vanilla_logits.size(-1)), dim=-1)
                                    vanilla_probs.detach_()
                                noisy_probs = F.softmax(logits.float().view(-1, logits.view(-1, logits.size(-1))),
                                                        dim=-1)

                                # Note: with the kl_div_loss we don't backward w.r.t the vanilla probs
                                kl_div_loss = F.kl_div(noisy_probs, vanilla_probs, reduction='sum')
                                if full_loss is None:
                                    full_loss = kl_div_loss
                                else:
                                    full_loss += kl_div_loss

                        # Now we only get the gradients for the weights of the network
                        grad_list = [p for p in self.model.parameters() if p.requires_grad]
                        self.grad_scaler.scale(full_loss).backward()
                        del outputs

                    # EWC training: no need for autograd here?
                    if self.optim._step % opt.ewc_decay_every == 0:

                        ewc_importance = ewc_importance / opt.ewc_decay_scale

                    # only run this ewc everytime we reduce

                # if isinstance(self.model, DDP_model):
                #     torch.cuda.synchronize(device=self.rank)

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('[WARNING]: ran out of memory on GPU %d' % self.rank, flush=True)
                    print('Input size at OOM position:', batch.get('source').size(),
                          batch.get('target').size())

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

                grad_denom = 1.0

                self.grad_scaler.unscale_(self.optim.optimizer)

                if self.opt.normalize_gradient:
                    grad_denom = num_accumulated_words.item() * grad_denom
                else:
                    grad_denom = 1

                # the gradient is scaled by world size, so in order to match the model without multiGPU
                # we rescale the model parameters w.r.t the world size
                # grad_denom = grad_denom / self.world_size

                # When we accumulate the gradients, each gradient is already normalized by a constant grad_scaler
                if grad_denom != 1:
                    normalize_gradients(self.model.parameters(), grad_denom)

                # Update the pagrameters.
                grad_norm = clip_grad_norm(self.model.parameters(), self.opt.max_grad_norm)

                if ewc_importance > 0:
                    ewc_penalty = 0

                    if self.optim._step >= opt.ewc_delay:
                        # if at the moment weights/gradients/mean and fisher_diag are all the same and unscaled
                        # then we don't need to synchronize the gradients
                        with self.model.no_sync():
                            for n, p in self.model.named_parameters():
                                if isinstance(self.model, DDP_model):
                                    n = n[len("module."):]
                                if n in self.fisher_info['mean']:
                                    penalty = self.fisher_info['fisher_diag'][n] * \
                                              torch.square(p - self.fisher_info['mean'][n].data)

                                    ewc_penalty = ewc_penalty + penalty.sum()

                            loss = ewc_penalty * ewc_importance
                            ewc_loss = ewc_penalty.item()
                            # accumulate the gradients from EWC loss
                            loss.backward()
                            report_ewc_loss.add_(ewc_loss)
                            report_ewc_count += 1

                self.optim.step(scaler=self.grad_scaler)
                self.grad_scaler.update()
                self.optim.zero_grad(set_to_none=opt.true_zero_grad)
                counter = 0
                num_accumulated_words.zero_()
                num_accumulated_sents.zero_()

                num_updates = self.optim._step
                if (opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every) \
                        or (num_updates >= opt.max_step):
                    valid_loss, valid_accuracy = self.eval(valid_data)
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
                # self.all_reduce(report_ewc_loss, op=dist.ReduceOp.SUM, group=self.group)
                self.all_reduce(report_tgt_words, op=dist.ReduceOp.SUM, group=self.group)
                self.all_reduce(report_src_words, op=dist.ReduceOp.SUM, group=self.group)
                # self.all_reduce(report_sents, op=dist.ReduceOp.SUM, group=self.group)
                # self.all_reduce(report_contrastive_loss, op=dist.ReduceOp.SUM, group=self.group)

                if self.is_main():
                    log_string = ("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; grad_norm: %6.4f " %
                                  (epoch, i + 1, len(data_iterator),
                                   math.exp(report_loss.item() / report_tgt_words.item()),
                                   grad_norm))

                    # if opt.reconstruct:
                    #     self.all_reduce(report_rec_loss, op=dist.ReduceOp.SUM, group=self.group)
                    #     rec_ppl = math.exp(report_rec_loss.item() / report_src_words.item())
                    #     log_string += (" rec_ppl: %6.2f ; " % rec_ppl)

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

                    if ewc_importance > 0.0:
                        try:
                            _ewc_loss = report_ewc_loss.item() / report_ewc_count
                        except ZeroDivisionError:
                            _ewc_loss =  float('nan')
                        log_string += (" ewcloss: %8.8f ; " % _ewc_loss)

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
                report_ewc_loss.zero_()
                report_ewc_count = 0
                # report_sents.zero_()
                if report_contrastive_loss is not None:
                    report_contrastive_loss.zero_()
                start = time.time()

            # increase i by world size
            i = i + self.world_size

        return total_loss / total_words

    def estimate_fisher(self, data):
        """
        This function estimates the Fisher Information (only diagonal) on a data

        :param data: train or dev data
        :return: fisher
        """

        def is_factorize_params(p_name):

            # feed forward neural net
            if p_name.endswith(".r_i") or p_name.endswith(".s_i") \
                    or p_name.endswith(".r_o") or p_name.endswith(".s_o") \
                    or p_name.endswith(".r_p") or p_name.endswith(".s_p"):
                return True

            if p_name.endswith(".r_q") or p_name.endswith(".s_q") \
                    or p_name.endswith(".r_o") or p_name.endswith(".s_o") \
                    or p_name.endswith(".r_kv") or p_name.endswith(".s_kv"):
                return True

            if p_name.endswith(".rm_q") or p_name.endswith(".sm_q") \
                    or p_name.endswith(".rm_o") or p_name.endswith(".sm_o") \
                    or p_name.endswith(".rm_kv") or p_name.endswith(".sm_kv"):
                return True

            if p_name.endswith(".sub_r_i") or p_name.endswith(".sub_s_i") \
                    or p_name.endswith(".sub_r_o") or p_name.endswith(".sub_s_o") \
                    or p_name.endswith(".sub_r_p") or p_name.endswith(".sub_s_p"):
                return True

            if p_name.endswith(".sub_r_q") or p_name.endswith(".sub_s_q") \
                    or p_name.endswith(".sub_r_o") or p_name.endswith(".sub_s_o") \
                    or p_name.endswith(".sub_r_kv") or p_name.endswith(".sub_s_kv"):
                return True

            if p_name.endswith(".sub_rm_q") or p_name.endswith(".sub_sm_q") \
                    or p_name.endswith(".sub_rm_o") or p_name.endswith(".sub_sm_o") \
                    or p_name.endswith(".sub_rm_kv") or p_name.endswith(".sub_sm_kv"):
                return True

            if p_name.endswith(".rm_i") or p_name.endswith(".sm_i") or \
                    p_name.endswith(".rm_o") or p_name.endswith(".sm_o") or \
                    p_name.endswith(".rm_p") or p_name.endswith(".sm_p"):
                return True

            if p_name.endswith(".sub_rm_i") or p_name.endswith(".sub_sm_i") or \
                    p_name.endswith(".sub_rm_o") or p_name.endswith(".sub_sm_o") or \
                    p_name.endswith(".sub_rm_p") or p_name.endswith(".sub_sm_p"):
                return True

            if "adapter" in p_name:
                return True

            return False

        if self.rank == 0:
            print("[INFO] Estimating fisher information ...\n")

        opt = self.opt
        epoch = 0
        assert len(opt.load_from) > 0

        # Clear the gradients of the model
        self.optim.zero_grad(set_to_none=False)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        else:
            model = self.model

        parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
        precision_matrices = dict()

        for n, p in parameters.items():
            if not is_factorize_params(n):
                precision_matrices[n] = torch.zeros_like(p)

        # note: for Training split_even=True
        dataset = data
        data_iterator = generate_data_iterator(dataset, self.rank, self.world_size,
                                               seed=self.opt.seed, num_workers=opt.num_workers,
                                               epoch=0, buffer_size=opt.buffer_size, split_even=True,
                                               dataset_ids=opt.train_sets)

        streaming = False
        epoch_iterator = data_iterator.next_epoch_itr(not streaming, pin_memory=opt.pin_memory)

        total_tokens, total_loss, total_words = zero_tensor(), zero_tensor(), zero_tensor()
        total_non_pads = zero_tensor()
        report_loss, report_tgt_words = zero_tensor(), zero_tensor()
        report_ctc_loss = zero_tensor()
        report_src_words = zero_tensor()
        report_rec_loss, report_rev_loss, report_mirror_loss = zero_tensor(), zero_tensor(), zero_tensor()
        start = time.time()
        n_samples = len(data_iterator)

        counter = 0
        num_accumulated_words = zero_tensor()
        num_accumulated_sents = zero_tensor()
        report_contrastive_loss = zero_tensor()

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        i = data_iterator.iterations_in_epoch if not is_list(dataset) else epoch_iterator.n_yielded
        i = i * self.world_size  # incorrect?
        self.model.train()  # eliminate dropout (is it necessary)?

        while not data_iterator.end_of_epoch():

            # this batch generator is not very clean atm
            # TODO: move everything to the multiGPU trainer
            samples = next(epoch_iterator)

            batch = prepare_sample(samples, device=self.device)
            targets = batch.get('target_output')

            if opt.streaming:
                if train_data.is_new_stream():
                    streaming_state = self.model.init_stream()
            else:
                streaming_state = None

            # TODO: dealing with oom during distributed training
            oom = zero_tensor()
            counter = counter + 1
            # reduce = True if counter >= opt.update_frequency or i == (n_samples - 1) else False
            reduce = False  # never reduce :))))

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

                        if opt.contrastive_loss_coeff > 0 and 'contrastive_loss' in outputs:
                            contrastive_loss = outputs['contrastive_loss']
                            full_loss = full_loss + opt.contrastive_loss_coeff * contrastive_loss
                            report_contrastive_loss.add_(contrastive_loss.item())

                        correct, total = loss_dict['correct'], loss_dict['total']
                        optimizer = self.optim.optimizer

                    # grad scaler has to be done outside of the autocast

                    # TODO for adversarial:

                    grad_list = [p for p in self.model.parameters() if p.requires_grad]
                    if opt.virtual_adversarial_training_mode > 0:
                        # if we use virtual adversarial training: add the input to the list of gradient to take
                        model_input = outputs['source']
                        vanilla_logits = outputs['logprobs']
                        grad_list += [model_input]
                    else:
                        model_input = None
                        vanilla_logits = None

                    self.grad_scaler.scale(full_loss).backward()

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('[WARNING]: ran out of memory on GPU %d' % self.rank, flush=True)
                    print('Input size at OOM position:', batch.get('source').size(),
                          batch.get('target').size())
                # always raise the error
                raise e

            batch_size = batch.size

            src_size = batch.src_size
            tgt_size = batch.tgt_size
            num_accumulated_words.add_(tgt_size)
            num_accumulated_sents.add_(batch_size)

            # unscale the gradient first
            self.grad_scaler.unscale_(self.optim.optimizer)

            # fake update. we need a learning rate = 0 for this
            grad_norm = clip_grad_norm(self.model.parameters(), 0)
            # self.optim.step(scaler=self.grad_scaler)
            self.grad_scaler.update()
            # Update the precision matrices.

            for n, p in parameters.items():
                if n in precision_matrices:
                    grad = p.grad.data
                    grad.masked_fill_(torch.logical_or(torch.isinf(grad), torch.isnan(grad)), 0)

                    precision_matrices[n].add_(torch.square(p.grad.data))

            self.optim.zero_grad(set_to_none=opt.true_zero_grad)
            counter = 0

            num_words = tgt_size
            report_loss.add_(loss_data)
            report_tgt_words.add_(num_words)
            report_src_words.add_(src_size) 
            total_loss.add_(loss_data)
            total_words.add_(num_words)

            # control the index a little bit to ensure the log is always printed
            if i == 0 or ((i + 1) % opt.log_interval < self.world_size):

                self.all_reduce(report_loss, op=dist.ReduceOp.SUM, group=self.group)
                self.all_reduce(report_tgt_words, op=dist.ReduceOp.SUM, group=self.group)
                self.all_reduce(report_src_words, op=dist.ReduceOp.SUM, group=self.group)
                self.all_reduce(report_contrastive_loss, op=dist.ReduceOp.SUM, group=self.group)

                if self.is_main():
                    log_string = ("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; grad_norm: %6.4f; gradscaler: %9.9f " %
                                  (epoch, i + 1, len(data_iterator),
                                   math.exp(report_loss.item() / report_tgt_words.item()),
                                   grad_norm,
                                   self.grad_scaler.get_scale()))

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
                if report_contrastive_loss is not None:
                    report_contrastive_loss.zero_()
                start = time.time()

            # increase i by world size
            i = i + self.world_size

        if isinstance(self.model, DDP_model):
            torch.cuda.synchronize(device=self.rank)

        loss = 0
        for n, p in parameters.items():
            loss = loss + p.sum() * 0
        # to force ddp to synchronize the last time (based on a zero loss -> zero grad
        loss.backward()

        self.all_reduce(num_accumulated_words, op=dist.ReduceOp.SUM, group=self.group)

        if self.world_size > 1:
            if self.rank == 0:
                print("[INFO] Synchronizing precision matrices")
            for n in precision_matrices:
                self.all_reduce(precision_matrices[n], op=dist.ReduceOp.SUM, group=self.group)

            if self.rank == 0:
                print("Done...")

        if self.rank == 0:
            # Accumulate fisher info from previous iteration
            if self.fisher_info is not None:
                print("[INFO] Accumulating fisher information from a previous iteration...")
                for n in precision_matrices:
                    if n in self.fisher_info:
                        precision_matrices[n] = self.fisher_info['fisher_diag'][n] + precision_matrices[n]

            # normalizing by the number of sentences
            # for n in precision_matrices:
            #     precision_matrices[n].div_(num_d_sents)

            means = dict()
            for n, p in parameters.items():
                if n in precision_matrices:
                    means[n] = p

            checkpoint = {
                'mean': means,
                'fisher_diag': precision_matrices,
                'opt': opt
            }

            file_name = opt.load_from + ".fisher"
            print("[INFO] Saving means and fisher information to %s" % file_name)
            torch.save(checkpoint, file_name)

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

        # if we are on a GPU: warm up the memory allocator
        # if self.cuda:
        #     self.warm_up(train_data=train_data)

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
            valid_loss, valid_accuracy = self.eval(valid_data)
            valid_ppl = math.exp(min(valid_loss, 100))

            if self.is_main():
                print('[INFO] Validation perplexity: %g' % valid_ppl)
                print('[INFO] Validation accuracy: %g percent' % (100 * valid_accuracy))
                self.save(epoch, valid_ppl if opt.save_metrics in ['ppl', 'perplexity'] else 1 - valid_accuracy)

            itr_progress = None
            resume = False
