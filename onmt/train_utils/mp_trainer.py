from __future__ import division

import datetime
import gc
import inspect
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
from onmt.utils import checkpoint_paths, normalize_gradients
from onmt.model_factory import build_model, optimize_model, init_model_parameters
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP_model
from torch.cuda.amp import autocast
import warnings

# ignore the pytorch -> numpy conversion warnings
warnings.filterwarnings("ignore", category=UserWarning)


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


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


def generate_data_iterator(dataset, rank, world_size, seed,
                           num_workers=1, epoch=1., buffer_size=0):
    # check if dataset is a list:
    if isinstance(dataset, list):
        # this is a multidataset
        data_iterator = MultiDataIterator(dataset, seed=seed, num_workers=num_workers,
                                          epoch=epoch, buffer_size=buffer_size,
                                          num_shards=world_size, shard_id=rank)
    else:
        data_iterator = DataIterator(dataset, dataset.collater, dataset.batches, seed=seed,
                                     num_workers=num_workers, epoch=epoch, buffer_size=buffer_size,
                                     num_shards=world_size, shard_id=rank)

    return data_iterator


def zero_tensor(device=None):
    if device is None:
        return torch.Tensor([0]).cuda()
    else:
        return torch.Tensor([0]).to(device)


def all_reduce_and_rescale_tensors(tensors, rescale_denom,
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

    def __init__(self, device, train_data, valid_data, dicts, opt, setup_optimizer=True):
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

        # in the case of single node distributed, it should equal self.device
        self.rank = self.device

        # make a group to later use with self.all_reduce
        self.group = dist.group.WORLD

        self.print("[INFO] Training Options:", opt)
        if self.world_size > 1:
            dist.init_process_group(backend='nccl', init_method='env://', world_size=self.world_size, rank=self.rank)

        self.model = None

        if self.rank == 0:
            self.train_data = train_data
            self.valid_data = valid_data
        else:
            # Do we really need to deepcopy the data instances (which could cause memory leak easily)
            self.train_data = copy.deepcopy(train_data)
            self.valid_data = copy.deepcopy(valid_data)

        self.dicts = dicts
        self.opt = opt
        self.cuda = (len(opt.gpus) >= 1 and opt.gpus[0] >= 0)

        assert self.cuda, "[ERROR] Training is only available on GPUs."

        self.start_time = 0

        # setting up models and others
        if opt.lfv_multilingual:
            from onmt.models.speech_recognizer.lid_loss import CrossEntropyLIDLoss
            lid_loss = CrossEntropyLIDLoss(opt.n_languages, opt.label_smoothing, opt.fast_xentropy)
            self.loss_function.add_loss_function(lid_loss, 'lid_loss')

        torch.manual_seed(self.opt.seed)

        # note: we must start creating models after ccreating the processes
        # for some reason passing a pre-created model to a process creates a "pickle" error
        if not opt.fusion:

            if self.is_main():
                print("[INFO] Building models .... ", flush=True)
            model = build_model(opt, dicts)

            """ Building the loss function """
            if opt.ctc_loss > 0.0:
                from onmt.speech.ctc_loss import CTC
                self.ctc_loss_function = CTC(dicts['tgt'].size(), opt.model_size, 0.0, reduce=True)

            if opt.nce:
                from onmt.modules.nce.nce_loss import NCELoss
                loss_function = NCELoss(opt.model_size, dicts['tgt'].size(), noise_ratio=opt.nce_noise,
                                        logz=9, label_smoothing=opt.label_smoothing)
            else:
                loss_function = NMTLossFunc(opt.model_size, dicts['tgt'].size(),
                                            label_smoothing=opt.label_smoothing,
                                            mirror=opt.mirror_loss,
                                            fast_xentropy=opt.fast_xentropy)

            # This function replaces modules with the more optimized counterparts so that it can run faster
            # Currently exp with LayerNorm
            if not opt.memory_profiling:
                # distributed is required to convert BatchNorm to SyncBatchNorm for DDP
                optimize_model(model, distributed=(self.world_size > 1))

        init_model_parameters(model, opt)
        self.model = model
        self.loss_function = loss_function
        self.grad_scaler = torch.cuda.amp.GradScaler()

        if opt.load_from:
            checkpoint = torch.load(opt.load_from, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(checkpoint['model'])
            if 'scaler' in checkpoint and checkpoint['scaler'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['scaler'])

        if self.cuda:
            torch.cuda.set_device(self.device)

            self.loss_function = self.loss_function.cuda(device=self.device)
            self.model = self.model.cuda(device=self.device)
            if opt.ctc_loss > 0.0:
                self.ctc_loss_function = self.ctc_loss_function.cuda(device=self.device)

            # Ensure that the distributed copies have the same initial parameters
            # Manual seed may not work the same for different GPU models.
            if self.world_size > 1:
                params = [p for p in self.model.parameters()]

                with torch.no_grad():
                    if not self.is_main():
                        # zero everything except for the main model
                        for p in params:
                            p.zero_()
                    else:
                        for p in params:
                            p.add_(0)

            # run all_reduce to ensure that all models have exactly the same parameters
            if self.world_size > 1:
                params = [p for p in self.model.parameters()]
                all_reduce_and_rescale_tensors(params, 1)

        if setup_optimizer:

            self.optim = onmt.Optim(opt)
            self.optim.set_parameters(self.model.parameters())

            if self.is_main():
                print("[INFO] Optimizer: ", self.optim.optimizer)

            if opt.load_from:
                if 'optim' in checkpoint and checkpoint['optim'] is not None and not opt.reset_optim:
                    self.optim.load_state_dict(checkpoint['optim'])

        if self.world_size > 1:
            # find_unused_parameters may be required for dropped layer (parameters that are not connected to
            # any particular graph)
            find_unused_parameters = True if opt.death_rate > 0.0 else True

            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank],
                                                                   output_device=self.rank,
                                                                   find_unused_parameters=find_unused_parameters)

        print("[INFO] Process %d ready." % self.rank, flush=True)

    def is_main(self):

        return self.rank == 0

    def all_reduce(self, tensor, **kwargs):

        if self.world_size > 1:
            dist.all_reduce(tensor, **kwargs)

        # otherwise, do nothing
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

    def load_encoder_weight(self, checkpoint_file):

        print("Loading pretrained models from %s" % checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)

        pretrained_model = build_model(checkpoint['opt'], checkpoint['dicts'])
        pretrained_model.load_state_dict(checkpoint['model'])

        print("Loading pretrained encoder weights ...")
        pretrained_model.encoder.language_embedding = None
        enc_language_embedding = self.model.encoder.language_embedding
        self.model.encoder.language_embedding = None
        encoder_state_dict = pretrained_model.encoder.state_dict()

        self.model.encoder.load_state_dict(encoder_state_dict)
        self.model.encoder.language_embedding = enc_language_embedding
        return

    def load_decoder_weight(self, checkpoint_file):

        self.print("Loading pretrained models from %s" % checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        chkpoint_dict = checkpoint['dicts']

        pretrained_model = build_model(checkpoint['opt'], chkpoint_dict)
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

    def warm_up(self):
        """
        Warmup the memory allocator, by attempting to fit the largest batch
        :return:
        """

        # if self.opt.memory_profiling:
        #     from pytorch_memlab import MemReporter
        #     reporter = MemReporter()
        #
        batch = self.train_data[0].get_largest_batch() if isinstance(self.train_data, list) \
            else self.train_data.get_largest_batch()
        opt = self.opt

        if self.cuda:
            batch.cuda(fp16=False)

        self.model.train()
        self.loss_function.train()
        self.model.zero_grad()
        oom = False

        if self.opt.memory_profiling:
            self.print("Input size: ")
            self.print(batch.size, batch.src_size, batch.tgt_size)

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        try:
            with autocast():
                targets = batch.get('target_output')
                tgt_mask = None
                outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
                                     zero_encoder=opt.zero_encoder,
                                     mirror=opt.mirror_loss, streaming_state=streaming_state,
                                     nce=opt.nce)

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

                if self.opt.memory_profiling:
                    reporter.report(verbose=True)

                # for obj in gc.get_objects():
                #     try:
                #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                #             # print(varname(obj))
                #             # we can rule out parameter cost later
                #             # if 'parameter' not in type(obj):
                #             # if len(obj.shape) == 3:
                #             # if not isinstance(obj, torch.nn.parameter.Parameter):
                #             #     tensor = obj
                #             #     numel = tensor.
                #             print(type(obj), obj.type(), obj.size())
                #     except:
                #         pass

                # print("Memory profiling complete.")
                # print(torch.cuda.memory_summary())
                # exit()

            self.grad_scaler.scale(full_loss).backward()
            # if self.cuda:
            #     with amp.scale_loss(full_loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            #     loss.div_(batch.tgt_size).backward()

            if self.opt.memory_profiling:
                print('========= after backward =========')
                reporter.report(verbose=True)

            self.model.zero_grad()
            self.optim.zero_grad()
            # self.optim.step()
            # self.optim.reset()

        except RuntimeError as e:
            if 'out of memory' in str(e):
                oom = True
            else:
                raise e

        if oom:
            print("[INFO] Warning: out-of-memory in warming up. "
                  "This is due to the largest batch is too big for the GPU.",
                  flush=True)
        else:
            self.print("[INFO] Warming up successfully.", flush=True)

        if self.opt.memory_profiling:
            if hasattr(torch.cuda, 'memory_summary'):
                print(torch.cuda.memory_summary())
            exit()

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
        rank = self.device
        world_size = self.world_size

        # the data iterator creates an epoch iterator
        data_iterator = generate_data_iterator(data, rank, world_size, seed=self.opt.seed,
                                               num_workers=opt.num_workers, epoch=1, buffer_size=opt.buffer_size)
        epoch_iterator = data_iterator.next_epoch_itr(False, pin_memory=False)

        data_size = len(epoch_iterator)
        i = 0

        self.model.eval()
        self.loss_function.eval()
        # self.model.module.reset_states()

        total_loss = zero_tensor()
        total_words = zero_tensor()

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        with torch.no_grad():
            while not data_iterator.end_of_epoch():
                samples = next(epoch_iterator)

                if samples:
                    with autocast():
                        batch = prepare_sample(samples, device=self.device)
                        targets = batch.get('target_output')
                        tgt_mask = targets.ne(onmt.constants.PAD)
                        outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
                                             mirror=opt.mirror_loss, streaming_state=streaming_state, nce=opt.nce)

                        outputs['tgt_mask'] = tgt_mask
                        loss_dict = self.loss_function(outputs, targets, model=self.model, eval=True)
                        loss_data = loss_dict['data']

                    total_loss.add_(loss_data)
                    total_words.add_(batch.tgt_size)
                    i = i + 1

        # allreduce the total loss and total words from other processes
        self.all_reduce(total_loss, op=dist.ReduceOp.SUM, group=self.group)
        self.all_reduce(total_words, op=dist.ReduceOp.SUM, group=self.group)

        self.model.train()
        self.loss_function.train()

        return total_loss / total_words

    def train_epoch(self, epoch, resume=False, itr_progress=None):

        global rec_ppl
        opt = self.opt
        train_data = self.train_data
        streaming = opt.streaming

        # Clear the gradients of the model
        self.model.zero_grad()
        # self.model.module.reset_states()

        dataset = train_data
        data_iterator = generate_data_iterator(dataset, self.rank, self.world_size,
                                               seed=self.opt.seed, num_workers=opt.num_workers,
                                               epoch=epoch, buffer_size=opt.buffer_size)

        # TODO: fix resume which is currently buggy
        if resume:
            data_iterator.load_state_dict(itr_progress)

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
        grad_div = 1

        nan = False
        nan_counter = zero_tensor()

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        i = data_iterator.iterations_in_epoch if not isinstance(train_data, list) else epoch_iterator.n_yielded
        i = i * self.world_size

        while not data_iterator.end_of_epoch():

            curriculum = (epoch < opt.curriculum)

            # this batch generator is not very clean atm
            # TODO: move everything to the multiGPU trainer
            samples = next(epoch_iterator)

            batch = prepare_sample(samples, device=self.device)

            if opt.streaming:
                if train_data.is_new_stream():
                    streaming_state = self.model.init_stream()
            else:
                streaming_state = None

            # TODO: dealing with oom during distributed training
            oom = False

            try:
                # outputs is a dictionary containing keys/values necessary for loss function
                # can be flexibly controlled within models for easier extensibility
                counter = counter + 1
                reduction_disabled = False if counter >= opt.update_frequency or i == (n_samples - 1) else True

                def maybe_no_sync():
                    if not reduction_disabled and isinstance(self.model, DDP_model):
                        return self.model.no_sync()
                    else:
                        # when we dont reach the updating step, we do not need to synchronize the gradients
                        # thus disabling the backward grad sync to improve speed
                        return contextlib.ExitStack()  # dummy contextmanager

                with maybe_no_sync():
                    with autocast():
                        targets = batch.get('target_output')
                        tgt_mask = targets.ne(onmt.constants.PAD)
                        outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
                                             zero_encoder=opt.zero_encoder,
                                             mirror=opt.mirror_loss, streaming_state=streaming_state,
                                             nce=opt.nce)

                        batch_size = batch.size
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

                        if opt.lfv_multilingual:
                            lid_logits = outputs['lid_logits']
                            lid_labels = batch.get('target_lang')
                            lid_loss_function = self.loss_function.get_loss_function('lid_loss')
                            lid_loss = lid_loss_function(lid_logits, lid_labels)
                            full_loss = full_loss + lid_loss

                        optimizer = self.optim.optimizer

                        # When the batch size is large, each gradient step is very easy to explode on fp16
                        # Normalizing the loss to grad scaler ensures this will not happen
                        full_loss.div_(grad_div)

                    # grad scaler has to be done outside of the autocast
                    self.grad_scaler.scale(full_loss).backward()

                del outputs

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('[WARNING]: ran out of memory on GPU %d' % self.rank, flush=True)
                    oom = True
                    torch.cuda.empty_cache()
                    loss = 0
                    if opt.streaming:  # reset stream in this case ...
                        streaming_state = self.model.init_stream()
                    raise e
                else:
                    raise e

            batch_size = batch.size

            src_size = batch.src_size
            tgt_size = batch.tgt_size
            num_accumulated_words.add_(tgt_size)
            num_accumulated_sents.add_(batch_size)

            # We only update the parameters after getting gradients from n mini-batches
            update_flag = False
            if counter >= opt.update_frequency:
                update_flag = True
            elif i == n_samples - 1:  # update for the last minibatch
                update_flag = True

            if update_flag:
                # accumulated gradient case, in this case the update frequency
                self.all_reduce(num_accumulated_words, op=dist.ReduceOp.SUM, group=self.group)

                grad_denom = 1.0 / grad_div

                if self.opt.normalize_gradient:
                    grad_denom = num_accumulated_words.item() * grad_denom
                else:
                    grad_denom = 1

                # the gradient is scaled by world size, so in order to match the model without multiGPU
                # we rescale the model parameters w.r.t the world size
                grad_denom = grad_denom / self.world_size

                # When we accumulate the gradients, each gradient is already normalized by a constant grad_scaler
                normalize_gradients(self.model.parameters(), grad_denom)

                # Update the parameters.
                if self.opt.max_grad_norm > 0:
                    self.grad_scaler.unscale_(self.optim.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.max_grad_norm)
                self.optim.step(scaler=self.grad_scaler)
                self.grad_scaler.update()
                # self.optim.zero_grad()
                self.model.zero_grad()
                counter = 0
                num_accumulated_words.zero_()
                num_accumulated_sents.zero_()

                num_updates = self.optim._step
                if opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every:
                    valid_loss = self.eval(self.valid_data)
                    valid_ppl = math.exp(min(valid_loss, 100))

                    if self.is_main():
                        print('Validation perplexity: %g' % valid_ppl)
                        ep = float(epoch) - 1. + ((float(i) + 1.) / n_samples)
                        self.save(ep, valid_ppl, itr=data_iterator)

            num_words = tgt_size
            report_loss.add_(loss_data)
            report_tgt_words.add_(num_words)
            report_src_words.add_(src_size)
            total_loss.add_(loss_data)
            total_words.add_(num_words)
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

                if self.is_main():
                    log_string = ("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; " %
                                  (epoch, i + 1, len(data_iterator),
                                   math.exp(report_loss.item() / report_tgt_words.item())))

                    if opt.reconstruct:
                        self.all_reduce(report_rec_loss, op=dist.ReduceOp.SUM, group=self.group)
                        rec_ppl = math.exp(report_rec_loss.item() / report_src_words.item())
                        log_string += (" rec_ppl: %6.2f ; " % rec_ppl)

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

                    log_string += ("lr: %.7f ; updates: %7d; " %
                                   (self.optim.getLearningRate(),
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
                start = time.time()

            # increase i by world size
            i = i + self.world_size

        return total_loss / total_words

    # def run(self, save_file=None):
    def run(self, checkpoint=None):

        opt = self.opt

        if checkpoint is not None:

            # TODO: have loading checkpoints for each process
            prec_opt = checkpoint['opt'] if 'opt' in checkpoint else None

            if not opt.reset_optim:

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

        valid_loss = self.eval(self.valid_data)
        valid_ppl = math.exp(min(valid_loss, 100))

        if self.is_main():
            print('[INFO] Validation perplexity: %g' % valid_ppl, flush=True)

        self.start_time = time.time()

        for epoch in range(start_epoch, start_epoch + opt.epochs):
            self.print('')

            #  (1) train for one epoch on the training set
            train_loss = self.train_epoch(epoch, resume=resume, itr_progress=itr_progress)
            train_ppl = math.exp(min(train_loss, 100))
            self.print('[INFO] Train perplexity: %g' % train_ppl)

            #  (2) evaluate on the validation set
            valid_loss = self.eval(self.valid_data)
            valid_ppl = math.exp(min(valid_loss, 100))

            if self.is_main():
                print('[INFO] Validation perplexity: %g' % valid_ppl)
                self.save(epoch, valid_ppl)

            itr_progress = None
            resume = False
