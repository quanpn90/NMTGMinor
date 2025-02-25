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
import functools


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

from .reservoir import Reservoir
from onmt.data.dataset import get_batch_from_multidataset

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP_model
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.cuda.amp import autocast
import warnings
from onmt.constants import add_tokenidx
import dill
from multiprocessing.managers import ListProxy as ListProxy

from distutils.version import LooseVersion
# ignore the pytorch -> numpy conversion warnings
warnings.filterwarnings("ignore", category=UserWarning)


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'),
                         ['', 'K', 'M', 'B', 'T'][magnitude])


def prepare_sample(batch, device=None, reservoir=None, dataset_id=None):
    """
    Put minibatch on the corresponding GPU
    :param batch:
    :param device:
    :param reservoir:
    :param dataset_id:
    :return:
    """

    if isinstance(batch, list):
        batch = batch[0]
    batch = rewrap(batch)

    if reservoir is not None:

        assert dataset_id is not None
        indices = batch.get('indices')
        src_lengths = batch.get('src_lengths')
        tgt_lengths = batch.get('tgt_lengths')
        reservoir.add_sample((dataset_id, indices, src_lengths, tgt_lengths))

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


class OCLTrainer(object):

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

        else:
            self.ctc_loss_function = None

        if opt.transducer_loss > 0.0:
            from onmt.modules.transducer import TransducerLoss
            # self.ctc_loss_function = CTC(dicts['tgt'].size(), opt.model_size, 0.0, reduce=True,
            #                              padding_idx=tgt_pad, blank_idx=0)
            self.transducer_loss_function = TransducerLoss(fuse_softmax_backward=True,
                                                  opt=1, packed_input=False,
                                                  blank_idx=tgt_pad)

        else:
            self.transducer_loss_function = None

        if opt.predict_language > 0:
            from onmt.models.speech_recognizer.lid_loss import CrossEntropyLIDLoss
            self.lid_loss_function = CrossEntropyLIDLoss(opt.n_languages, label_smoothing=0.0)

        if opt.nce:
            from onmt.modules.nce.nce_loss import NCELoss
            loss_function = NCELoss(opt.model_size, dicts['tgt'].size(), noise_ratio=opt.nce_noise,
                                    logz=9, label_smoothing=opt.label_smoothing)
        else:
            self.print("Target pad idx:", tgt_pad)
            loss_function = NMTLossFunc(opt.model_size, dicts['tgt'].size(),
                                        label_smoothing=opt.label_smoothing,
                                        mirror=opt.mirror_loss > 0.0,
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

        if opt.load_from:
            checkpoint = torch.load(opt.load_from, map_location=lambda storage, loc: storage)

            if opt.finalize_only:
                self.print("[INFO] Finalizing only ... skipping loading parameters")
            else:
                try:
                    self.model.load_state_dict(checkpoint['model'])
                except RuntimeError as e:
                    self.model.load_state_dict(checkpoint['model'], strict=True)

        self.agem_training = False
        if opt.agem_training:
            self.agem_training = True

            mem_model = build_model(opt, dicts, False, self.constants, verbose=False)
            optimize_model(mem_model, distributed=(self.world_size > 1))

            self.mem_model = mem_model
            self.mem_model.load_state_dict(self.model.state_dict())

        if self.cuda:
            self.loss_function = self.loss_function.cuda(device=self.device)
            self.model = self.model.cuda(device=self.device)
            if opt.ctc_loss > 0.0:
                self.ctc_loss_function = self.ctc_loss_function.cuda(device=self.device)
            if opt.load_pretrained_classifier:
                self.classifier = self.classifier.cuda(device=self.device)

            if self.agem_training:
                self.mem_model = self.mem_model.cuda(device=self.device)

        # if self.opt.flatten_parameters:
        #     self.optim.flatten_parameters()
        #     if self.agem_training:
        #         self.mem_optim.flatten_parameters()

        fpSixteen = MixedPrecision(
            param_dtype=torch.float16,
            # Gradient communication precision.
            reduce_dtype=torch.float16,
            # Buffer precision.
            buffer_dtype=torch.float16,
        )

        bfSixteen = MixedPrecision(
            param_dtype=torch.bfloat16,
            # Gradient communication precision.
            reduce_dtype=torch.bfloat16,
            # Buffer precision.
            buffer_dtype=torch.bfloat16,
        )

        fp32_policy = MixedPrecision(
            param_dtype=torch.float32,
            # Gradient communication precision.
            reduce_dtype=torch.float32,
            # Buffer precision.
            buffer_dtype=torch.float32,
        )

        self.bf16_ready = (
                opt.bf16
                and torch.version.cuda
                and torch.cuda.is_bf16_supported()
                and LooseVersion(torch.version.cuda) >= "11.0"
                and dist.is_nccl_available()
                and torch.cuda.nccl.version() >= (2, 10)
        )

        if self.bf16_ready:
            mp_policy = bfSixteen
        else:
            mp_policy = None  # defaults to fp32

        # change later
        if opt.fp16 and not opt.bf16:
            self.grad_scaler = torch.cuda.amp.GradScaler()
            if self.agem_training:
                self.mem_grad_scaler = torch.cuda.amp.GradScaler()
        else:
            self.grad_scaler = None
            self.mem_grad_scaler = None

        if opt.fsdp and self.world_size > 1:
            device_id = self.rank
            torch.cuda.set_device(device_id)

            from torch.distributed.fsdp.wrap import (
                transformer_auto_wrap_policy,
                enable_wrap,
                wrap,
            )

            # Testing: fsdp wrap these blocks independently

            from onmt.models.conformer.block import ConformerBlock
            from pretrain_module.modeling_mbart import MBartEncoderLayer

            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={
                    ConformerBlock, MBartEncoderLayer
                },
            )

            # otherwise maybe FSDP is only applicable to either Conformer or Language Model :)

            self.model = FSDP(self.model,
                              auto_wrap_policy=auto_wrap_policy,
                              # mixed_precision=mp_policy,
                              device_id=torch.cuda.current_device())

            if setup_optimizer:

                self.optim = onmt.Optim(opt)
                self.optim.set_parameters(self.model.parameters(), flattened=opt.flatten_parameters)

                if self.is_main():
                    print("[INFO] Optimizer: ", self.optim.optimizer)

                if opt.load_from and not opt.reset_optim:
                    if 'optim' in checkpoint and checkpoint['optim'] is not None and not opt.reset_optim:

                        # TODO: load state dict after optim ...
                        # optim_state_dict = FSDP.optim_state_dict_to_load(
                        #                    >> > optim_state_dict, model, optim
                        #                    >> > )
                        # >> > optim.load_state_dict(optim_state_dict)
                        self.optim.load_state_dict(checkpoint['optim'])

                if opt.starting_step > 0:
                    print("[INFO] Optimizer starting from state %d " % opt.starting_step)
                    self.optim.set_starting_step(opt.starting_step)
        else:
            if setup_optimizer:

                self.optim = onmt.Optim(opt)
                self.optim.set_parameters(self.model.parameters(), flattened=opt.flatten_parameters)

                if self.agem_training:
                    # probably adam is fine, we don't have to ever call update
                    # so the memory clones are never required
                    self.mem_optim = onmt.Optim(opt)
                    self.mem_optim.set_parameters(self.mem_model.parameters(), flattened=opt.flatten_parameters)

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
                                                                       find_unused_parameters=find_unused_parameters,
                                                                       )

                if self.agem_training:
                    self.mem_model = torch.nn.parallel.DistributedDataParallel(self.mem_model, device_ids=[self.rank],
                                                                               output_device=self.rank,
                                                                               find_unused_parameters=find_unused_parameters,)


        if self.is_main():
            nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("[INFO] Total number of trainable paramaters: %d" % nparams)
            nparams = sum(p.numel() for p in model.parameters())
            print("[INFO] Total number of paramaters: %d" % nparams)

            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.is_main = True
            else:
                self.model.is_main = True

        if opt.load_fisher:
            if self.is_main():
                self.print("[INFO] Loading fisher information from: %s" % opt.load_fisher)
            self.fisher_info = torch.load(opt.load_fisher, map_location=lambda storage, loc: storage)
            if self.cuda:
                for n in self.fisher_info['mean']:
                    self.fisher_info['mean'][n] = self.fisher_info['mean'][n].cuda()
                for n in self.fisher_info['fisher_diag']:
                    self.fisher_info['fisher_diag'][n] = self.fisher_info['fisher_diag'][n].cuda()
        else:
            self.fisher_info = None

        self.print("[INFO] Creating Reservoir ...")

        # TODO: add option for reservoir size

        reservoir_size = opt.reservoir_size // self.world_size
        if reservoir_size > 0:
            self.reservoir = Reservoir(max_samples=reservoir_size,
                                       update_method="reservoir_sampling",
                                       unit="sample",
                                       batch_size_frames=opt.batch_size_frames,
                                       batch_size_sents=opt.batch_size_sents,
                                       batch_size_words=opt.batch_size_words
                                       )
        else:
            self.reservoir = None

        print("[INFO] Process %d ready." % self.rank, flush=True)

        self.total_accumulated_src = zero_tensor()
        self.total_accumulated_tgt = zero_tensor()

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

    def save(self, epoch, round, valid_ppl, itr=None):

        opt = self.opt
        model = self.model
        dicts = self.dicts

        if self.opt.fsdp:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                    self.model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                model_state_dict = self.model.state_dict()
                optim_state_dict = FSDP.optim_state_dict(self.model, self.optim.optimizer)

        else:
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
            'round': round,
            'epoch': epoch,
            'itr': itr_state_dict,
            'optim': optim_state_dict,
            'scaler': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
            'reservoir': self.reservoir.state_dict() if self.reservoir is not None else None
        }

        file_name = '%s_epoch%.2f.round%d' % (opt.save_model, epoch, round)
        if self.is_main():
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

        def maybe_no_sync():
            if isinstance(self.model, DDP_model) or isinstance(self.model, FSDP):
                return self.model.no_sync()
            else:
                return contextlib.ExitStack()  # dummy contextmanager

        with maybe_no_sync():
            with torch.no_grad():
                # while not data_iterator.end_of_epoch():
                while i < len(epoch_iterator):
                    samples = next(epoch_iterator)

                    if samples:
                        with autocast(enabled=opt.fp16, dtype=torch.bfloat16 if self.bf16_ready else torch.float16):
                            batch = prepare_sample(samples, device=self.device)
                            targets = batch.get('target_output')
                            tgt_mask = targets.ne(onmt.constants.PAD)

                            outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
                                                 mirror=opt.mirror_loss, streaming_state=streaming_state,
                                                 pretrained_layer_states=None,
                                                 ctc_loss_function=self.ctc_loss_function,
                                                 ctc_labels=targets,
                                                 grad_scaler=self.grad_scaler,
                                                 ctc_coeff=opt.ctc_loss
                                                 )

                            ctc_only = False
                            if outputs["hidden"] != None:
                                outputs['tgt_mask'] = tgt_mask
                                loss_dict = self.loss_function(outputs, targets, model=self.model, eval=True)
                                loss_data = loss_dict['data']
                                correct, total = loss_dict['correct'], loss_dict['total']

                                assert (total == batch.tgt_size), \
                                    "Process %i, Minibatch %d/%d: Expected %d tokens from the batch, got %d" \
                                    % (self.rank, i, data_size, batch.tgt_size, total)

                            else:
                                ctc_only = True
                                loss_data = 0
                                loss = None
                                full_loss = 0
                                correct = 0
                                ctc_loss = outputs['ctc_loss']
                                n_ctc_targets = outputs['n_ctc_targets']
                                loss_data = ctc_loss.item()

                        total_loss.add_(loss_data)
                        total_words.add_(batch.tgt_size)
                        total_correct.add_(correct)
                        i = i + 1

        # allreduce the total loss and total words from other processes
        self.all_reduce(total_loss, op=dist.ReduceOp.SUM, group=self.group)
        self.all_reduce(total_words, op=dist.ReduceOp.SUM, group=self.group)
        self.all_reduce(total_correct, op=dist.ReduceOp.SUM, group=self.group)

        if opt.use_memory:
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                model = self.model.module
            else:
                model = self.model
            if hasattr(model, "memory_stats"):
                self.all_reduce(model.memory_stats, op=dist.ReduceOp.SUM, group=self.group)
            else:
                print("WARNING: Could not all_reduce in mp_trainer")

        self.model.train()
        self.loss_function.train()
        if opt.load_pretrained_classifier:
            self.classifier.train()

        return total_loss.item() / total_words.item(), total_correct.item() / total_words.item()

    def train_epoch(self, train_data, valid_data, epoch, resume=False, itr_progress=None):

        opt = self.opt
        streaming = False
        grad_norm = -1
        ema = self.opt.ema

        # Clear the gradients of the model
        self.optim.zero_grad(set_to_none=not opt.true_zero_grad)
        # self.model.module.reset_states()

        # note: for Training split_even=True
        dataset = train_data

        assert is_list(dataset)

        data_iterators = list()

        for _dataset in dataset:
            data_iterator = generate_data_iterator(_dataset, self.rank, self.world_size,
                                                   seed=self.opt.seed, num_workers=opt.num_workers,
                                                   epoch=epoch, buffer_size=opt.buffer_size, split_even=True,
                                                   dataset_ids=None)

            data_iterators.append(data_iterator)

        # TODO: fix resume which is currently buggy
        if resume:
            data_iterator.load_state_dict(itr_progress)

        epoch_iterators = list()

        for _data_iterator in data_iterators:
            epoch_iterators.append(_data_iterator.next_epoch_itr(not streaming, pin_memory=opt.pin_memory))

        total_tokens, total_loss, total_words = zero_tensor(), zero_tensor(), zero_tensor()
        total_non_pads = zero_tensor()
        report_loss, report_tgt_words = zero_tensor(), zero_tensor()
        report_ctc_loss = zero_tensor()
        report_transducer_loss = zero_tensor()
        report_ewc_loss = zero_tensor()
        report_ctc_targets = zero_tensor()
        report_transducer_targets = zero_tensor()
        report_ewc_count = 0
        report_src_words = zero_tensor()
        report_sents = zero_tensor()
        report_rec_loss, report_rev_loss, report_mirror_loss = zero_tensor(), zero_tensor(), zero_tensor()

        report_enc_lid_loss = zero_tensor()
        report_enc_lid_count = 0
        report_dec_lid_loss = zero_tensor()
        report_dec_lid_count = 0

        start = time.time()

        counter = 0
        num_accumulated_words = zero_tensor()
        num_accumulated_sents = zero_tensor()
        report_contrastive_loss = zero_tensor()

        total_accumulated_src += batch.src_size

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

        assert len(data_iterators) == len(epoch_iterators)
        assert len(data_iterators) == len(train_data)

        for dataset_id, (_data_iterator, _epoch_iterator) in enumerate(zip(data_iterators, epoch_iterators)):

            # maybe clean up everything from the last round?
            gc.collect()

            n_samples = len(_data_iterator) * 2 - 1 if (epoch > 1 or dataset_id > 0) else len(_data_iterator)
            i = _data_iterator.iterations_in_epoch
            i = i * self.world_size
            self.print("Training Epoch %d - Round (dataset) %d" % (epoch, dataset_id))

            rehearse = False

            update_frequency = 2 * opt.update_frequency

            while not (_data_iterator.end_of_epoch() and not rehearse) :

                if not rehearse or opt.reservoir_size <= 0:
                    samples = next(_epoch_iterator)
                    lagrangian_weights = None

                    # sample is also added to the reservoir here
                    batch = prepare_sample(samples, device=self.device, dataset_id=dataset_id,
                                                                            reservoir=self.reservoir)

                    rehearse = False
                    rehearsing = False

                    # if (epoch > 1 or dataset_id > 0) and self.reservoir is not None:
                    if self.reservoir is not None:

                        # we start to rehearse immediately

                        rehearse = True  # so that the next one is to rehearse
                else:
                    rehearsed_data = self.reservoir.sample()
                    rehearsed_dataset_ids = rehearsed_data[0]
                    rehearsed_indices = rehearsed_data[1]

                    samples = get_batch_from_multidataset(train_data, rehearsed_dataset_ids,
                                                          rehearsed_indices)

                    batch = prepare_sample(samples, device=self.device)

                    rehearsing = True
                    rehearse = False

                targets = batch.get('target_output')

                # DEPRECATE streaming state
                streaming_state = None

                # TODO: dealing with oom during distributed training
                oom = zero_tensor()
                counter = counter + 1
                reduce = True if counter >= update_frequency or i == (n_samples - 1) else False

                try:
                    def maybe_no_sync():
                        if not reduce and (isinstance(self.model, DDP_model) or isinstance(self.model, FSDP)):
                            return self.model.no_sync()
                        else:
                            # when we dont reach the updating step, we do not need to synchronize the gradients
                            # thus disabling the backward grad sync to improve speed
                            return contextlib.ExitStack()  # dummy contextmanager

                    with maybe_no_sync():
                        with autocast(enabled=opt.fp16, dtype=torch.bfloat16 if self.bf16_ready else torch.float16):

                            tgt_mask = targets.ne(onmt.constants.PAD)

                            outputs = self.model(batch, streaming=False, target_mask=tgt_mask,
                                                 zero_encoder=opt.zero_encoder,
                                                 mirror=opt.mirror_loss > 0,
                                                 adv_ptb_grad=False,
                                                 checkpointing_ffn=opt.checkpointing_ffn,
                                                 checkpointing_cross_attn=opt.checkpointing_cross_attn,
                                                 checkpointing_self_attn=opt.checkpointing_self_attn,
                                                 ctc_loss_function = self.ctc_loss_function,
                                                 ctc_labels = targets,
                                                 grad_scaler = self.grad_scaler,
                                                 ctc_coeff = opt.ctc_loss if self.optim._step > opt.ctc_loss_delay else 0.0,
                                                 transducer_loss_function = self.transducer_loss_function,
                                                 transducer_coeff = opt.transducer_loss
                                                 )

                            batch_size = batch.size
                            # outputs is a dictionary containing keys/values necessary for loss function
                            # can be flexibly controlled within models for easier extensibility
                            outputs['tgt_mask'] = tgt_mask

                            ctc_only = False
                            if outputs["hidden"] != None:
                                loss_dict = self.loss_function(outputs, targets,
                                                               model=self.model, lagrangian_weights=lagrangian_weights)
                                loss_data = loss_dict['data']
                                loss = loss_dict['loss']  # a little trick to avoid gradient overflow with fp16
                                full_loss = loss
                            else:
                                ctc_only = True
                                loss_data = 0
                                loss = None
                                full_loss = 0

                            if opt.ctc_loss > 0.0:
                                ctc_loss = outputs['ctc_loss']
                                n_ctc_targets = outputs['n_ctc_targets']
                                # TODO: add CTC loss to models
                                ctc_loss_data = ctc_loss.item()
                                full_loss = full_loss + opt.ctc_loss * ctc_loss
                            else:
                                n_ctc_targets = 0
                                ctc_loss_data = 0

                            if opt.transducer_loss > 0.0:
                                transducer_loss = outputs['transducer_loss']
                                n_transducer_targets = outputs['transducer_numel']
                                # TODO: add CTC loss to models
                                transducer_loss_data = transducer_loss.item()
                                full_loss = full_loss + opt.transducer_loss * transducer_loss
                            else:
                                n_transducer_targets = 0
                                transducer_loss_data = 0

                            if opt.mirror_loss > 0.0:
                                rev_loss = loss_dict['rev_loss']
                                rev_loss_data = loss_dict['rev_loss_data']
                                mirror_loss = loss_dict['mirror_loss']
                                full_loss = full_loss + rev_loss + mirror_loss * opt.mirror_loss
                                mirror_loss_data = loss_dict['mirror_loss'].item()
                            else:
                                rev_loss_data = None
                                mirror_loss_data = 0


                            enc_lid_loss = None
                            enc_lid_loss_data = None
                            dec_lid_loss = None
                            dec_lid_loss_data = None
                            rec_loss_data = None

                            # correct, total = loss_dict['correct'], loss_dict['total']
                            # optimizer = self.optim.optimizer

                        # grad scaler has to be done outside of the autocast
                        if self.grad_scaler is not None:
                            self.grad_scaler.scale(full_loss).backward()
                        else:
                            full_loss.backward()

                        # EWC training: no need for autograd here?
                        if self.optim._step % opt.ewc_decay_every == 0:

                            ewc_importance = ewc_importance / opt.ewc_decay_scale

                        # only run this ewc everytime we reduce

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('[WARNING]: ran out of memory on GPU %d' % self.rank, flush=True)
                        print('Input size at OOM position:', batch.get('source').size() if batch.get('source') is not None else None,
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

                self.total_accumulated_src.add_(src_size)
                self.total_accumulated_tgt.add_(tgt_size)

                # We only update the parameters after getting gradients from n mini-batches
                update_flag = reduce

                if update_flag:

                    # accumulated gradient case, in this case the update frequency
                    self.all_reduce(num_accumulated_words, op=dist.ReduceOp.SUM, group=self.group)

                    grad_denom = 1.0

                    if self.grad_scaler is not None:
                        self.grad_scaler.unscale_(self.optim.optimizer)

                    if self.opt.normalize_gradient:
                        grad_denom = num_accumulated_words.item() * grad_denom
                    else:
                        grad_denom = 1

                    params = self.optim.get_params()
                    # When we accumulate the gradients, each gradient is already normalized by a constant grad_scaler
                    if grad_denom != 1 and not self.opt.fsdp:
                        normalize_gradients(params, grad_denom)

                    # Update the pagrameters.
                    if self.opt.fsdp:
                        if self.opt.max_grad_norm > 0:
                            grad_norm = self.model.clip_grad_norm_(self.opt.max_grad_norm)
                        else:
                            grad_norm = -1
                    else:
                        grad_norm = clip_grad_norm(params, self.opt.max_grad_norm)

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

                    if ema:
                        cur_model = self.model.module if self.world_size > 1 else self.model
                        encoder_weights = cur_model.encoder.state_dict()
                        decoder_weights = cur_model.decoder.state_dict()
                    else:
                        encoder_weights, decoder_weights = None, None

                    if self.grad_scaler is not None:
                        self.optim.step(scaler=self.grad_scaler)
                        self.grad_scaler.update()
                    else:
                        self.optim.step(scaler=None)
                    self.optim.zero_grad(set_to_none=True if self.opt.fsdp else not opt.true_zero_grad)

                    if ema:
                        # print("[INFO] Exponential Moving Averaging weights ...")
                        cur_model = self.model.module if self.world_size > 1 else self.model
                        new_encoder_weights = cur_model.encoder.state_dict()
                        new_decoder_weights = cur_model.decoder.state_dict()

                        # TODO: compute the alphas
                        encoder_alpha = 0.0001
                        decoder_alpha = 0.0001

                        # alpha is small so most of the weights focus on the previous weights
                        for key in new_encoder_weights.keys():
                            new_encoder_weights[key] = new_encoder_weights[key] * encoder_alpha + (1 - encoder_alpha) * \
                                                       encoder_weights[key]

                        for key in new_decoder_weights.keys():
                            new_decoder_weights[key] = new_decoder_weights[key] * decoder_alpha + (1 - decoder_alpha) * \
                                                       decoder_weights[key]

                        cur_model.encoder.load_state_dict(new_encoder_weights)
                        cur_model.decoder.load_state_dict(new_decoder_weights)

                    counter = 0
                    num_accumulated_words.zero_()
                    num_accumulated_sents.zero_()

                    num_updates = self.optim._step
                    if (opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every) \
                            or (num_updates >= opt.max_step):
                        torch.cuda.synchronize()
                        valid_loss, valid_accuracy = self.eval(valid_data)
                        valid_ppl = math.exp(min(valid_loss, 100))

                        self.print('Validation perplexity: %g' % valid_ppl)
                        self.print('Validation accuracy: %g percent' % (100 * valid_accuracy))
                        ep = float(epoch) - 1. + ((float(i) + 1.) / n_samples)
                        if opt.save_metrics in ['ppl', 'perplexity']:
                            value = valid_ppl
                        elif opt.save_metrics == "memory":
                            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                                value = self.model.module.choose_best_epoch_by
                            else:
                                value = self.model.choose_best_epoch_by
                        else:
                            value = 1-valid_accuracy
                        self.save(ep, value,
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
                    report_ctc_targets.add_(n_ctc_targets)

                if opt.transducer_loss > 0.0:
                    report_transducer_loss.add_(transducer_loss_data)
                    report_transducer_targets.add_(n_transducer_targets)

                # control the index a little bit to ensure the log is always printed
                if i == 0 or ((i + 1) % opt.log_interval < self.world_size) and not rehearsing:

                    self.all_reduce(report_loss, op=dist.ReduceOp.SUM, group=self.group)
                    # self.all_reduce(report_ewc_loss, op=dist.ReduceOp.SUM, group=self.group)
                    self.all_reduce(report_tgt_words, op=dist.ReduceOp.SUM, group=self.group)
                    self.all_reduce(report_src_words, op=dist.ReduceOp.SUM, group=self.group)
                    # self.all_reduce(report_sents, op=dist.ReduceOp.SUM, group=self.group)
                    # self.all_reduce(report_contrastive_loss, op=dist.ReduceOp.SUM, group=self.group)
                    if opt.ctc_loss > 0.0:
                        self.all_reduce(report_ctc_loss, op=dist.ReduceOp.SUM, group=self.group)
                        self.all_reduce(report_ctc_targets, op=dist.ReduceOp.SUM, group=self.group)

                    if opt.transducer_loss > 0.0:
                        self.all_reduce(report_transducer_loss, op=dist.ReduceOp.SUM, group=self.group)
                        self.all_reduce(report_transducer_targets, op=dist.ReduceOp.SUM, group=self.group)

                    if self.is_main():

                        if ctc_only:
                            log_string = ("Epoch %2d, Rd %d, %5d/%5d; ; grad_norm: %6.4f " %
                                          (epoch, dataset_id, i + 1, len(_data_iterator),
                                           grad_norm))
                        else:
                            log_string = ("Ep %2d, Rd %d, %5d/%5d; ; ppl: %6.2f ; grad_norm: %6.4f " %
                                          (epoch, dataset_id, i + 1, len(_data_iterator),
                                           math.exp(report_loss.item() / report_tgt_words.item()),
                                           grad_norm))

                        if opt.mirror_loss:
                            self.all_reduce(report_rev_loss, op=dist.ReduceOp.SUM, group=self.group)
                            rev_ppl = math.exp(report_rev_loss.item() / report_tgt_words.item())
                            log_string += (" rev_ppl: %6.2f ; " % rev_ppl)
                            log_string += (" mir_loss: %6.2f ; " % (report_mirror_loss / report_tgt_words))

                        if opt.ctc_loss > 0.0:
                            ctc_loss_string = report_ctc_loss.item() / report_ctc_targets.item()
                            log_string += (" ctc_ppl: %5.2f ; " % math.exp(ctc_loss_string))

                        if opt.transducer_loss > 0.0:
                            transducer_loss_string = report_transducer_loss.item() / report_transducer_targets.item()
                            log_string += (" trc_ppl: %5.2f ; " % math.exp(transducer_loss_string))

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

                        if opt.predict_language > 0:
                            try:
                                _enc_lid_loss = report_enc_lid_loss.item() / report_enc_lid_count
                                _dec_lid_loss = report_dec_lid_loss.item() / report_dec_lid_count
                            except ZeroDivisionError:
                                _enc_lid_loss =  float('nan')
                                _dec_lid_loss = float('nan')
                            log_string += (" enc_lidloss: %8.8f ; " % _enc_lid_loss)
                            log_string += (" dec_lidloss: %8.8f ; " % _dec_lid_loss)

                        log_string += ("lr: %.7f ; updates: %7d; " %
                                       (self.optim.get_learning_rate(),
                                        self.optim._step))

                        src_speed = report_src_words.item() / (time.time() - start)
                        src_speed = human_format(src_speed)

                        tgt_speed = report_tgt_words.item() / (time.time() - start)
                        tgt_speed = human_format(tgt_speed)

                        log_string += ("%s src tok/s; %s tgt tok/s; " %
                                       (src_speed, tgt_speed))

                        log_string += ("%s" %
                                       str(datetime.timedelta(seconds=int(time.time() - self.start_time))))

                        self.print(log_string, flush=True)

                    report_loss.zero_()
                    report_tgt_words.zero_()
                    report_src_words.zero_()
                    report_rec_loss.zero_()
                    report_rev_loss.zero_()
                    report_mirror_loss.zero_()
                    report_ctc_loss.zero_()
                    report_ctc_targets.zero_()

                    report_transducer_loss.zero_()
                    report_transducer_targets.zero_()

                    report_ewc_loss.zero_()
                    report_ewc_count = 0
                    # report_sents.zero_()
                    if report_contrastive_loss is not None:
                        report_contrastive_loss.zero_()
                    start = time.time()

                # increase i by world size
                if not rehearsing:
                    i = i + self.world_size

            # END OF ROUND -> run validation and save
            # we run validation on all valid datasets
            valid_loss, valid_accuracy = self.eval(valid_data)
            valid_ppl = math.exp(min(valid_loss, 100))
            self.print('[INFO] Validation perplexity: %g' % valid_ppl)
            self.print('[INFO] Validation accuracy: %g percent' % (100 * valid_accuracy))
            if opt.save_metrics in ['ppl', 'perplexity']:
                value = valid_ppl
            elif opt.save_metrics == "memory":
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                    value = self.model.module.choose_best_epoch_by
                else:
                    value = self.model.choose_best_epoch_by
            else:
                value = 1 - valid_accuracy
            self.save(epoch, dataset_id, value)

            if self.reservoir is not None:
                total_per_dataset = self.reservoir.get_stats()

                # some stupid code to grab the memory statistics
                n_dataset = len(train_data)
                _tensor = torch.zeros((n_dataset, )).cuda()
                for _dataset_id in total_per_dataset:
                    _tensor[_dataset_id] = total_per_dataset[_dataset_id]

                if self.world_size > 1:
                    self.all_reduce(_tensor, op=dist.ReduceOp.SUM, group=self.group)

                self.print("Memory statistics:")
                _sum = torch.sum(_tensor).item()
                for dataset_id in total_per_dataset:
                    prob = _tensor[dataset_id].item() / _sum
                    self.print("Dataset ", dataset_id, _tensor[dataset_id].item(),
                               "samples", f"{prob:.0%}")
                    self.print("")

        return total_loss / total_words

    def compute_gradient_agem(self, summation=False):
        """

        Args:
            summation: we add the gradients when the gradients don't conflict

        Returns:

        """
        # in this code, ref is actually g in the paper
        # and mem is the g_ref in the paper
        nom = 0
        denom = 0
        clash = False

        if self.opt.flatten_parameters:
            ref_param = self.optim.flattened_params
            mem_param = self.mem_optim.flattened_params

            g_ref = ref_param.grad.data
            g_mem = mem_param.grad.data

            nom += torch.dot(g_ref, g_mem)

            if nom >= 0:
                if summation:
                    ref_param.grad.data.add_(mem_param.grad.data)

                return clash

            denom += torch.dot(g_mem, g_mem)
            projection_term = nom / denom
            ref_param.grad.data.sub_(mem_param.grad.data, alpha=projection_term)

            clash = True

            return clash

        else:
            ref_params = list(self.model.parameters())
            mem_params = list(self.mem_model.parameters())

            with torch.no_grad():

                projection_term = 0

                # g_ref = torch.cat([torch.flatten(p.grad.data) for p in ref_params if p.requires_grad]).view(1, -1)
                # g_mem = torch.cat([torch.flatten(p.grad.data) for p in mem_params if p.requires_grad]).view(1, -1)

                # self.print("computing projection term ...")
                for ref_param, mem_param in zip(ref_params, mem_params):
                    if ref_param.grad is None or mem_param.grad is None:
                        continue

                    if mem_param.grad is not None:
                        g_mem = torch.flatten(mem_param.grad.data)
                        denom += torch.dot(g_mem, g_mem)

                    assert (ref_param.numel() == mem_param.numel())

                    g_ref = torch.flatten(ref_param.grad.data)
                    g_mem = torch.flatten(mem_param.grad.data)

                    nom += torch.dot(g_ref, g_mem)
                    denom += torch.dot(g_mem, g_mem)

                # constraint satisfied
                if nom >= 0:
                    if summation:
                        for ref_param, mem_param in zip(ref_params, mem_params):
                            if mem_param.grad is None:
                                continue

                            # if the new model doesn't have any gradient for some reason -> continue
                            if ref_param.grad is None and mem_param.grad is None:
                                continue

                            if ref_param.grad is None:
                                ref_param.grad = mem_param.grad
                                continue

                            # g = g - projection_term * g_mem
                            ref_param.grad.data.add_(mem_param.grad.data)
                    return clash

                # constraint violated
                projection_term = nom/denom

                if torch.isnan(projection_term):
                    projection_term.fill_(1.0)
                # self.print("done", projection_term, flush=True)
                # self.print("computing agem grads ...")

                # probably we only need 1 extra model
                for ref_param, mem_param in zip(ref_params, mem_params):
                    if mem_param.grad is None:
                        continue

                    # if the new model doesn't have any gradient for some reason -> continue
                    if ref_param.grad is None and mem_param.grad is None:
                        continue

                    if ref_param.grad is None:
                        ref_param.grad = mem_param.grad * projection_term
                        continue

                    # g = g - projection_term * g_mem
                    ref_param.grad.data.sub_(mem_param.grad.data, alpha=projection_term)

                clash = True

                return clash
            # self.print("done")

    def update_agem(self):

        self.optim

    def train_epoch_agem(self, train_data, valid_data, epoch, resume=False, itr_progress=None):

        opt = self.opt
        streaming = False
        grad_norm = -1

        # Clear the gradients of the model
        self.optim.zero_grad(set_to_none=not opt.true_zero_grad)
        self.mem_optim.zero_grad(set_to_none=not opt.true_zero_grad)

        # note: for Training split_even=True
        dataset = train_data

        assert is_list(dataset)

        data_iterators = list()

        for _dataset in dataset:
            data_iterator = generate_data_iterator(_dataset, self.rank, self.world_size,
                                                   seed=self.opt.seed, num_workers=opt.num_workers,
                                                   epoch=epoch, buffer_size=opt.buffer_size, split_even=True,
                                                   dataset_ids=None)

            data_iterators.append(data_iterator)

        # TODO: fix resume which is currently buggy
        if resume:
            data_iterator.load_state_dict(itr_progress)

        epoch_iterators = list()

        for _data_iterator in data_iterators:
            epoch_iterators.append(_data_iterator.next_epoch_itr(not streaming, pin_memory=opt.pin_memory))

        total_tokens, total_loss, total_words = zero_tensor(), zero_tensor(), zero_tensor()
        total_non_pads = zero_tensor()
        report_loss, report_tgt_words = zero_tensor(), zero_tensor()
        report_ctc_loss = zero_tensor()
        report_transducer_loss = zero_tensor()
        report_ewc_loss = zero_tensor()
        report_ctc_targets = zero_tensor()
        report_transducer_targets = zero_tensor()
        report_ewc_count = 0
        report_src_words = zero_tensor()
        report_sents = zero_tensor()
        report_rec_loss, report_rev_loss, report_mirror_loss = zero_tensor(), zero_tensor(), zero_tensor()

        report_enc_lid_loss = zero_tensor()
        report_enc_lid_count = 0
        report_dec_lid_loss = zero_tensor()
        report_dec_lid_count = 0

        report_clash = 0

        start = time.time()

        counter = 0
        num_accumulated_words = zero_tensor()
        num_accumulated_sents = zero_tensor()
        report_contrastive_loss = zero_tensor()

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

        assert len(data_iterators) == len(epoch_iterators)
        assert len(data_iterators) == len(train_data)

        for dataset_id, (_data_iterator, _epoch_iterator) in enumerate(zip(data_iterators, epoch_iterators)):

            # maybe clean up everything from the last round?
            gc.collect()

            n_samples = len(_data_iterator) * 2 - 1 if (epoch > 1 or dataset_id > 0) else len(_data_iterator)
            i = _data_iterator.iterations_in_epoch
            i = i * self.world_size
            self.print("Training Epoch %d with AGEM - Round (dataset) %d" % (epoch, dataset_id))

            rehearse = False

            # because after 1 real sample -> 1 memory sample
            update_frequency = 2 * opt.update_frequency

            # stop_condition = (_data_iterator.end_of_epoch() and not rehearse)
            while not  (_data_iterator.end_of_epoch() and not rehearse):

                if not rehearse or opt.reservoir_size <= 0:
                    samples = next(_epoch_iterator)

                    # sample is also added to the reservoir here
                    batch = prepare_sample(samples, device=self.device, dataset_id=dataset_id,
                                           reservoir=self.reservoir)

                    rehearse = False
                    rehearsing = False

                    # if (epoch > 1 or dataset_id > 0) and self.reservoir is not None:
                    if self.reservoir is not None:
                        # we start to rehearse immediately
                        rehearse = True  # so that the next one is to rehearse
                else:
                    rehearsed_data = self.reservoir.sample()
                    rehearsed_dataset_ids = rehearsed_data[0]
                    rehearsed_indices = rehearsed_data[1]
                    samples = get_batch_from_multidataset(train_data, rehearsed_dataset_ids, rehearsed_indices)

                    batch = prepare_sample(samples, device=self.device)

                    rehearsing = True
                    rehearse = False

                targets = batch.get('target_output')

                # TODO: dealing with oom during distributed training
                oom = zero_tensor()
                counter = counter + 1
                reduce = True if counter >= (update_frequency - 1) else False

                current_model = self.model if not rehearsing else self.mem_model
                grad_scaler = self.grad_scaler if not rehearsing else self.mem_grad_scaler

                try:
                    def maybe_no_sync():
                        if not reduce and (isinstance(current_model, DDP_model) or isinstance(current_model, FSDP)):
                            return current_model.no_sync()
                        else:
                            # when we dont reach the updating step, we do not need to synchronize the gradients
                            # thus disabling the backward grad sync to improve speed
                            return contextlib.ExitStack()  # dummy contextmanager

                    with maybe_no_sync():

                        # self.print(counter, "rehearsing:", rehearsing, "syncing:", reduce, flush=True)

                        with autocast(enabled=opt.fp16, dtype=torch.bfloat16 if self.bf16_ready else torch.float16):

                            tgt_mask = targets.ne(onmt.constants.PAD)
                            outputs = current_model( batch, streaming=False, target_mask=tgt_mask,
                                                     zero_encoder=opt.zero_encoder,
                                                     mirror=opt.mirror_loss > 0.0,
                                                     adv_ptb_grad=False,
                                                     checkpointing_ffn=opt.checkpointing_ffn,
                                                     checkpointing_cross_attn=opt.checkpointing_cross_attn,
                                                     checkpointing_self_attn=opt.checkpointing_self_attn,
                                                     ctc_loss_function=self.ctc_loss_function,
                                                     ctc_labels=targets,
                                                     grad_scaler=self.grad_scaler,
                                                     ctc_coeff=opt.ctc_loss if self.optim._step > opt.ctc_loss_delay else 0.0,
                                                     transducer_loss_function=self.transducer_loss_function,
                                                     transducer_coeff=opt.transducer_loss
                                                 )

                            batch_size = batch.size
                            # outputs is a dictionary containing keys/values necessary for loss function
                            # can be flexibly controlled within models for easier extensibility
                            outputs['tgt_mask'] = tgt_mask

                            ctc_only = False
                            if outputs["hidden"] != None:
                                loss_dict = self.loss_function(outputs, targets, model=self.model)
                                loss_data = loss_dict['data']
                                loss = loss_dict['loss']  # a little trick to avoid gradient overflow with fp16
                                full_loss = loss
                            else:
                                ctc_only = True
                                loss_data = 0
                                loss = None
                                full_loss = 0

                            if opt.ctc_loss > 0.0:
                                ctc_loss = outputs['ctc_loss']
                                n_ctc_targets = outputs['n_ctc_targets']
                                # TODO: add CTC loss to models
                                ctc_loss_data = ctc_loss.item()
                                full_loss = full_loss + opt.ctc_loss * ctc_loss
                            else:
                                n_ctc_targets = 0
                                ctc_loss_data = 0

                            if opt.transducer_loss > 0.0:
                                transducer_loss = outputs['transducer_loss']
                                n_transducer_targets = outputs['transducer_numel']
                                # TODO: add CTC loss to models
                                transducer_loss_data = transducer_loss.item()
                                full_loss = full_loss + opt.transducer_loss * transducer_loss
                            else:
                                n_transducer_targets = 0
                                transducer_loss_data = 0

                            if opt.mirror_loss:
                                rev_loss = loss_dict['rev_loss']
                                rev_loss_data = loss_dict['rev_loss_data']
                                mirror_loss = loss_dict['mirror_loss']
                                full_loss = full_loss + rev_loss + mirror_loss
                                mirror_loss_data = loss_dict['mirror_loss'].item()
                            else:
                                rev_loss_data = None
                                mirror_loss_data = 0

                            if opt.predict_language > 0:
                                enc_pred_lang = outputs['enc_pred_lang']
                                enc_mask = outputs['src_mask']
                                enc_lid_loss = self.lid_loss_function(enc_pred_lang,
                                                                      batch.get("source_lang"), enc_mask)

                                dec_pred_lang = outputs['dec_pred_lang']
                                # dec_mask = outputs['target_mask']
                                # dec_mask = targets.eq(onmt.constants.PAD)
                                dec_mask = batch.get('target_input_selfattn_mask')
                                dec_lid_loss = self.lid_loss_function(dec_pred_lang,
                                                                      batch.get("target_lang"), dec_mask)

                                full_loss = full_loss + 0.01 * (enc_lid_loss + dec_lid_loss)

                                report_enc_lid_loss.add_(enc_lid_loss.item())
                                report_enc_lid_count += enc_mask.ne(1).int().sum().item()

                                report_dec_lid_loss.add_(dec_lid_loss.item())
                                report_dec_lid_count += dec_mask.ne(1).int().sum().item()

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

                            if hasattr(opt, "use_memory") and opt.use_memory and "loss_memory" in outputs:
                                loss_memory = outputs['loss_memory']
                                # full_loss = full_loss + loss_memory
                                full_loss = loss_memory

                            if opt.contrastive_loss_coeff > 0 and 'contrastive_loss' in outputs:
                                contrastive_loss = outputs['contrastive_loss']
                                full_loss = full_loss + opt.contrastive_loss_coeff * contrastive_loss
                                report_contrastive_loss.add_(contrastive_loss.item())

                            # correct, total = loss_dict['correct'], loss_dict['total']
                            # optimizer = self.optim.optimizer

                        # TODO for adversarial:
                        # grad_list = [p for p in self.model.parameters() if p.requires_grad]
                        # if opt.virtual_adversarial_training_mode > 0:
                        #     # if we use virtual adversarial training: add the input to the list of gradient to take
                        #     model_input = outputs['source']
                        #     vanilla_logits = outputs['logprobs']
                        #     grad_list += [model_input]
                        # else:
                        #     model_input = None
                        #     vanilla_logits = None

                        # grad scaler has to be done outside of the autocast
                        if self.grad_scaler is not None:
                            grad_scaler.scale(full_loss).backward()
                        else:
                            full_loss.backward()

                        # EWC training: no need for autograd here?
                        if self.optim._step % opt.ewc_decay_every == 0:
                            ewc_importance = ewc_importance / opt.ewc_decay_scale

                        # only run this ewc everytime we reduce

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
                update_flag = counter >= update_frequency

                if update_flag:
                    # accumulated gradient case, in this case the update frequency
                    self.all_reduce(num_accumulated_words, op=dist.ReduceOp.SUM, group=self.group)

                    grad_denom = 1.0

                    if self.grad_scaler is not None:
                        self.grad_scaler.unscale_(self.optim.optimizer)
                        self.mem_grad_scaler.unscale_(self.mem_optim.optimizer)

                    clash = self.compute_gradient_agem(summation=opt.agem_summation)
                    if clash: report_clash += 1

                    if self.opt.normalize_gradient:
                        grad_denom = num_accumulated_words.item() * grad_denom
                    else:
                        grad_denom = 1

                    params = self.optim.get_params()
                    # When we accumulate the gradients, each gradient is already normalized by a constant grad_scaler
                    if grad_denom != 1 and not self.opt.fsdp:
                        normalize_gradients(params, grad_denom)

                    # Update the pagrameters.
                    if self.opt.fsdp:
                        if self.opt.max_grad_norm > 0:

                            grad_norm = self.model.clip_grad_norm_(self.opt.max_grad_norm)
                        else:
                            grad_norm = -1
                    else:
                        grad_norm = clip_grad_norm(params, self.opt.max_grad_norm)
                        # _ = clip_grad_norm(self.mem_model.parameters(), self.opt.max_grad_norm)

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


                    if self.grad_scaler is not None:
                        self.optim.step(scaler=self.grad_scaler)
                        self.grad_scaler.update()

                        # self.mem_optim.step(scaler=self.grad_scaler)
                        self.mem_grad_scaler.update()
                    else:
                        self.optim.step(scaler=None)

                    # syncrhonize between 2 models
                    self.mem_model.load_state_dict(self.model.state_dict())

                    self.optim.zero_grad(set_to_none=True if self.opt.fsdp else not opt.true_zero_grad)
                    self.mem_optim.zero_grad(set_to_none=True if self.opt.fsdp else not opt.true_zero_grad)
                    counter = 0
                    num_accumulated_words.zero_()
                    num_accumulated_sents.zero_()

                    num_updates = self.optim._step
                    if (opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every) \
                            or (num_updates >= opt.max_step):
                        torch.cuda.synchronize()
                        valid_loss, valid_accuracy = self.eval(valid_data)
                        valid_ppl = math.exp(min(valid_loss, 100))

                        self.print('Validation perplexity: %g' % valid_ppl)
                        self.print('Validation accuracy: %g percent' % (100 * valid_accuracy))
                        ep = float(epoch) - 1. + ((float(i) + 1.) / n_samples)
                        if opt.save_metrics in ['ppl', 'perplexity']:
                            value = valid_ppl
                        elif opt.save_metrics == "memory":
                            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                                value = self.model.module.choose_best_epoch_by
                            else:
                                value = self.model.choose_best_epoch_by
                        else:
                            value = 1 - valid_accuracy
                        self.save(ep, value,
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
                    report_ctc_targets.add_(n_ctc_targets)

                if opt.transducer_loss > 0.0:
                    report_transducer_loss.add_(transducer_loss_data)
                    report_transducer_targets.add_(n_transducer_targets)

                # control the index a little bit to ensure the log is always printed
                if i == 0 or ((i + 1) % opt.log_interval < self.world_size) and not rehearsing:

                    self.all_reduce(report_loss, op=dist.ReduceOp.SUM, group=self.group)
                    # self.all_reduce(report_ewc_loss, op=dist.ReduceOp.SUM, group=self.group)
                    self.all_reduce(report_tgt_words, op=dist.ReduceOp.SUM, group=self.group)
                    self.all_reduce(report_src_words, op=dist.ReduceOp.SUM, group=self.group)
                    # self.all_reduce(report_sents, op=dist.ReduceOp.SUM, group=self.group)
                    # self.all_reduce(report_contrastive_loss, op=dist.ReduceOp.SUM, group=self.group)
                    if opt.ctc_loss > 0.0:
                        self.all_reduce(report_ctc_loss, op=dist.ReduceOp.SUM, group=self.group)
                        self.all_reduce(report_ctc_targets, op=dist.ReduceOp.SUM, group=self.group)

                    if opt.transducer_loss > 0.0:
                        self.all_reduce(report_transducer_loss, op=dist.ReduceOp.SUM, group=self.group)
                        self.all_reduce(report_transducer_targets, op=dist.ReduceOp.SUM, group=self.group)

                    if self.is_main():

                        if ctc_only:
                            log_string = ("Epoch %2d, Rd %d, %5d/%5d; ; grad_norm: %6.4f ; clash: %d" %
                                          (epoch, dataset_id, i + 1, len(_data_iterator),
                                           grad_norm, report_clash))
                        else:
                            log_string = ("Ep %2d, Rd %d, %5d/%5d; ; ppl: %6.2f ; grad_norm: %6.4f ; clash: %d " %
                                          (epoch, dataset_id, i + 1, len(_data_iterator),
                                           math.exp(report_loss.item() / report_tgt_words.item()),
                                           grad_norm, report_clash))

                        if opt.mirror_loss:
                            self.all_reduce(report_rev_loss, op=dist.ReduceOp.SUM, group=self.group)
                            rev_ppl = math.exp(report_rev_loss.item() / report_tgt_words.item())
                            log_string += (" rev_ppl: %6.2f ; " % rev_ppl)
                            log_string += (" mir_loss: %6.2f ; " % (report_mirror_loss / report_tgt_words))

                        if opt.ctc_loss > 0.0:
                            ctc_loss_string = report_ctc_loss.item() / report_ctc_targets.item()
                            log_string += (" ctc_ppl: %5.2f ; " % math.exp(ctc_loss_string))

                        if opt.transducer_loss > 0.0:
                            transducer_loss_string = report_transducer_loss.item() / report_transducer_targets.item()
                            log_string += (" trc_ppl: %5.2f ; " % math.exp(transducer_loss_string))

                        if opt.contrastive_loss_coeff > 0.0:
                            #
                            ctv_loss = report_contrastive_loss.item() / report_tgt_words.item()
                            log_string += (" ctv_loss: %8.2f ; " % ctv_loss)

                        if ewc_importance > 0.0:
                            try:
                                _ewc_loss = report_ewc_loss.item() / report_ewc_count
                            except ZeroDivisionError:
                                _ewc_loss = float('nan')
                            log_string += (" ewcloss: %8.8f ; " % _ewc_loss)

                        if opt.predict_language > 0:
                            try:
                                _enc_lid_loss = report_enc_lid_loss.item() / report_enc_lid_count
                                _dec_lid_loss = report_dec_lid_loss.item() / report_dec_lid_count
                            except ZeroDivisionError:
                                _enc_lid_loss = float('nan')
                                _dec_lid_loss = float('nan')
                            log_string += (" enc_lidloss: %8.8f ; " % _enc_lid_loss)
                            log_string += (" dec_lidloss: %8.8f ; " % _dec_lid_loss)

                        log_string += ("lr: %.7f ; updates: %7d; " %
                                       (self.optim.get_learning_rate(),
                                        self.optim._step))

                        src_speed = report_src_words.item() / (time.time() - start)
                        src_speed = human_format(src_speed)

                        tgt_speed = report_tgt_words.item() / (time.time() - start)
                        tgt_speed = human_format(tgt_speed)

                        log_string += ("%s src tok/s; %s tgt tok/s; " %
                                       (src_speed, tgt_speed))

                        log_string += ("%s" %
                                       str(datetime.timedelta(seconds=int(time.time() - self.start_time))))

                        self.print(log_string, flush=True)

                    report_loss.zero_()
                    report_tgt_words.zero_()
                    report_src_words.zero_()
                    report_rec_loss.zero_()
                    report_rev_loss.zero_()
                    report_mirror_loss.zero_()
                    report_ctc_loss.zero_()
                    report_ctc_targets.zero_()

                    report_transducer_loss.zero_()
                    report_transducer_targets.zero_()

                    report_ewc_loss.zero_()
                    report_ewc_count = 0
                    # report_sents.zero_()
                    if report_contrastive_loss is not None:
                        report_contrastive_loss.zero_()
                    start = time.time()

                # increase i by world size
                if not rehearsing:
                    i = i + self.world_size

            # END OF ROUND -> run validation and save
            # we run validation on all valid datasets
            valid_loss, valid_accuracy = self.eval(valid_data)
            valid_ppl = math.exp(min(valid_loss, 100))
            self.print('[INFO] Validation perplexity: %g' % valid_ppl)
            self.print('[INFO] Validation accuracy: %g percent' % (100 * valid_accuracy))
            if opt.save_metrics in ['ppl', 'perplexity']:
                value = valid_ppl
            elif opt.save_metrics == "memory":
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                    value = self.model.module.choose_best_epoch_by
                else:
                    value = self.model.choose_best_epoch_by
            else:
                value = 1 - valid_accuracy
            self.save(epoch, dataset_id, value)

            total_per_dataset = self.reservoir.get_stats()

            # some stupid code to grab the memory statistics
            n_dataset = len(train_data)
            _tensor = torch.zeros((n_dataset,)).cuda()
            for _dataset_id in total_per_dataset:
                _tensor[_dataset_id] = total_per_dataset[_dataset_id]

            if self.world_size > 1:
                self.all_reduce(_tensor, op=dist.ReduceOp.SUM, group=self.group)

            self.print("Memory statistics:")
            _sum = torch.sum(_tensor).item()
            for dataset_id in total_per_dataset:
                prob = _tensor[dataset_id].item() / _sum
                self.print("Dataset ", dataset_id, _tensor[dataset_id].item(),
                           "samples", f"{prob:.0%}")
                self.print("")

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
        self.optim.zero_grad(set_to_none=not opt.true_zero_grad)
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
        report_ctc_targets = zero_tensor()
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
                    if not reduce and (isinstance(self.model, DDP_model) or isinstance(self.model, FSDP)):
                        return self.model.no_sync()
                    else:
                        # when we dont reach the updating step, we do not need to synchronize the gradients
                        # thus disabling the backward grad sync to improve speed
                        return contextlib.ExitStack()  # dummy contextmanager

                with maybe_no_sync():
                    with autocast(enabled=opt.fp16, dtype=torch.bfloat16 if self.bf16_ready else torch.float16):

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
                                             adv_ptb_grad=False,
                                             checkpointing_ffn=opt.checkpointing_ffn,
                                             checkpointing_cross_attn=opt.checkpointing_cross_attn,
                                             checkpointing_self_attn=opt.checkpointing_self_attn,
                                             ctc_loss_function = self.ctc_loss_function,
                                             ctc_labels = targets,
                                             grad_scaler = self.grad_scaler
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
                            ctc_loss = outputs['ctc_loss']
                            full_loss = full_loss + ctc_loss

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

                        # correct, total = loss_dict['correct'], loss_dict['total']
                        # optimizer = self.optim.optimizer

                    # grad scaler has to be done outside of the autocast

                    # TODO for adversarial:

                    if self.grad_scaler is not None:
                        self.grad_scaler.scale(full_loss).backward()
                    else:
                        full_loss.backward()

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
            if self.grad_scaler is not None:
                self.grad_scaler.unscale_(self.optim.optimizer)

                # fake update. we need a learning rate = 0 for this
                # self.optim.step(scaler=self.grad_scaler)
                self.grad_scaler.update()
            grad_norm = clip_grad_norm(self.model.parameters(), 0)

            # Update the precision matrices.

            for n, p in parameters.items():
                if n in precision_matrices:
                    grad = p.grad.data
                    grad.masked_fill_(torch.logical_or(torch.isinf(grad), torch.isnan(grad)), 0)

                    precision_matrices[n].add_(torch.square(p.grad.data))

            self.optim.zero_grad(set_to_none=not opt.true_zero_grad)
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
                    log_string = ("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; grad_norm: %6.4f" %
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

        if opt.cache_encoder_output:
            import onmt.data.indexed_file as indexed_file
            import hashlib

            worked = False
            for datasets in [valid_data, train_data]:
                for i, dataset in enumerate(datasets):
                    # cache decoder output
                    id = hashlib.sha256(dataset.src_sizes.tobytes()).hexdigest()

                    basename_e = opt.cache_dir + "encoder_features_" + id
                    name1_e = basename_e + ".data"
                    name2_e = basename_e + ".index"

                    # basename_d = opt.cache_dir + "decoder_output_" + id

                    # name1_d = basename_d + ".data"
                    # name2_d = basename_d + ".index"
                    # name3_d = basename_d + ".label"

                    if os.path.isfile(name1_e) and os.path.isfile(
                            name2_e):  # and os.path.isfile(name1_d) and os.path.isfile(name2_d) and os.path.isfile(name3_d):
                        dataset.encoder_feature_files = (name1_e, torch.load(name2_e))
                        # dataset.decoder_feature_files = (name1_d,torch.load(name2_d),name3_d)
                        print("Using cached features:", id, "for dataset", i)
                        continue

                    print("Caching features:", id)
                    # continue
                    breakpoint()

                    dataset.index = 0  # when caching features, use indices only within one dataset
                    dataset.anz_sets = 1
                    worked = True

                    self.model.eval()

                    with open(name2_e, "w") as f:
                        f.write("In progress")

                    file_data_e = open(name1_e, "wb")
                    file_index_e = {}

                    # file_data_d = open(name1_d, "wb")
                    # file_index_d = {}
                    # label_d = {}

                    iterator = generate_data_iterator(dataset, 0, 1, self.opt.seed, num_workers=8)
                    iterator = iterator.next_epoch_itr(shuffle=False)

                    for samples in tqdm(iterator):
                        batch = prepare_sample(samples, device=self.device)

                        with torch.no_grad():
                            with autocast(enabled=opt.fp16, dtype=torch.bfloat16 if self.bf16_ready else torch.float16):
                                output = self.model(batch)

                        context = output["context"].transpose(1, 0)
                        mask = output["src_mask"].eq(0).sum(-1)
                        indices = batch.tensors["indices"]

                        for index, con, frames in zip(indices, context, mask):
                            data = con[:frames]
                            indexed_file.add_data(file_data_e, file_index_e, index, data)

                        """context = output["hidden"].transpose(1,0)
                        labels = batch.tensors["target_output"].transpose(1,0)
                        mask = labels.ge(2).sum(-1)

                        for index, con, tokens, label in zip(indices,context,mask,labels):
                            data = con[:tokens]
                            indexed_file.add_data(file_data_d, file_index_d, index, data)
                            label_d[index] = label[:tokens]"""

                    file_data_e.close()
                    torch.save(file_index_e, name2_e)

                    """file_data_d.close()
                    torch.save(file_index_d, name2_d)
                    torch.save(label_d, name3_d)"""

                    print("Finished caching features:", id)

                if worked:
                    print("Caching finished, exiting.")
                    sys.exit()

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

            self.print('[INFO] Validation perplexity: %g' % valid_ppl, flush=True)
            # percent is never used in plural :)
            self.print('[INFO] Validation accuracy: %g percent' % (100 * valid_accuracy))

            if opt.max_step <= 0:
                self.save(0, valid_ppl if opt.save_metrics in ['ppl', 'perplexity'] else 1 - valid_accuracy)

                return

        self.start_time = time.time()   

        for epoch in range(start_epoch, start_epoch + opt.epochs):
            self.print('')

            #  (1) train for one epoch on the training set
            if self.opt.agem_training:
                train_loss = self.train_epoch_agem(train_data, valid_data, epoch,
                                          resume=resume, itr_progress=itr_progress)
            else:
                train_loss = self.train_epoch(train_data, valid_data, epoch,
                                            resume=resume, itr_progress=itr_progress)
            train_ppl = math.exp(min(train_loss, 100))
            self.print('[INFO] Train perplexity: %g' % train_ppl)

            #  (2) evaluate on the validation set
            #  done after every round

            # valid_loss, valid_accuracy = self.eval(valid_data)
            # valid_ppl = math.exp(min(valid_loss, 100))
            #
            # self.print('[INFO] Validation perplexity: %g' % valid_ppl)
            # self.print('[INFO] Validation accuracy: %g percent' % (100 * valid_accuracy))
            #
            # if opt.save_metrics in ['ppl', 'perplexity']:
            #     value = valid_ppl
            # elif opt.save_metrics == "memory":
            #     if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            #         value = self.model.module.choose_best_epoch_by
            #     else:
            #         value = self.model.choose_best_epoch_by
            # else:
            #     value = 1 - valid_accuracy
            # self.save(epoch, value)

            itr_progress = None
            resume = False


