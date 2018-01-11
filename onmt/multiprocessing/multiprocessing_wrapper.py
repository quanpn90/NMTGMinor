# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

"""
Train a network on multiple GPUs using multiprocessing.
"""

from itertools import cycle, islice
import math
import torch

from onmt.multiprocessing.multiprocessing_event_loop import MultiprocessingEventLoop, Future
import onmt.NoamOptim as NoamOptim


class MultiprocessingRunner(MultiprocessingEventLoop):
    """Main class for multi-GPU training.
    Each GPU has a full copy of the model and is assigned to its own Python
    process. Gradients are accumulated with all-reduce and all model replicas
    are updated synchronously after each batch.
    The methods in this class are divided into synchronous functions, which
    prepare and dispatch the input to each process, and asynchronous functions
    (prefixed with `_async_`), which run on each process in parallel.
    """
    
    def __init__(self, opt, model, criterion, device_ids=None,
                 multiprocessing_method='spawn'):
                     
        if device_ids is None:
            device_ids = tuple(range(torch.cuda.device_count()))
            
        super().__init__(device_ids, multiprocessing_method)
    
        if not torch.cuda.is_available():
            raise NotImplementedError('Training on CPU is not supported')
            
        model = model.share_memory()    
        #~ criterion = criterion.share_memory() # is this necessary ? maybe not 
        nccl_uid = nccl.get_unique_id()
        self.criterion = criterion
        
        Future.gen_list([
            self.call_async(rank, '_async_init', args=opt, model=model,
                            criterion=criterion, nccl_uid=nccl_uid)
            for rank in range(self.num_replicas)
        ])
        
        self._grads_initialized = False
        
    def _async_init(self, rank, device_id, args, model, criterion, nccl_uid):
        """Initialize child processes."""
        self.args = args

        # set CUDA device
        torch.cuda.set_device(device_id)

        # initialize NCCL
        nccl.initialize(self.num_replicas, nccl_uid, device_id)

        # copy model and criterion to current device
        self.model = model.cuda()
        self.criterion = criterion.cuda()

        # initialize optimizer and LR scheduler
        self.optimizer = self._build_optimizer()
        self.optimizer.set_parameters(self.model.parameters())
        
        self.loss = None
        self._max_bsz_seen = 0    
    
    
    def _build_optimizer(self):
        
        optimizer = NoamOptim(self.args)
        
        return optimizer
    
    def get_model(self):
        """Get one of the model replicas."""
        # just return the first model, since all replicas are the same
        return self.call_async(0, '_async_get_model').gen()
        
    def _async_get_model(self, rank, device_id):
        return self.model
        
    def save_checkpoint(self, filename, extra_info):
        """Save a checkpoint for the current model."""
        self.call_async(0, '_async_save_checkpoint', filename=filename, extra_info=extra_info).gen()
    
    def _async_save_checkpoint(self, rank, device_id, filename, extra_info):
        
        model_state_dict = self.model.state_dict()
        optim_state_dict = self.optimizer.state_dict()
        
        checkpoint = {
            'model': model_state_dict,
            'dicts': extra_info['dicts'],
            'opt': self.args,
            'epoch': extra_info['epoch'],
            'iteration' : extra_info['iteration'],
            'batchOrder' : extra_info['batchOrder'],
            'optim': optim_state_dict
        }
        
        #~ valid_ppl = extra_info['valid_ppl']
        #~ file_name = '%s_ppl_%.2f_e%.2f.pt' % (opt.save_model, valid_ppl, epoch)
        print('Writing model to %s' % filename)
        torch.save(checkpoint, filename)
        
    def load_checkpoint(self, checkpoint):
        """Load a checkpoint into the model replicas in each process."""
        results = Future.gen_list([
            self.call_async(rank, '_async_load_checkpoint', checkpoint=checkpoint)
            for rank in range(self.num_replicas)
        ])
        
        
    def _async_load_checkpoint(self, rank, device_id, filename):
        
        self.model.load_state_dict(checkpoint['model'])
        
        self.optimizer.load_state_dict(checkpoint['optim'])
        
        
    def set_seed(self, seed):
        Future.gen_list([
            self.call_async(rank, '_async_set_seed', seed=seed)
            for rank in range(self.num_replicas)
        ])
    
        
    def _async_set_seed(self, rank, device_id, seed):
        torch.manual_seed(seed)
        
        
    def _async_forward(self, rank, device_id, eval=False):
        if eval:
            self.model.eval()
        else:
            self.model.train()
            #~ self.optimizer.zero_grad()

        sample_size, logging_output, oom = 0, {}, False
        if self._sample is not None:
            try:
                # calculate loss and sample size
                self.loss, sample_size, logging_output = self.criterion(self.model, self._sample)
            except RuntimeError as e:
                if not eval and 'out of memory' in str(e):
                    print('| WARNING: ran out of memory on GPU #{}, skipping batch'.format(device_id))
                    oom = True
                    self.loss = None
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e

        return sample_size, logging_output, oom
        
        
    
    
    
    
    
    
    
    
    def _scatter_samples(self, samples, volatile=False, replace_empty_samples=False):
        """Split and distribute a sample across GPUs."""
        if not replace_empty_samples:
            # pad with None until its size is equal to the number of replicas
            samples = samples + [None]*(self.num_replicas - len(samples))
        else:
            # pad by cycling through the given samples
            samples = list(islice(cycle(samples), self.num_replicas))

        Future.gen_list([
            self.call_async(rank, '_async_prepare_sample', sample=samples[rank], volatile=volatile)
            for rank in range(self.num_replicas)
        ])

    def _async_prepare_sample(self, rank, device_id, sample, volatile):
        if sample is None:
            self._sample = None
        else:
            if hasattr(torch.cuda, 'empty_cache'):
                # clear the caching allocator if this is the largest sample we've seen
                if sample['target'].size(0) > self._max_bsz_seen:
                    self._max_bsz_seen = sample['target'].size(0)
                    torch.cuda.empty_cache()

            self._sample = utils.prepare_sample(sample, volatile=volatile, cuda_device=device_id)
    
    
    
    
    
    def _all_reduce_and_rescale_grads(self, grad_denom=1, buffer_size=10485760):
        """All-reduce and rescale gradients in chunks of the specified size."""
        grads = [p.grad.data for p in self.model.parameters() if p.requires_grad]
        buffer_t = grads[0].new(math.ceil(buffer_size / grads[0].element_size())).zero_()
        buffer = []

        def all_reduce_buffer():
            # copy grads into buffer_t
            offset = 0
            for g in buffer:
                numel = g.numel()
                buffer_t[offset:offset+numel].copy_(g.view(-1))
                offset += numel
            # all-reduce and rescale
            nccl.all_reduce(buffer_t[:offset])
            
            if grad_denom > 1:
                buffer_t.div_(grad_denom)
            # copy all-reduced buffer back into grads
            offset = 0
            for g in buffer:
                numel = g.numel()
                g.view(-1).copy_(buffer_t[offset:offset+numel])
                offset += numel

        filled = 0
        for g in grads:
            sz = g.numel() * g.element_size()
            if sz > buffer_size:
                # grad is bigger than buffer, all-reduce and rescale directly
                nccl.all_reduce(g)
                g.div_(grad_denom)
            elif filled + sz > buffer_size:
                # buffer is full, all-reduce and replace buffer with grad
                all_reduce_buffer()
                buffer = [g]
                filled = sz
            else:
                # add grad to buffer
                buffer.append(g)
                filled += sz
        if len(buffer) > 0:
            all_reduce_buffer()
        
