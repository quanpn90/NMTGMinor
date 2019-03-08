from __future__ import division
import math
import torch
from collections import defaultdict

import onmt


class Batch(object):

    def __init__(self, src_data, tgt_data=None,
                 src_attbs=None, tgt_attbs=None,
                 src_align_right=False, tgt_align_right=False):

        self.tensors = defaultdict(lambda: None)
        self.has_target = False
        self.tensors['source'], self.src_lengths = self.collate(src_data, align_right=src_align_right)
        self.tensors['source'] = self.tensors['source'].t().contiguous()
        # self.tensors['src_attn_mask'] = self.tensors['source'].eq(onmt.Constants.PAD).unsqueeze(1)
        # self.tensors['src_pad_mask'] = self.tensors['source'].ne(onmt.Constants.PAD)
        self.tensors['src_length'] = torch.LongTensor(self.src_lengths)
        self.tensors['src_attbs'] = torch.LongTensor(src_attbs)

        # always need tgt attbs to know which language we translate to
        assert (tgt_attbs is not None)
        self.tensors['tgt_attbs'] = torch.LongTensor(tgt_attbs)

        if tgt_data is not None:
            target_full, self.tgt_lengths = self.collate(tgt_data, align_right=tgt_align_right)
            target_full = target_full.t().contiguous()
            self.tensors['target_input'] = target_full[:-1]
            self.tensors['target_output'] = target_full[1:]
            # self.tensors['tgt_pad_mask'] = self.tensors['target_input'].ne(onmt.Constants.PAD)
            # self.tensors['tgt_attn_mask'] = self.tensors['target_input'].ne(onmt.Constants.PAD)
            self.tensors['tgt_mask'] = self.tensors['target_output'].ne(onmt.Constants.PAD)
            # self.tensors['src_mask'] = self.tensors['source'].ne(onmt.Constants.PAD)
            self.tensors['tgt_length'] = torch.LongTensor(self.tgt_lengths)
            self.has_target = True
            self.tgt_size = sum([len(x) - 1 for x in tgt_data])

        self.size = len(src_data)

        self.src_size = sum([len(x)     for x in src_data])

    def collate(self, data, align_right=False):
    
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        # initialize with batch_size * length first
        tensor = data[0].new(len(data), max_length).fill_(onmt.Constants.PAD)
        
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            tensor[i].narrow(0, offset, data_length).copy_(data[i])
            
        return tensor, lengths
        
    def get(self, name):
        if name in self.tensors:
            return self.tensors[name]
        else:
            return None
            
    def cuda(self):
        for key, value in self.tensors.items():
            self.tensors[key] = value.cuda()


class Dataset(object):
    '''
    batchSize is now changed to have word semantic (probably better)
    '''
    def __init__(self, src_data, tgt_data, batch_size_words,
                 batch_size_sents=128,
                 multiplier=1):
        self.src = src_data['words']
        self.src_attbs = src_data['attbs']
        if tgt_data is not None:
            self.tgt = tgt_data['words']
            self.tgt_attbs = tgt_data['attbs']

            if self.tgt is not None:
                assert(len(self.src) == len(self.tgt))
        else:
            self.tgt = None

        assert(self.tgt_attbs is not None)
        self.fullSize = len(self.src)

        self.batch_size_words = batch_size_words
        self.batch_size_sents = batch_size_sents
        self.multiplier = multiplier

        self.cur_index = 0
        self.batchOrder = None

        self.batches = []

        self.num_batches = 0
        self.allocate_batch()

    # This function allocates the mini-batches (grouping sentences with the same size)
    def allocate_batch(self):

        # The sentence pairs are sorted by source already
        self.batches = []

        cur_batch = []
        cur_batch_size = 0
        cur_batch_sizes = []

        def oversize_(batch_):

            if len(batch_) == self.batch_size_sents:
                    return True

            # here we assume the new sentence's participation in the minibatch
            longest_length = sentence_length

            if len(cur_batch_sizes) > 0:
                longest_length = max(max(cur_batch_sizes), sentence_length)

            if longest_length * (len(cur_batch)+1) > self.batch_size_words:
                return True

        i = 0
        while i < self.fullSize:            
            sentence_length = max(self.src[i].size(0), self.tgt[i].size(0) - 1 if self.tgt is not None else 0)

            over_sized = oversize_(cur_batch)
            # if the current length makes the batch exceeds
            # the we create a new batch
            if over_sized:
                current_size = len(cur_batch)
                scaled_size = max(
                    self.multiplier * (current_size // self.multiplier),
                    current_size % self.multiplier)
               
                batch_ = cur_batch[:scaled_size]
                if self.multiplier > 1:
                    assert(len(batch_) % self.multiplier == 0), "batch size is not multiplied, current batch_size is %d " % len(batch_)
                self.batches.append(batch_) # add this batch into the batch list
                
                cur_batch = cur_batch[scaled_size:] # reset the current batch
                cur_batch_sizes = cur_batch_sizes[scaled_size:]
                cur_batch_size  = sum(cur_batch_sizes)

            cur_batch.append(i)
            cur_batch_size += sentence_length
            cur_batch_sizes.append(sentence_length)
            
            i = i + 1
            
        # catch the last batch
        if len(cur_batch) > 0:
            self.batches.append(cur_batch)
        
        self.num_batches = len(self.batches)

    def __getitem__(self, index):

        assert index < self.num_batches, "%d > %d" % (index, self.num_batches)

        
        batch = self.batches[index]
        src_data = [self.src[i] for i in batch]
        src_attbs =  [self.src_attbs[i] for i in batch]

        tgt_attbs = [self.tgt_attbs[i] for i in batch]

        if self.tgt:
            tgt_data = [self.tgt[i] for i in batch]
        else:
            tgt_data = None
            
        batch = Batch(src_data, tgt_data=tgt_data, src_align_right=False, tgt_align_right=False,    
                      src_attbs=src_attbs, tgt_attbs=tgt_attbs)

        return batch

    def __len__(self):
        return self.num_batches
        
    def create_order(self, random=True):
        
        if random:
            self.batchOrder = torch.randperm(self.num_batches)
        else:
            self.batchOrder = torch.arange(self.num_batches).long()
        self.cur_index = 0
        
        return self.batchOrder
        
    def next(self, curriculum=False, reset=True, split_sizes=1):
        
        # reset iterator if reach data size limit
        if self.cur_index >= self.num_batches:
            if reset:
                self.cur_index = 0
            else: return None
        
        if curriculum or self.batchOrder is None:
            batch_index = self.cur_index
        else:
            batch_index = self.batchOrder[self.cur_index]
            
        batch = self[batch_index]
        
        # move the iterator one step
        self.cur_index += 1

        return [batch]

    def shuffle(self):
        data = list(zip(self.src, self.tgt))
        self.src, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])
        
    def set_index(self, iteration):
        
        assert(iteration >= 0 and iteration < self.num_batches)
        self.cur_index = iteration
