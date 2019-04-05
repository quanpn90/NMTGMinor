from __future__ import division

import math
import torch
from collections import defaultdict


import onmt


class Batch(object):
    "An object to manage the "
    def __init__(self, src_data, tgt_data=None,
                 src_type='text',
                 src_align_right=False, tgt_align_right=False):

        self.tensors = defaultdict(lambda: None)
        self.has_target = False
        self.src_type = src_type
        if src_data is not None:
            self.tensors['source'], self.src_lengths = self.collate(src_data,
                                                                    align_right=src_align_right,
                                                                    type=self.src_type)
            self.tensors['source'] = self.tensors['source'].transpose(0, 1).contiguous()
            # self.tensors['src_attn_mask'] = self.tensors['source'].eq(onmt.Constants.PAD).unsqueeze(1)
            # self.tensors['src_pad_mask'] = self.tensors['source'].ne(onmt.Constants.PAD)
            self.tensors['src_length'] = torch.LongTensor(self.src_lengths)
            self.src_size = sum(self.src_lengths)
        else:
            self.src_size = 0


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
        else:
            self.tgt_size = 0

        self.size = len(src_data) if src_data is not None else len(tgt_data)



    def collate(self, data, align_right=False, type="text"):

        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        # initialize with batch_size * length first
        if type == "text":
            tensor = data[0].new(len(data), max_length).fill_(onmt.Constants.PAD)

            for i in range(len(data)):
                data_length = data[i].size(0)
                offset = max_length - data_length if align_right else 0
                tensor[i].narrow(0, offset, data_length).copy_(data[i])

        elif type == "audio":
            # the last feature dimension is for padding or not, hence + 1
            tensor = data[0].new(len(data), max_length, data[0].size(1) + 1).fill_(onmt.Constants.PAD)

            for i in range(len(data)):
                data_length = data[i].size(0)
                offset = max_length - data_length if align_right else 0

                tensor[i].narrow(0, offset, data_length).narrow(1, 1, data[0].size(1)).copy_(data[i])
                tensor[i].narrow(0, offset, data_length).narrow(1, 0, 1).fill_(1)
        else:
            raise NotImplementedError

        return tensor, lengths

    def get(self, name):
        if name in self.tensors:
            return self.tensors[name]
        else:
            return None

    def cuda(self, fp16=False):
        for key, tensor in self.tensors.items():
            if tensor.type() == "torch.FloatTensor" and fp16:
                self.tensors[key] = tensor.half()
            self.tensors[key] = self.tensors[key].cuda()


class Dataset(object):
    def __init__(self, src_data, tgt_data, batch_size_words,
                 data_type="text", balance=False, batch_size_sents=128,
                 multiplier=1, sort_by_target=False):
        self.src = src_data
        self._type = data_type
        if tgt_data:
            self.tgt = tgt_data

            if src_data:
                assert(len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        self.fullSize = len(self.src) if self.src is not None else len(self.tgt)
        self.batch_size_words = batch_size_words

        self.balance = balance
        self.batch_size_sents = batch_size_sents 
        
        self.multiplier = multiplier
        self.sort_by_target = sort_by_target

        self.pad_count = True
        # if self.balance:
        self.allocateBatch()
        self.cur_index = 0
        self.batchOrder = None

    def size(self):

        return self.fullSize

    # This function allocates the mini-batches (grouping sentences with the same size)
    def allocateBatch(self):
            
        self.batches = []
        cur_batch = []
        cur_batch_size = 0
        cur_batch_sizes = [0]
        
        def oversize_(cur_batch):

            if len(cur_batch) == self.batch_size_sents:
                    return True

            if not self.pad_count:
                if cur_batch_size + sentence_length > self.batch_size_words:
                    return True
            else:
                if (max(max(cur_batch_sizes), sentence_length)) * (len(cur_batch)+1) > self.batch_size_words:
                    return True
            return False

        i = 0
        while i < self.fullSize:
        #~ for i in range(1, self.fullSize):

            if self.tgt is not None:
                sentence_length = self.tgt[i].size(0) - 1
            else:
                sentence_length = self.src[i].size(0)

            oversized = oversize_(cur_batch)
            # if the current length makes the batch exceeds
            # the we create a new batch
            if oversized:

                current_size = len(cur_batch)
                scaled_size = max(
                    self.multiplier * (current_size // self.multiplier),
                    current_size % self.multiplier)


                batch_ =  cur_batch[:scaled_size]
                self.batches.append(batch_) # add this batch into the batch list

                cur_batch = cur_batch[scaled_size:] # reset the current batch
                cur_batch_sizes = cur_batch_sizes[:-scaled_size]
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
        
        batch_ids = self.batches[index]
        if self.src:
            src_data = [self.src[i] for i in batch_ids]
        else:
            src_data = None

        if self.tgt:
            tgt_data = [self.tgt[i] for i in batch_ids]
        else:
            tgt_data = None

        batch = Batch(src_data, tgt_data=tgt_data,
                      src_align_right=False, tgt_align_right=False,
                      src_type=self._type)

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
