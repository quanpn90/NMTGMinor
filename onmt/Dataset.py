from __future__ import division

import math
import torch
from torch.autograd import Variable

import onmt

class Dataset(object):
    '''
    batchSize is now changed to have word semantic (probably better)
    '''
    def __init__(self, srcData, tgtData, batchSize, cuda,
                 volatile=False, data_type="text", balance=False, max_seq_num=128):
        self.src = srcData
        self._type = data_type
        if tgtData:
            self.tgt = tgtData
            assert(len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        self.cuda = cuda
        self.fullSize = len(self.src)

        self.batchSize = batchSize
        #~ self.numBatches = math.ceil(len(self.src)/batchSize)
        self.volatile = volatile
        
        self.balance = balance
        self.max_seq_num = max_seq_num 
        
        # if self.balance:
        self.allocateBatch()
        # else:
            # self.numBatches = math.ceil(len(self.src)/batchSize)

    #~ # This function allocates the mini-batches (grouping sentences with the same size)
    def allocateBatch(self):
			
        # The sentence pairs are sorted by source already (cool)
        self.batches = []
        
        cur_batch = [0]
        cur_batch_size = self.src[0].size(0)
        
        for i in range(1, self.fullSize):
            
            sentence_length = self.src[i].size(0)
            # if the current length makes the batch exceeds
            # the we create a new batch
            
            if ( cur_batch_size + sentence_length > self.batchSize ) or len(cur_batch) == self.max_seq_num:
                self.batches.append(cur_batch) # add this batch into the batch list
                cur_batch = [] # reset the current batch
                cur_batch_size = 0
            
            cur_batch.append(i)
            cur_batch_size += sentence_length
                
            
        # catch the last batch
        if len(cur_batch) > 0:
            self.batches.append(cur_batch)
        
        self.numBatches = len(self.batches)
				
    def _batchify(self, data, align_right=False,
                  include_lengths=False, dtype="text"):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(onmt.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])
        if include_lengths:
            return out, lengths 
        else:
            return out
        
				
				
    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        
        batch = self.batches[index]
        srcData = [self.src[i] for i in batch]
        srcBatch, lengths = self._batchify(
            srcData,
            align_right=False, include_lengths=True, dtype=self._type)

        if self.tgt:
            tgtData = [self.tgt[i] for i in batch]
            tgtBatch = self._batchify(
                        tgtData,
                        dtype="text")
        else:
                tgtBatch = None
        

        def wrap(b, dtype="text"):
            if b is None:
                return b
            b = torch.stack(b, 0)
            if dtype == "text":
                b = b.t().contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b)
            return b

        srcTensor = wrap(srcBatch, self._type)
        tgtTensor = wrap(tgtBatch, "text")
        
        
        return (srcTensor, tgtTensor)
       

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(zip(self.src, self.tgt))
        self.src, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])

# class AugmentedDataset(object):
