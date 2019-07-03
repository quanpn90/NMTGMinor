from __future__ import division

import math
import torch
from collections import defaultdict
import onmt
from onmt.speech.Augmenter import Augmenter


class Batch(object):
    # An object to manage the data within a minibatch
    def __init__(self, src_data, tgt_data=None,
                 src_atb_data=None, tgt_atb_data=None,
                 src_type='text',
                 src_align_right=False, tgt_align_right=False,
                 reshape_speech=0, augmenter=None):

        self.tensors = defaultdict(lambda: None)
        self.has_target = False
        self.src_type = src_type
        self.reshape_speech = reshape_speech
        if src_data is not None:
            self.tensors['source'], self.src_lengths = self.collate(src_data,
                                                                    align_right=src_align_right,
                                                                    type=self.src_type,
                                                                    augmenter=augmenter)
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

        if src_atb_data is not None:
            self.src_atb_data = dict()

            for i in src_atb_data:
                self.src_atb_data[i] = torch.cat(src_atb_data[i])

            self.tensors['src_atb'] = self.src_atb_data

        if tgt_atb_data is not None:
            self.tgt_atb_data = dict()

            for i in tgt_atb_data:
                self.tgt_atb_data[i] = torch.cat(tgt_atb_data[i])

            self.tensors['tgt_atb'] = self.tgt_atb_data

    # down sampling the speech signal by simply concatenating n features (reshaping)
    def downsample(self, data):

        if self.reshape_speech == 0:
            return data

        else:
            concat = self.reshape_speech
            tensor_ = data.float()  # adding float because of fp16 data storage
            add = (concat - tensor_.size()[0] % concat) % concat
            z = torch.FloatTensor(add, tensor_.size()[1]).zero_()

            # adding an additional dimension as padding
            tensor_ = torch.cat((tensor_, z), 0)
            tensor_ = tensor_.reshape((int(tensor_.size()[0] / concat), tensor_.size()[1] * concat))

            return tensor_

    def augment_speech(self):

        return

    def collate(self, data, align_right=False, type="text", augmenter=None):

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

            def find_length(x, concat):

                add = (concat - x.size(0) % concat) % concat

                return int((x.size(0) + add) / concat)

            if self.reshape_speech >= 1:
                lengths = [find_length(x, self.reshape_speech) for x in data]

            # allocate data for the batch speech
            feature_size = data[0].size(1) * self.reshape_speech if self.reshape_speech >= 1 else data[0].size(1)
            batch_size = len(data)
            tensor = data[0].float().new(batch_size, max_length, feature_size + 1).fill_(onmt.Constants.PAD)

            for i in range(len(data)):

                sample = data[i]

                if augmenter is not None:
                    sample = augmenter.augment(sample)

                feature = self.downsample(sample)

                data_length = feature.size(0)
                offset = max_length - data_length if align_right else 0

                tensor[i].narrow(0, offset, data_length).narrow(1, 1, feature.size(1)).copy_(feature)
                # padding dimension: 1 is not padded, 1 is padded
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
            if isinstance(tensor, dict):
                for k in tensor:
                    tensor[k].cuda()
            else:
                if tensor.type() == "torch.FloatTensor" and fp16:
                    self.tensors[key] = tensor.half()
                self.tensors[key] = self.tensors[key].cuda()


class Dataset(object):

    def __init__(self, src_data, tgt_data,
                 src_atbs=None, tgt_atbs=None,
                 batch_size_words=2048,
                 data_type="text", batch_size_sents=128,
                 multiplier=1, sort_by_target=False,
                 reshape_speech=0, augment=False):
        self.src = src_data
        self._type = data_type
        self.reshape_speech = reshape_speech
        if tgt_data:
            self.tgt = tgt_data

            if src_data:
                assert(len(self.src) == len(self.tgt))
        else:
            self.tgt = None

        self.src_atbs=src_atbs
        self.tgt_atbs=tgt_atbs

        self.fullSize = len(self.src) if self.src is not None else len(self.tgt)

        # maximum number of tokens in a mb
        self.batch_size_words = batch_size_words

        # maximum sequences in a mb
        self.batch_size_sents = batch_size_sents

        # the actual batch size must divide by this multiplier (for fp16 it has to be 4 or 8)
        self.multiplier = multiplier

        # by default the minibatch length is sort by target
        self.sort_by_target = sort_by_target

        # by default: count the amount of padding when we group mini-batches
        self.pad_count = True

        # group samples into mini-batches
        self.batches = []
        self.allocate_batch()

        self.cur_index = 0
        self.batchOrder = None

        if augment:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None

    def size(self):

        return self.fullSize

    def switchout(self, batch):

        pass

    # This function allocates the mini-batches (grouping sentences with the same size)
    def allocate_batch(self):

        cur_batch = []
        cur_batch_size = 0
        cur_batch_sizes = []
        
        def oversize_(cur_batch, sent_size):

            if len(cur_batch) >= self.batch_size_sents:
                return True

            if not self.pad_count:
                if cur_batch_size + sent_size > self.batch_size_words:
                    return True
            else:
                if len(cur_batch_sizes) == 0:
                    return False

                if (max(max(cur_batch_sizes), sent_size)) * (len(cur_batch)+1) > self.batch_size_words:
                    return True
            return False

        i = 0
        while i < self.fullSize:

            if self.tgt is not None and self.src is not None:
                sentence_length = max(self.tgt[i].size(0) - 1, self.src[i].size(0))
                # print(sentence_length)
            elif self.tgt is not None:
                sentence_length = self.tgt[i].size(0) - 1
            else:
                sentence_length = self.src[i].size(0)

            oversized = oversize_(cur_batch, sentence_length)
            # if the current item makes the batch exceed max size
            # then we create a new batch
            if oversized:
                # cut-off the current list to fit the multiplier
                current_size = len(cur_batch)
                scaled_size = max(
                    self.multiplier * (current_size // self.multiplier),
                    current_size % self.multiplier)

                batch_ = cur_batch[:scaled_size]
                self.batches.append(batch_)  # add this batch into the batch list

                cur_batch = cur_batch[scaled_size:]  # reset the current batch
                cur_batch_sizes = cur_batch_sizes[scaled_size:]
                cur_batch_size = sum(cur_batch_sizes)

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

        src_atb_data = None
        if self.src_atbs is not None:
            src_atb_data = dict()

            for i in self.src_atbs:
                src_atb_data[i] = [self.src_atbs[i][j] for j in batch_ids]

        tgt_atb_data = None
        if self.tgt_atbs is not None:
            tgt_atb_data = dict()

            for i in self.tgt_atbs:
                tgt_atb_data[i] = [self.tgt_atbs[i][j] for j in batch_ids]

        batch = Batch(src_data, tgt_data=tgt_data,
                      src_atb_data=src_atb_data, tgt_atb_data=tgt_atb_data,
                      src_align_right=False, tgt_align_right=False,
                      src_type=self._type, reshape_speech=self.reshape_speech,
                      augmenter=self.augmenter)

        return batch

    def __len__(self):
        return self.num_batches

    # genereate a new batch - order (static)
    def create_order(self, random=True):
        
        if random:
            self.batchOrder = torch.randperm(self.num_batches)
        else:
            self.batchOrder = torch.arange(self.num_batches).long()

        self.cur_index = 0
        
        return self.batchOrder

    # return the next batch according to the iterator
    def next(self, curriculum=False, reset=True, split_sizes=1):

        # reset iterator if reach data size limit
        if self.cur_index >= self.num_batches:
            if reset:
                self.cur_index = 0
            else:
                return None

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


# LANGUAGE MODEL DATASET AND DATAHOLDER
class LMBatch(Batch):

    def __init__(self, input, target=None):

        self.tensors = defaultdict(lambda: None)

        self.tensors['target_input'] = input  # T x B
        self.tensors['target_output'] = target # T x B or None

        # batch size
        self.size = input.size(1)
        self.length = input.size(0)

        self.tgt_size = self.size * self.length
        self.src_size = 0

    def collate(self, **kwargs):

        raise NotImplementedError


class LanguageModelDataset(Dataset):

    def __init__(self, data, batch_size_sents=128, seq_length=128):

        self.data = data

        self.batch_size_sents = batch_size_sents

        self.seq_length = seq_length

        # group samples into mini batches
        self.num_batches = 0
        self.allocate_batch()

        self.fullSize = self.num_batches
        # self.cur_index = 0
        # self.batchOrder = None

    def allocate_batch(self):

        nsequence = self.data.size(0) // self.batch_size_sents

        self.data = self.data.narrow(0, 0, nsequence * self.batch_size_sents)

        # Evenly divide the data across the bsz batches.
        self.data = self.data.view(self.batch_size_sents, -1).t().contiguous()

        # self.num_steps = nbatch - 1

        self.num_batches = math.ceil( ( self.data.size(0) - 1 ) / self.seq_length )

    # genereate a new batch - order (static)
    def create_order(self, random=False):

        # For language model order shouldn't be random
        if random:
            self.batchOrder = torch.randperm(self.num_batches)
        else:
            self.batchOrder = torch.arange(self.num_batches).long()

        self.cur_index = 0

        return self.batchOrder

    # return the next batch according to the iterator
    # for language model
    def next(self, curriculum=True, reset=True, split_sizes=1):

        # reset iterator if reach data size limit
        if self.cur_index >= self.num_batches:
            if reset:
                self.cur_index = 0
            else:
                return None

        batch_index = self.cur_index

        seq_len = self.seq_length

        top_index = min(batch_index + seq_len, self.data.size(0)-1)

        batch = LMBatch(self.data[batch_index:top_index], target=self.data[batch_index+1:top_index+1])

        # move the iterator one step
        self.cur_index += seq_len

        return [batch]

