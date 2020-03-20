from __future__ import division

import math
import torch
import torch.utils.data
from collections import defaultdict
import onmt
from onmt.data.dataset import Dataset


class LanguageModelBatch(object):

    def __init__(self, data, target, lang, **kwargs):

        self.data = data
        self.target = target
        self.lang = lang

        self.tensors = defaultdict(lambda: None)
        self.tensors['target_input'] = data
        self.tensors['target_output'] = target
        self.tensors['target_lang'] = lang

        self.tgt_size = target.numel()
        self.src_size = 0
        self.size = target.size(1)

    def get(self, name):
        if name in self.tensors:
            return self.tensors[name]
        else:
            return None

    def cuda(self, fp16=False):
        """
        Send the minibatch data into GPU. Old-fashioned without the 'device' control
        :param fp16:
        :return: None
        """
        for key, tensor in self.tensors.items():
            if isinstance(tensor, dict):
                for k in tensor:
                    v = tensor[k]
                    tensor[k] = v.cuda()
            elif tensor is not None:
                if tensor.type() == "torch.FloatTensor" and fp16:
                    self.tensors[key] = tensor.half()
                self.tensors[key] = self.tensors[key].cuda()
            else:
                continue


class LanguageModelDataset(Dataset):

    def __init__(self, data, langs, batch_size_sents=128, batch_size_words=9999,
                 seq_length=64, **kwargs):

        # concatenate all sentences in the data to get a stream
        if len(langs) <= 1:
            self.single_language = True
        else:
            self.single_language = False

        if not self.single_language:
            self.langs = [torch.Tensor([data[i].size(0)]).fill_(langs[i]) for i in range(len(langs))]
        else:
            self.langs = langs

        self.langs = torch.cat(self.langs, dim=0).long()
        self.data = torch.cat(data, dim=0).long()
            
        self.batch_size_sents = batch_size_sents
        self.batch_size_words = batch_size_words
        self.seq_length = seq_length
        self.bptt = seq_length

        full_length = sum([x.size(0) for x in data])
        # group samples into mini batches
        self.num_batches = 0
        self.batches = []
        self.allocate_batch()

        self.fullSize = self.num_batches
        self.cur_index = 0
        self.batchOrder = None

    def allocate_batch(self):

        self.n_step = self.data.size(0) // self.batch_size_sents

        self.data = self.data.narrow(0, 0, self.n_step * self.batch_size_sents)

        # Evenly divide the data across the bsz batches.
        self.data = self.data.view(self.batch_size_sents, -1).t().contiguous()

        # self.num_steps = nbatch - 1

        # self.num_batches = (self.n_step + self.seq_length - 1) // self.seq_length

        self.batches = []

        for i in range(0, self.data.size(0) - 1, self.bptt):
            bptt = self.seq_length
            seq_len = min(bptt, self.data.size(0) - 1 - i)

            end_idx = i + seq_len
            beg_idx = max(0, i)

            data = self.data[beg_idx:end_idx]
            target = self.data[i + 1:i + 1 + seq_len]

            if self.single_language:
                lang = self.langs
            else:
                lang = self.langs[beg_idx:end_idx]

            self.batches.append((data, target, lang))

        self.num_batches = len(self.batches)

    # genereate a new batch - order (static)
    def create_order(self, random=False):

        # For language model order shouldn't be random
        self.batchOrder = torch.arange(self.num_batches).long()

        self.cur_index = 0

        return self.batchOrder

    # return the next batch according to the iterator
    # for language model
    def next(self, curriculum=True, reset=True, split_sizes=1):

        # reset iterator if reach data size limit
        # if self.cur_index >= self.num_batches:
        #     if reset:
        #         self.cur_index = 0
        #     else:
        #         return None
        #
        # batch_index = self.cur_index
        #
        # seq_len = self.seq_length
        #
        # top_index = min(batch_index + seq_len, self.data.size(0) - 1)
        #
        # batch = LMBatch(self.data[batch_index:top_index], target=self.data[batch_index + 1:top_index + 1])
        #
        # # move the iterator one step
        # self.cur_index += seq_len
        if self.cur_index >= self.num_batches:
            if reset:
                self.cur_index = 0
            else:
                return None

        data, target, lang = self.batches[self.cur_index]
        batch = LanguageModelBatch(data, target, lang)
        self.cur_index += 1

        return [batch]
