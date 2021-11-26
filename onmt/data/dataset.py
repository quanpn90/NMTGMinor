from __future__ import division

import math
import torch
import torch.utils.data
from collections import defaultdict
import onmt
from onmt.speech.Augmenter import Augmenter
from onmt.modules.dropout import switchout
import numpy as np
from .batch_utils import allocate_batch

"""
Data management for sequence-to-sequence models
Two basic classes: 
- Batch stores the input / output sequences, grouped into tensors with the same length (by padding)
- Dataset stores all of the data and 
"""


def merge_data(data, align_right=False, type='text', augmenter=None, upsampling=False,
               feature_size=40, dataname="source", src_pad="<blank>", tgt_pad="<blank>" ):
    """
            Assembling the individual sequences into one single tensor, included padding
            :param tgt_pad:
            :param src_pad:
            :param dataname:
            :param feature_size:
            :param upsampling:
            :param data: the list of sequences
            :param align_right: aligning the sequences w.r.t padding
            :param type: text or audio
            :param augmenter: for augmentation in audio models
            :return:
            """
    # initialize with batch_size * length
    # TODO: rewrite this function in Cython
    if type == "text":
        lengths = [x.size(0) for x in data]
        # positions = [torch.arange(length_) for length_ in lengths]
        max_length = max(lengths)
        # if max_length > 8:
        #     max_length = math.ceil(max_length / 8) * 8

        if dataname == "source":
            tensor = data[0].new(len(data), max_length).fill_(src_pad)
        elif dataname == "target":
            tensor = data[0].new(len(data), max_length).fill_(tgt_pad)
        else:
            print("Warning: check the dataname")
            raise NotImplementedError
        pos = None

        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            tensor[i].narrow(0, offset, data_length).copy_(data[i])

        return tensor, pos, lengths

    elif type in ["audio", "scp"]:

        # First step: on-the-fly processing for the samples
        # Reshaping: either downsampling or upsampling
        # On the fly augmentation
        samples = []

        for i in range(len(data)):
            sample = data[i]

            if augmenter is not None:
                sample = augmenter.augment(sample)

            if upsampling:
                sample = sample.view(-1, feature_size)

            samples.append(sample)

        # compute the lengths afte on-the-fly processing
        lengths = [x.size(0) for x in samples]

        max_length = max(lengths)
        # max_length = math.ceil(max_length / 8) * 8

        # allocate data for the batch speech
        feature_size = samples[0].size(1)
        batch_size = len(data)

        # feature size + 1 because the last dimension is created for padding
        tensor = data[0].float().new(batch_size, max_length, feature_size + 1).fill_(0)

        for i in range(len(samples)):
            sample = samples[i]

            data_length = sample.size(0)
            offset = max_length - data_length if align_right else 0

            tensor[i].narrow(0, offset, data_length).narrow(1, 1, sample.size(1)).copy_(sample)
            # in padding dimension: 1 is not padded, 0 is padded
            tensor[i].narrow(0, offset, data_length).narrow(1, 0, 1).fill_(1)

        return tensor, None, lengths
    elif type == 'wav':
        samples = data
        lengths = [x.size(0) for x in samples]
        max_length = max(lengths)
        # allocate data for the batch speech
        feature_size = 1  # samples[0].size(1)  # most likely 1
        assert feature_size == 1, "expecting feature size = 1 but get %2.f" % feature_size
        batch_size = len(data)

        # feature size + 1 because the last dimension is created for padding
        tensor = data[0].float().new(batch_size, max_length, feature_size + 1).fill_(0)

        for i in range(len(samples)):
            sample = samples[i]

            # normalize
            data_length = sample.size(0)
            offset = max_length - data_length if align_right else 0

            channels = 1
            tensor[i].narrow(0, offset, data_length).narrow(1, 1, channels).copy_(sample)
            # in padding dimension: 1 is not padded, 0 is padded
            tensor[i].narrow(0, offset, data_length).narrow(1, 0, 1).fill_(1)

        return tensor, None, lengths

    else:
        raise NotImplementedError


def collect_fn(src_data, tgt_data,
               src_lang_data, tgt_lang_data,
               src_align_right, tgt_align_right,
               src_type='text',
               augmenter=None, upsampling=False,
               bilingual=False, vocab_mask=None,
               past_src_data=None, src_pad="<blank>", tgt_pad="<blank>", feature_size=40):
    tensors = dict()
    if src_data is not None:
        tensors['source'], tensors['source_pos'], src_lengths = merge_data(src_data, align_right=src_align_right,
                                                                           type=src_type, augmenter=augmenter,
                                                                           upsampling=upsampling, feature_size=feature_size,
                                                                           dataname="source", src_pad=src_pad)
        tensors['src_type'] = src_type
        # batch-first, padded positions are 1, non-padded are 0
        tensors['src_selfattn_mask'] = tensors['source'].eq(src_pad).transpose(0, 1).contiguous()
        tensors['source'] = tensors['source'].transpose(0, 1).contiguous()
        if tensors['source_pos'] is not None:
            tensors['source_pos'] = tensors['source_pos'].transpose(0, 1)
        tensors['src_lengths'] = torch.LongTensor(src_lengths)
        tensors['src_size'] = sum(src_lengths)

    if tgt_data is not None:
        target_full, target_pos, tgt_lengths = merge_data(tgt_data, align_right=tgt_align_right,
                                                          dataname="target", tgt_pad=tgt_pad)
        target_full = target_full.t().contiguous()  # transpose BxT to TxB
        tensors['target'] = target_full
        tensors['target_input'] = target_full[:-1]
        tensors['target_output'] = target_full[1:]
        if target_pos is not None:
            tensors['target_pos'] = target_pos.t().contiguous()[:-1]
        tgt_size = sum([len(x) - 1 for x in tgt_data])
        tensors['tgt_lengths'] = tgt_lengths
        # batch-first, padded positions are 1, non-padded are 0
        tensors['tgt_selfattn_mask'] = tensors['target_input'].eq(src_pad).transpose(0, 1).contiguous()
    else:
        tgt_size = 0
        tensors['tgt_lengths'] = None

    # merge data for the previous source
    if past_src_data is not None:
        tensors['past_source'], tensors['past_source_pos'], past_src_lengths = merge_data(past_src_data,
                                                                                          align_right=src_align_right,
                                                                                          type=src_type,
                                                                                          augmenter=augmenter,
                                                                                          upsampling=upsampling,
                                                                                          feature_size=feature_size,
                                                                                          dataname="source",
                                                                                          src_pad=src_pad)

        tensors['past_source'] = tensors['past_source'].transpose(0, 1).contiguous()
        if tensors['past_source_pos'] is not None:
            tensors['past_source_pos'] = tensors['past_source_pos'].transpose(0, 1)
        tensors['past_src_lengths'] = torch.LongTensor(past_src_lengths)
        tensors['past_src_size'] = sum(past_src_lengths)

    tensors['tgt_size'] = tgt_size
    tensors['size'] = len(src_data) if src_data is not None else len(tgt_data)

    if src_lang_data is not None:
        tensors['source_lang'] = torch.cat(src_lang_data).long()
    if tgt_lang_data is not None:
        tensors['target_lang'] = torch.cat(tgt_lang_data).long()

    tensors['vocab_mask'] = vocab_mask

    return LightBatch(tensors)


def rewrap(light_batch):
    """
    Currently this light batch is used in data collection to avoid pickling error
    After that it is converted to Batch
    :param light_batch:
    :return:
    """
    return Batch(light_batch.tensors)


class Batch(object):
    # An object to manage the data within a minibatch
    def __init__(self, tensors):
        self.tensors = defaultdict(lambda: None, tensors)
        self.src_size = tensors['src_size']
        self.tgt_size = tensors['tgt_size']
        self.size = tensors['size']
        self.src_lengths = tensors['src_lengths']
        self.tgt_lengths = tensors['tgt_lengths']
        self.has_target = True if self.tensors['target'] is not None else False
        self.vocab_mask = tensors['vocab_mask']

    def get(self, name):
        if name in self.tensors:
            return self.tensors[name]
        else:
            return None

    def cuda(self, fp16=False, device=None):
        """
        Send the minibatch data into GPU.
        :param device: default = None (default CUDA device)
        :param fp16:
        :return: None
        """
        for key, tensor in self.tensors.items():
            if isinstance(tensor, dict):
                for k in tensor:
                    if isinstance(k, torch.Tensor):
                        v = tensor[k]
                        tensor[k] = v.cuda(device=device)
            elif tensor is not None:
                if isinstance(tensor, torch.Tensor):
                    if tensor.type() == "torch.FloatTensor" and fp16:
                        self.tensors[key] = tensor.half()
                    self.tensors[key] = self.tensors[key].cuda(device=device)
            else:
                continue

    def switchout(self, swrate, src_vocab_size, tgt_vocab_size):
        # Switch out function ... currently works with only source text data
        # if self.src_type == 'text':
        if len(self.tensors['source'].shape) == 2:
            self.tensors['source'] = switchout(self.tensors['source'], src_vocab_size, swrate, transpose=True)

        if self.has_target:
            self.tensors['target'] = switchout(self.tensors['target'], tgt_vocab_size, swrate, transpose=True, offset=1)
            # target_full = self.tensors['target']
            # self.tensors['target_input'] = target_full[:-1]
            # self.tensors['target_output'] = target_full[1:]
            # self.tensors['tgt_mask'] = self.tensors['target_output'].ne(onmt.constants.PAD)

    # Masked Predictive Coding mask
    # Randomly choose positions and set features to Zero
    # For later reconstruction
    def mask_mpc(self, p=0.5):

        # the audio has size [T x B x (F+1)] the FIRST dimension is padding
        # need to sample a mask
        source = self.tensors['source']
        with torch.no_grad():
            source = source.narrow(2, 1, source.size(2) - 1)

            # p drop -> 1 - p keeping probability
            masked_positions = source.new(source.size(0), source.size(1)).bernoulli_(1 - p)
            self.tensors['original_source'] = source.clone()
            source.mul_(
                masked_positions.unsqueeze(-1))  # in-place multiplication that will change the underlying storage

            # remember the positions to be used later in losses
            self.tensors['masked_positions'] = masked_positions

        return


class LightBatch:

    def __init__(self, tensors):
        self.tensors = tensors

    def pin_memory(self):
        """
        Enable memory pinning
        :return:
        """
        for key, tensor in self.tensors.items():
            if isinstance(tensor, dict):
                for k in tensor:
                    v = tensor[k]
                    if isinstance(v, torch.Tensor):
                        tensor[k] = v.pin_memory()
            elif tensor is not None:
                if isinstance(tensor, torch.Tensor):
                    self.tensors[key] = self.tensors[key].pin_memory()
            else:
                continue
        return self


class Dataset(torch.utils.data.Dataset):
    def __init__(self, src_data, tgt_data,
                 src_sizes=None, tgt_sizes=None,
                 src_langs=None, tgt_langs=None,
                 batch_size_words=16384,
                 data_type="text", batch_size_sents=128,
                 multiplier=1, sorting=False,
                 augment=False,
                 src_align_right=False, tgt_align_right=False,
                 verbose=False, cleaning=False, debug=False,
                 num_split=1,
                 sa_f=8, sa_t=64, input_size=40,
                 past_src_data=None,
                 past_src_data_sizes=None,
                 constants=None,
                 **kwargs):
        """
        :param src_data: List of tensors for the source side (1D for text, 2 or 3Ds for other modalities)
        :param tgt_data: List of tensors (1D text) for the target side (already padded with <s> and </s>
        :param src_langs: Source languages (list of one-tensors)
        :param tgt_langs: Target Languages (list of one-tensors)
        :param batch_size_words: Maximum number of words in the minibatch (MB can't have more than this)
        :param data_type: Text or Audio
        :param batch_size_sents: Maximum number of sequences in the minibatch (MB can't have more than this)
        :param multiplier: The number of sequences must divide by this number (for fp16 when multiplier=8)
        :param reshape_speech: Put N frames together to reduce the length (this might be done already in preprocessing)
        :param augment: Speech Augmentation (currently only spec augmentation is implemented)
        """

        """
        For alignment, the right-aligned data looks like:
        P P P P D D D D
        P P D D D D D D
        P P P P P D D D
        P P P D D D D D
        This can affect positional encoding (whose implementation is not consistent w.r.t padding)
        For models with absolute positional encoding, src and tgt should be aligned left (This is default)
        For models with relative positional encoding, src should be right and tgt should be left
        """
        if constants is not None:
            self.tgt_pad = constants.TGT_PAD
            self.src_pad = constants.SRC_PAD
        else:
            self.tgt_pad = onmt.constants.TGT_PAD
            self.src_pad = onmt.constants.SRC_PAD

        self.src = src_data
        self.past_src = past_src_data
        self._type = data_type
        self.src_align_right = src_align_right
        if self.src_align_right and verbose:
            print("* Source sentences aligned to the right side.")
        self.tgt_align_right = tgt_align_right
        self.upsampling = kwargs.get('upsampling', False)

        self.max_src_len = kwargs.get('max_src_len', None)
        self.max_tgt_len = kwargs.get('max_tgt_len', 256)
        self.cleaning = int(cleaning)
        self.debug = debug
        self.num_split = num_split
        self.vocab_mask = None
        self.use_past_src = self.past_src is not None
        self.min_tgt_len = kwargs.get('min_tgt_len', 4)
        self.min_src_len = kwargs.get('min_src_len', 4)

        # self.max_tgt_len = 128  # try to hard code this ... so tired of datasets with crappy samples

        if self.max_src_len is None:
            if self._type == 'text':
                self.max_src_len = 256
            elif self._type == 'wav':
                self.max_src_len = 160000
            else:
                # for audio set this to 2048 frames
                self.max_src_len = 2048 if not self.use_past_src else 4095

        # self.reshape_speech = reshape_speech
        if tgt_data:
            self.tgt = tgt_data

        else:
            self.tgt = None

        self.order = np.arange(len(self.src))

        # Processing data sizes
        if self.src is not None:
            if src_sizes is not None:
                self.src_sizes = np.asarray(src_sizes)
            else:
                self.src_sizes = np.asarray([data.size(0) for data in self.src])
        else:
            self.src_sizes = None

        # add the past source size to source size (to balance out the encoder part during allocation)
        if self.use_past_src:
            if past_src_data_sizes is not None:
                self.src_sizes += np.asarray(past_src_data_sizes)
            else:
                self.src_sizes += np.asarray([data.size(0) for data in self.past_src])

        if self.tgt is not None:
            if tgt_sizes is not None:
                self.tgt_sizes = np.asarray(tgt_sizes)
            else:
                self.tgt_sizes = np.asarray([data.size(0) for data in self.tgt])
        else:
            self.tgt_sizes = None

        # sort data to have efficient mini-batching during training
        if sorting:

            if self._type == 'text':
                sorted_order = np.lexsort((self.src_sizes, self.tgt_sizes))
            elif self._type in ['audio', 'wav']:
                sorted_order = np.lexsort((self.tgt_sizes, self.src_sizes))

            self.order = sorted_order

        # store data length in numpy for fast query
        if self.tgt is not None and self.src is not None:
            stacked_sizes = np.stack((self.src_sizes, self.tgt_sizes - 1), axis=0)
            self.data_lengths = np.amax(stacked_sizes, axis=0)
        elif self.src is None:
            self.data_lengths = self.tgt_sizes
        else:
            self.data_lengths = self.src_sizes

        # Processing language ids
        self.src_langs = src_langs
        self.tgt_langs = tgt_langs

        if self.src_langs is not None and self.tgt_langs is not None:
            assert (len(src_langs) == len(tgt_langs))

        # In "bilingual" case, the src_langs only contains one single vector
        # Which is broadcasted to batch_size
        if len(src_langs) <= 1:
            self.bilingual = True
        else:
            self.bilingual = False

        self.full_size = len(self.src) if self.src is not None else len(self.tgt)

        # maximum number of tokens in a mb
        self.batch_size_words = batch_size_words

        # maximum sequences in a mb
        self.batch_size_sents = batch_size_sents

        # the actual batch size must divide by this multiplier (for fp16 it has to be 4 or 8)
        self.multiplier = multiplier

        # by default: count the amount of padding when we group mini-batches
        self.pad_count = True

        # group samples into mini-batches
        # if verbose:
        #     print("* Allocating mini-batches ...")
        self.batches = allocate_batch(self.order, self.data_lengths,
                                      self.src_sizes, self.tgt_sizes,
                                      batch_size_words, batch_size_sents, self.multiplier,
                                      self.max_src_len, self.max_tgt_len,
                                      self.min_src_len, self.min_tgt_len, self.cleaning)

        # the second to last mini-batch is likely the largest
        # (the last one can be the remnant after grouping samples which has less than max size)
        self.largest_batch_id = len(self.batches) - 3

        self.num_batches = len(self.batches)
        lengths = [len(x) for x in self.batches]
        print("Number of sentences after cleaning and sorting: %d" % sum(lengths) )
        print("Number of batches after cleaning and sorting: %d" % self.num_batches)

        self.cur_index = 0
        self.batchOrder = None
        self.input_size = input_size

        if augment:
            self.augmenter = Augmenter(F=sa_f, T=sa_t, input_size=input_size)
        else:
            self.augmenter = None

    def size(self):
        return self.full_size

    def switchout(self, batch):

        pass

    def set_epoch(self, epoch):

        pass

    def set_mask(self, vocab_mask):
        self.vocab_mask = vocab_mask

    def get_largest_batch(self):

        return self.get_batch(self.largest_batch_id)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):

        src_lang, tgt_lang = None, None

        if self.bilingual:
            if self.src_langs is not None:
                src_lang = self.src_langs[0]  # should be a tensor [0]
            if self.tgt_langs is not None:
                tgt_lang = self.tgt_langs[0]  # should be a tensor [1]
        else:
            if self.src_langs is not None:
                src_lang = self.src_langs[index]
            if self.tgt_langs is not None:
                tgt_lang = self.tgt_langs[index]

        # move augmenter here?

        if self.use_past_src:
            past_src = self.past_src[index]
        else:
            past_src = None

        sample = {
            'src': self.src[index] if self.src is not None else None,
            'tgt': self.tgt[index] if self.tgt is not None else None,
            'src_lang': src_lang,
            'tgt_lang': tgt_lang,
            'past_src': past_src
        }

        return sample

    def get_batch(self, index):
        """
        This function is only used in when we need to access a batch directly from the dataset
        (Without an external loader)
        :param index: the index of the mini-batch in the list
        :return: Batch
        """
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

        src_lang_data = None
        tgt_lang_data = None

        if self.bilingual:
            if self.src_langs is not None:
                src_lang_data = [self.src_langs[0]]  # should be a tensor [0]
            if self.tgt_langs is not None:
                tgt_lang_data = [self.tgt_langs[0]]  # should be a tensor [1]
        else:
            if self.src_langs is not None:
                src_lang_data = [self.src_langs[i] for i in batch_ids]
            if self.tgt_langs is not None:
                tgt_lang_data = [self.tgt_langs[i] for i in batch_ids]

        if self.use_past_src:
            past_src = [self.past_src[i] for i in batch_ids]
        else:
            past_src = None

        batch = rewrap(collect_fn(src_data, tgt_data=tgt_data,
                                  src_lang_data=src_lang_data, tgt_lang_data=tgt_lang_data,
                                  src_align_right=self.src_align_right, tgt_align_right=self.tgt_align_right,
                                  src_type=self._type,
                                  augmenter=self.augmenter, upsampling=self.upsampling, vocab_mask=self.vocab_mask,
                                  past_src_data=past_src,
                                  src_pad=self.src_pad,
                                  tgt_pad=self.tgt_pad,
                                  feature_size=self.input_size),
                       )
        return batch

    def collater(self, collected_samples):
        """
        Merge a list of samples into a Batch
        :param collected_samples: list of dicts (the output of the __getitem__)
        :return: batch
        """

        split_size = math.ceil(len(collected_samples) / self.num_split)
        sample_list = [collected_samples[i:i + split_size]
                       for i in range(0, len(collected_samples), split_size)]

        batches = list()

        for samples in sample_list:

            src_data, tgt_data = None, None
            src_lang_data, tgt_lang_data = None, None
            past_src_data = None

            if self.src:
                src_data = [sample['src'] for sample in samples]

            if self.tgt:
                tgt_data = [sample['tgt'] for sample in samples]

            if self.bilingual:
                if self.src_langs is not None:
                    src_lang_data = [self.src_langs[0]]  # should be a tensor [0]
                if self.tgt_langs is not None:
                    tgt_lang_data = [self.tgt_langs[0]]  # should be a tensor [1]
            else:
                if self.src_langs is not None:
                    src_lang_data = [sample['src_lang'] for sample in samples]  # should be a tensor [0]
                if self.tgt_langs is not None:
                    tgt_lang_data = [sample['tgt_lang'] for sample in samples]  # should be a tensor [1]

            if self.use_past_src:
                past_src_data = [sample['past_src'] for sample in samples]

            batch = collect_fn(src_data, tgt_data=tgt_data,
                               src_lang_data=src_lang_data, tgt_lang_data=tgt_lang_data,
                               src_align_right=self.src_align_right, tgt_align_right=self.tgt_align_right,
                               src_type=self._type,
                               augmenter=self.augmenter, upsampling=self.upsampling, vocab_mask=self.vocab_mask,
                               past_src_data=past_src_data, src_pad=self.src_pad, tgt_pad=self.tgt_pad,
                               feature_size=self.input_size)

            batches.append(batch)

        return batches

    def full_size(self):
        return self.full_size

    # genereate a new batch - order (static)
    def create_order(self, random=True):

        if random:
            self.batchOrder = torch.randperm(self.num_batches)
        else:
            self.batchOrder = torch.arange(self.num_batches).long()

        self.cur_index = 0

        return self.batchOrder

    # return the next batch according to the iterator
    def next(self, curriculum=False, reset=True):

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

        assert (0 <= iteration < self.num_batches)
        self.cur_index = iteration

#
# # LANGUAGE MODEL DATASET AND DATAHOLDER
# class LMBatch(Batch):
#
#     def __init__(self, input, target=None):
#         self.tensors = defaultdict(lambda: None)
#
#         self.tensors['target_input'] = input  # T x B
#         self.tensors['target_output'] = target  # T x B or None
#
#         # batch size
#         self.size = input.size(1)
#         self.length = input.size(0)
#
#         self.tgt_size = self.size * self.length
#         self.src_size = 0
#
#     def collate(self, **kwargs):
#         raise NotImplementedError
#
#
# class LanguageModelDataset(Dataset):
#
#     def __init__(self, data, batch_size_sents=128, seq_length=128):
#
#         self.data = data
#
#         self.batch_size_sents = batch_size_sents
#
#         self.seq_length = seq_length
#
#         # group samples into mini batches
#         self.num_batches = 0
#         self.allocate_batch()
#
#         self.full_size = self.num_batches
#         # self.cur_index = 0
#         # self.batchOrder = None
#
#     def allocate_batch(self):
#
#         nsequence = self.data.size(0) // self.batch_size_sents
#
#         self.data = self.data.narrow(0, 0, nsequence * self.batch_size_sents)
#
#         # Evenly divide the data across the bsz batches.
#         self.data = self.data.view(self.batch_size_sents, -1).t().contiguous()
#
#         # self.num_steps = nbatch - 1
#
#         self.num_batches = math.ceil((self.data.size(0) - 1) / self.seq_length)
#
#     # genereate a new batch - order (static)
#     def create_order(self, random=False):
#
#         # For language model order shouldn't be random
#         if random:
#             self.batchOrder = torch.randperm(self.num_batches)
#         else:
#             self.batchOrder = torch.arange(self.num_batches).long()
#
#         self.cur_index = 0
#
#         return self.batchOrder
#
#     # return the next batch according to the iterator
#     # for language model
#     def next(self, curriculum=True, reset=True, split_sizes=1):
#
#         # reset iterator if reach data size limit
#         if self.cur_index >= self.num_batches:
#             if reset:
#                 self.cur_index = 0
#             else:
#                 return None
#
#         batch_index = self.cur_index
#
#         seq_len = self.seq_length
#
#         top_index = min(batch_index + seq_len, self.data.size(0) - 1)
#
#         batch = LMBatch(self.data[batch_index:top_index], target=self.data[batch_index + 1:top_index + 1])
#
#         # move the iterator one step
#         self.cur_index += seq_len
#
#         return [batch]
