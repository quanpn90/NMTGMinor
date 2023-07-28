from __future__ import division

import math
import torch
import torch.utils.data
from collections import defaultdict
import onmt
from onmt.speech.Augmenter import Augmenter
from onmt.modules.dropout import switchout
import numpy as np
from .batch_utils import allocate_batch, allocate_batch_unbalanced, allocate_batch_simple
import dill
import random

from onmt.data.indexed_file import read_data

"""
Data management for sequence-to-sequence models
Two basic classes: 
- Batch stores the input / output sequences, grouped into tensors with the same length (by padding)
- Dataset stores all of the data and 
"""

def merge_concat_data(data, type="text", src_pad=0, tgt_pad=0,
                      max_len=640000, feature_size=40, bilingual=False):

    """
    Args:
        data: list of list of Samples
        type:
        src_pad:
        tgt_pad:
        dataname:

    Returns:

    """

    _sample = data[0][0]  # take the first sample tensor
    has_src = _sample["src"] is not None
    has_tgt = _sample["tgt"] is not None
    assert (has_src and has_tgt)

    _src_data = _sample["src"]
    _tgt_data = _sample["tgt"]
    _src_lang_data = _sample["src_lang"]
    _tgt_lang_data = _sample["tgt_lang"]
    batch_size = len(data)


    #
    src_lengths = [sum(__data["src"].size(0) for __data in _data) for _data in data]
    max_src_len = max(src_lengths)

    tgt_lengths = [sum(__data["tgt"].size(0) for __data in _data) for _data in data]
    max_tgt_len = max(tgt_lengths)


    # allocate tensor
    src_tensor = _src_data.float().new(batch_size, max_src_len, feature_size + 1).fill_(0)

    # tensor = data[0].float().new(batch_size, max_length, feature_size + 1).fill_(0)
    #
    # for i in range(len(samples)):
    #     sample = samples[i]
    #
    #     # normalize
    #     data_length = sample.size(0)
    #     offset = max_length - data_length if align_right else 0
    #
    #     channels = 1
    #     tensor[i].narrow(0, offset, data_length).narrow(1, 1, channels).copy_(sample)
    #     # in padding dimension: 1 is not padded, 0 is padded
    #     tensor[i].narrow(0, offset, data_length).narrow(1, 0, 1).fill_(1)

    tgt_tensor = _src_data.new(batch_size, max_tgt_len).long().fill_(tgt_pad)

    src_lang_tensor = _src_lang_data.new(batch_size, max_src_len).long().fill_(0)
    tgt_lang_tensor = _tgt_lang_data.new(batch_size, max_tgt_len).long().fill_(0)

    for i, _data in enumerate(data):
        # each element in the minibatch is a list of samples

        src_offset = 0  # align left so we start from 0
        tgt_offset = 0  # align left
        assert type == "wav"

        for _sample in _data:

            # fill in the source
            src_sample = _sample["src"]
            # normalize
            data_length = src_sample.size(0)
            channels = 1
            src_tensor[i].narrow(0, src_offset, data_length).narrow(1, 1, channels).copy_(src_sample)

            src_lang_sample = _sample["src_lang"]
            if src_lang_sample.numel() == 1:
                src_lang_sample = src_lang_sample.repeat(data_length)
            src_lang_tensor[i].narrow(0, src_offset, data_length).copy_(src_lang_sample)

            # update offset for source
            src_offset = src_offset + data_length

            # fill in the target
            tgt_sample = _sample["tgt"]
            data_length = tgt_sample.size(0)
            tgt_tensor[i].narrow(0, tgt_offset, data_length).copy_(tgt_sample)

            tgt_lang_sample = _sample["tgt_lang"]
            if tgt_lang_sample.numel() == 1:
                tgt_lang_sample = tgt_lang_sample.repeat(data_length)
            tgt_lang_tensor[i].narrow(0, tgt_offset, data_length).copy_(tgt_lang_sample)

            # update offset for target
            tgt_offset = tgt_offset + data_length

    return src_tensor, tgt_tensor, src_lang_tensor, tgt_lang_tensor, src_lengths, tgt_lengths


def merge_data(data, align_right=False, type='text', augmenter=None, upsampling=False,
               feature_size=40, dataname="source", src_pad=1, tgt_pad=1 ):
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

def collate_fn(src_data, tgt_data,
               src_lang_data, tgt_lang_data,
               src_atbs_data, tgt_atbs_data,
               src_align_right, tgt_align_right,
               src_type='text',
               augmenter=None, upsampling=False,
               bilingual=False, vocab_mask=None,
               past_src_data=None, src_pad="<blank>", tgt_pad="<blank>", feature_size=40, use_memory=None,
               src_features=None, deterministic=False):
    tensors = dict()
    if src_data is not None:
        tensors['source'], tensors['source_pos'], src_lengths = merge_data(src_data, align_right=src_align_right,
                                                                           type=src_type, augmenter=augmenter,
                                                                           upsampling=upsampling, feature_size=feature_size,
                                                                           dataname="source", src_pad=src_pad)
        tensors['src_type'] = src_type
        tensors['src_selfattn_mask'] = tensors['source'].eq(src_pad)
        tensors['source'] = tensors['source'].transpose(0, 1).contiguous()
        if tensors['source_pos'] is not None:
            tensors['source_pos'] = tensors['source_pos'].transpose(0, 1)
        tensors['src_lengths'] = torch.LongTensor(src_lengths)
        tensors['src_size'] = sum(src_lengths)

    if tgt_data is not None:
        target_full, target_pos, tgt_lengths = merge_data(tgt_data, align_right=tgt_align_right,
                                                          dataname="target", tgt_pad=tgt_pad)
        tensors['tgt_selfattn_mask'] = target_full.eq(tgt_pad)
        target_full = target_full.t().contiguous()  # transpose BxT to TxB
        tensors['target'] = target_full
        tensors['target_input'] = target_full[:-1]
        tensors['target_input_selfattn_mask'] = tensors['target_input'].transpose(0, 1).eq(tgt_pad)
        tensors['target_output'] = target_full[1:]
        if target_pos is not None:
            tensors['target_pos'] = target_pos.t().contiguous()[:-1]
        tgt_size = sum([len(x) - 1 for x in tgt_data])
        tensors['tgt_lengths'] = tgt_lengths

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

    if src_atbs_data is not None:
        tensors['source_atbs'] = torch.cat(src_atbs_data).long()
    if tgt_atbs_data is not None:
        tensors['target_atbs'] = torch.cat(tgt_atbs_data).long()

    tensors['vocab_mask'] = vocab_mask

    if src_features is not None and len(src_features) > 0:
        src_size = 0
        features = torch.zeros(max(f.shape[0] for f in src_features), len(src_features), src_features[0].shape[1])
        mask = torch.ones(len(src_features), max(f.shape[0] for f in src_features), dtype=torch.uint8)
        for i, f in enumerate(src_features):
            features[:f.shape[0], i] = f
            mask[i, :f.shape[0]] = 0
            src_size += f.shape[0]
        tensors["src_features"] = features
        tensors["src_features_mask"] = mask
        tensors["src_size"] = src_size

    if use_memory:
        if deterministic:
            random.seed(42)

        tokenizer = use_memory
        n_max = 3

        remove_chars = [".",",","!","?",";",":"]
        def remove_last_punctuation(s):
            if len(s) > 0 and s[-1] in remove_chars:
                return s[:-1]
            else:
                return s

        sentences = [tokenizer.decode(tokens[1:-1]).split() for tokens in tgt_data]

        ngrams_to_indices = {} # maps all ngrams in the batch to the indices of the sample in the batch containing it
        for n in range(1,n_max+1):
            for i,s in enumerate(sentences):
                for j in range(len(s)-n+1):
                    ngram = remove_last_punctuation(" ".join(s[j:j+n]))
                    if any(r in ngram for r in remove_chars): # have no e.g. comma in memory
                        continue
                    ngram = tuple(ngram.split())
                    if not ngram in ngrams_to_indices:
                        ngrams_to_indices[ngram] = set([i])
                    else:
                        ngrams_to_indices[ngram].add(i)

        num_ngrams = max(1, len(ngrams_to_indices) // 5 // n_max) # take every 5th ngram

        chosen_ngrams = [] # choose ngrams for memory
        while len(chosen_ngrams) < num_ngrams and len(ngrams_to_indices)  > 0:
            index = random.randint(0,len(ngrams_to_indices)-1)
            for j,ngram in enumerate(ngrams_to_indices):
                if j==index:
                    i = ngrams_to_indices.pop(ngram)
                    break

            chosen_ngrams.append((" ".join(ngram),i))

            key_del = []
            for ngram_ in ngrams_to_indices:
                if any(word in ngram for word in ngram_):
                    key_del.append(ngram_)

            for k in key_del:
                del ngrams_to_indices[k]

        chosen_ngrams_tokens = [(torch.as_tensor(tokenizer.encode(w[0])), w[1]) for w in chosen_ngrams]

        memory_text_ids = merge_data([x[0] for x in chosen_ngrams_tokens], align_right=tgt_align_right,
                                     dataname="target", tgt_pad=tgt_pad)[0]
        tensors["memory_text_ids"] = memory_text_ids

        target_output = tensors["target_output"]
        label_mem = -target_output.transpose(1,0).lt(2).to(tensors["target"].dtype)
        for i,(tokens, indices) in enumerate(chosen_ngrams_tokens):
            tokens = tokens[1:-1]
            for index in indices:
                for j in range(target_output.shape[0] - len(tokens)):
                    if target_output[j:j+len(tokens),index].eq(tokens).all():
                        label_mem[index,j:j+len(tokens)] = i+1
        tensors["label_mem"] = label_mem.transpose(1,0)

    return LightBatch(tensors)


# default = align left
# applicable for text?
def collate_concat_fn(samples, src_type='text', bilingual=False,
                      src_pad=0, tgt_pad=0,
                      feature_size=40):

    """
    Args:
        samples: a list of list of samples (dictionary containing tensors for src, tgt, src_lang, tgt_lang)
        src_type:
        bilingual: True/False. If Bilingual: the lang tensor will be
        src_pad:
        tgt_pad:

    Returns:

    """

    tensors = dict()
    tensors['source'], target_full, tensors['source_lang'], target_lang_full, src_lengths, tgt_lengths = \
        merge_concat_data(samples, type=src_type,
        src_pad=src_pad, tgt_pad=tgt_pad,
        bilingual=bilingual, feature_size=feature_size)

    tensors['src_type'] = src_type
    tensors['src_selfattn_mask'] = tensors['source'].eq(src_pad)
    tensors['source'] = tensors['source'].transpose(0, 1).contiguous()
    tensors['source_lang'] = tensors['source'].transpose(0, 1).contiguous()

    tensors['src_lengths'] = torch.LongTensor(src_lengths)
    tensors['src_size'] = sum(src_lengths)

    tensors['tgt_selfattn_mask'] = target_full.eq(tgt_pad)
    target_full = target_full.t().contiguous()  # transpose BxT to TxB
    tensors['target'] = target_full
    tensors['target_input'] = target_full[:-1]
    tensors['target_input_selfattn_mask'] = tensors['target_input'].transpose(0, 1).eq(tgt_pad)
    tensors['target_output'] = target_full[1:]
    tensors['tgt_size'] = sum(tgt_lengths)
    tensors['tgt_lengths'] = torch.LongTensor(tgt_lengths)
    tensors['target_lang'] = target_lang_full.transpose(0, 1).contiguous()[1:]

    tensors['vocab_mask'] = None
    tensors['source_atbs'] = None
    tensors['target_atbs'] = None
    tensors['size'] = len(src_lengths)

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
        self.src_size = tensors['src_size'] if "src_size" in tensors else 0
        self.tgt_size = tensors['tgt_size']
        self.size = tensors['size']
        self.src_lengths = tensors['src_lengths'] if "src_lengths" in tensors else None
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


class Sample(object):

    def __init__(self, src_data, tgt_data,
                 src_lang_data, tgt_lang_data,
                 past_src_data=None):

        self.src = src_data
        self.tgt = tgt_data
        self.src_lang = src_lang_data
        self.tgt_lang = tgt_lang_data
        self.past_src_data = past_src_data

        #
        # batch = collate_fn(src_data, tgt_data=tgt_data,
        #                    src_lang_data=src_lang_data, tgt_lang_data=tgt_lang_data,
        #                    src_atbs_data=src_atbs_data, tgt_atbs_data=tgt_atbs_data,
        #                    src_align_right=self.src_align_right, tgt_align_right=self.tgt_align_right,
        #                    src_type=self._type,
        #                    augmenter=self.augmenter, upsampling=self.upsampling, vocab_mask=self.vocab_mask,
        #                    past_src_data=past_src_data, src_pad=self.src_pad, tgt_pad=self.tgt_pad,
        #                    feature_size=self.input_size)


class Dataset(torch.utils.data.Dataset):

    def get_tgt_pad(self):
        return self.tgt_pad

    def get_batches(self):
        return self.batches

    def get_collater(self):

        if self.concat:
            return self.collater_concat

        return self.collater

    def get_size(self):
        return self.num_batches

    def __init__(self, src_data, tgt_data,
                 src_sizes=None, tgt_sizes=None,
                 src_langs=None, tgt_langs=None,
                 src_atbs=None, tgt_atbs=None,
                 batch_size_frames=1280000,
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
                 concat=False,
                 constants=None,
                 dataset_factor=None,
                 use_memory=False,
                 validation=True,
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
            constants = dill.loads(constants)
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
        self.max_tgt_len = kwargs.get('max_tgt_len', 256 )
        self.cleaning = int(cleaning)
        self.debug = debug
        self.num_split = num_split
        self.vocab_mask = None
        self.use_past_src = self.past_src is not None
        self.min_tgt_len = kwargs.get('min_tgt_len', 3)
        self.min_src_len = kwargs.get('min_src_len', 2)
        self.batch_size_frames = batch_size_frames

        cut_off_size = kwargs.get('cut_off_size', 256000)
        smallest_batch_size = kwargs.get('smallest_batch_size', 4)

        self.concat = concat
        if self.concat:
            # if we use concat dataset, then how do we do minibatch?
            pass

        if self.max_src_len is None:
            if self._type == 'text':
                self.max_src_len = 256
            elif self._type == 'wav':
                self.max_src_len = 320000
            else:
                # for audio set this to 2048 frames
                self.max_src_len = 4096 if not self.use_past_src else 8192

        # self.reshape_speech = reshape_speech
        if tgt_data:
            self.tgt = tgt_data

        else:
            self.tgt = None

        # Processing data sizes
        if self.src is not None:
            if src_sizes is not None:
                if verbose:
                    print("Loading source size from binarized data ...")
                src_sizes = np.asarray(src_sizes)
            else:
                if verbose:
                    print("Source size not available. Computing source size from data...")
                src_sizes = np.asarray([data.size(0) for data in self.src])
        else:
            src_sizes = None

        # add the past source size to source size (to balance out the encoder part during allocation)
        if self.use_past_src:
            if past_src_data_sizes is not None:
                src_sizes += np.asarray(past_src_data_sizes)
            else:
                src_sizes += np.asarray([data.size(0) for data in self.past_src])

        if self.tgt is not None:
            if tgt_sizes is not None:
                print("Loading target size from binarized data ...")
                tgt_sizes = np.asarray(tgt_sizes)
            else:
                print("Target size not available. Computing target size from data...")
                tgt_sizes = np.asarray([data.size(0) for data in self.tgt])
        else:
            tgt_sizes = None

        # sort data to have efficient mini-batching during training
        if sorting and not self.concat:

            if self._type == 'text':
                sorted_order = np.lexsort((src_sizes, tgt_sizes))
            elif self._type in ['audio', 'wav']:
                sorted_order = np.lexsort((tgt_sizes, src_sizes))

        else:
            sorted_order = np.random.permutation(np.arange(len(self.src)))

        self.order = None

        # if concat then don't have to sort :)
        # store data length in numpy for fast query
        if self.tgt is not None and self.src is not None:
            stacked_sizes = np.stack((src_sizes, tgt_sizes - 1), axis=0)
            data_lengths = np.amax(stacked_sizes, axis=0)
        elif self.src is None:
            data_lengths = tgt_sizes
        else:
            data_lengths = src_sizes

        # Processing language ids
        self.src_langs = src_langs
        self.tgt_langs = tgt_langs

        if self.src_langs is not None and self.tgt_langs is not None:
            assert (len(src_langs) == len(tgt_langs))

        # Processing attributes
        self.src_atbs = src_atbs
        self.tgt_atbs = tgt_atbs

        # In "bilingual" case, the src_langs only contains one single vector
        # Which is broadcasted to batch_size
        if len(src_langs) <= 1:
            self.bilingual = True
            if self.src_atbs is not None:
                assert(len(src_atbs) <= 1), "For a bilingual dataset, expect attributes to be 'singular' too"
        else:
            self.bilingual = False

        self.full_size = len(src_sizes)
        # self.full_size = len(self.src) if self.src is not None else len(self.tgt)

        # maximum number of tokens in a mb
        self.batch_size_words = batch_size_words

        # maximum sequences in a mb
        self.batch_size_sents = batch_size_sents

        # the actual batch size must divide by this multiplier (for fp16 it has to be 4 or 8)
        self.multiplier = multiplier

        # by default: count the amount of padding when we group mini-batches
        self.pad_count = True

        # group samples into mini-batches
        if self.concat:
            _batch_size = math.floor(batch_size_frames / self.max_src_len)

            self.batches = allocate_batch_simple(sorted_order,
                                                 src_sizes, tgt_sizes,
                                                 _batch_size,
                                                 self.max_src_len, self.max_tgt_len,
                                                 self.min_src_len, self.min_tgt_len)

            # allocate_batch_simple(indices,
            #                       src_sizes, tgt_sizes,
            #                       batch_size_sents,
            #                       max_src_len, max_tgt_len,
            #                       min_src_len, min_tgt_len):

            self.src_sizes = src_sizes
            self.tgt_sizes = tgt_sizes

        else:
            self.src_sizes = None
            self.tgt_sizes = None

            if self._type in ['audio', 'wav']:

                self.batches = allocate_batch_unbalanced(sorted_order, data_lengths,
                                                         src_sizes, tgt_sizes,
                                                         batch_size_frames, batch_size_words,
                                                         batch_size_sents, self.multiplier,
                                                         self.max_src_len, self.max_tgt_len,
                                                         self.min_src_len, self.min_tgt_len, self.cleaning,
                                                         cut_off_size, smallest_batch_size)
            else:
                self.batches = allocate_batch(sorted_order, data_lengths,
                                              src_sizes, tgt_sizes,
                                              batch_size_words, batch_size_sents, self.multiplier,
                                              self.max_src_len, self.max_tgt_len,
                                              self.min_src_len, self.min_tgt_len, self.cleaning)

        # if using a concat dataset, we want to find a batch contains fixed N segments with length M < max_src_len
        # and then in the collate function we randomly sample from the same dataset to fill max_src_len

        # the second to last mini-batch is likely the largest
        # (the last one can be the remnant after grouping samples which has less than max size)
        self.largest_batch_id = len(self.batches) - 3

        if dataset_factor is not None:
            self.batches = [b for b in self.batches for _ in range(dataset_factor)]

        self.num_batches = len(self.batches)
        self.batch_sizes = [len(x) for x in self.batches]
        self.filtered_samples = []
        # map(self.filtered_samples.extend, self.batches)
        for x in self.batches:
            for _sample in x:
                self.filtered_samples.append(_sample)

        print("Number of sentences before cleaning and sorting: %d" % len(src_sizes) )
        print("Number of sentences after cleaning and sorting: %d" % sum(self.batch_sizes) )
        print("Number of sentences after cleaning and sorting: %d" % len(self.filtered_samples) )
        print("Number of batches after cleaning and sorting: %d" % self.num_batches)

        self.cur_index = 0
        self.batchOrder = None
        self.input_size = input_size
        self.src_sizes = src_sizes

        if augment:
            self.augmenter = Augmenter(F=sa_f, T=sa_t, input_size=input_size)
        else:
            self.augmenter = None

        if use_memory:
            from transformers import MBart50TokenizerFast
            self.use_memory = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50") # to split transcripts by word boundaries
        else:
            self.use_memory = False
        self.validation = validation

    def flush_cache(self):
        if hasattr(self.src, 'flush_cache'):
            self.src.flush_cache()

    def size(self):
        return self.full_size

    def switchout(self, batch):

        pass

    def set_epoch(self, epoch):

        pass

    def set_mask(self, vocab_mask):
        self.vocab_mask = vocab_mask

    def get_largest_batch(self, bsz=-1, src_size=-1, tgt_size=-1):
        if bsz == -1 and src_size == -1 and tgt_size == -1:
            return self.get_batch(self.largest_batch_id)
        else:
            raise NotImplementedError
            # batch = None
            # for i in range(self.num_batches):
            #
            #     src_size_ = self.batch_src_sizes[i]
            #     tgt_size_ = self.batch_tgt_sizes[i]
            #     bsz_size_ = self.batch_sizes[i]
            #
            #     get_batch = True
            #     if bsz > 0:
            #         if bsz_size_ != bsz:
            #             get_batch = False
            #
            #     if src_size > 0:
            #         if src_size_ != src_size:
            #             get_batch = False
            #
            #     if tgt_size > 0:
            #         if tgt_size_ != tgt_size:
            #             get_batch = False
            #
            #     if get_batch:
            #         # print("Found batch satisfying the conditions bsz %d src_size %d tgt_size %d" % (bsz, src_size, tgt_size))
            #         return self.get_batch(i)

            # print("Cannot find the batch satisfying those conditions")
            return self.get_batch(self.largest_batch_id)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index, load_src=True):

        src_lang, tgt_lang = None, None
        src_atb, tgt_atb  = None, None

        if self.bilingual:
            if self.src_langs is not None:
                src_lang = self.src_langs[0]  # should be a tensor [0]
            if self.tgt_langs is not None:
                tgt_lang = self.tgt_langs[0]  # should be a tensor [1]
            if self.src_atbs is not None:
                src_atb = self.src_atbs[0]
            if self.tgt_atbs is not None:
                tgt_atb = self.tgt_atbs[0]
        else:
            if self.src_langs is not None:
                src_lang = self.src_langs[index]
            if self.tgt_langs is not None:
                tgt_lang = self.tgt_langs[index]
            # if self.src_atbs is not None:
            #     src_atb = self.src_atbs[index]
            # if self.tgt_atbs is not None:
            #     tgt_atb = self.tgt_atbs[index]
            src_atb = None
            tgt_atb = None

        # move augmenter here?

        if self.use_past_src:
            past_src = self.past_src[index]
        else:
            past_src = None

        if not hasattr(self, "encoder_feature_files"):
            sampledata = {
                'src': self.src[index] if self.src is not None and load_src else None,
                'tgt': self.tgt[index] if self.tgt is not None else None,
                'src_lang': src_lang,
                'tgt_lang': tgt_lang,
                'src_atb': src_atb,  # depricated
                'tgt_atb': tgt_atb,  # depricated
                'past_src': past_src,
            }
        else:
            if load_src:
                with open(self.encoder_feature_files[0], "rb") as f:
                    src_features = read_data(f, self.encoder_feature_files[1], index)
            else:
                src_features = None
            sample = {
                'src_features': src_features,
                'tgt': self.tgt[index] if self.tgt is not None else None,
                'src_lang': src_lang,
                'tgt_lang': tgt_lang,
                'src_atb': src_atb,  # depricated
                'tgt_atb': tgt_atb,  # depricated
                'past_src': past_src,
            }

        return sampledata

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
        src_atbs_data = None
        tgt_atbs_data = None

        if self.bilingual:
            if self.src_langs is not None:
                src_lang_data = [self.src_langs[0]]  # should be a tensor [0]
            if self.tgt_langs is not None:
                tgt_lang_data = [self.tgt_langs[0]]  # should be a tensor [1]
            if self.src_atbs is not None:
                src_atbs_data = [self.src_atbs[0]]
            if self.tgt_atbs is not None:
                tgt_atbs_data = [self.tgt_atbs[0]]
        else:
            if self.src_langs is not None:
                src_lang_data = [self.src_langs[i] for i in batch_ids]
            if self.tgt_langs is not None:
                tgt_lang_data = [self.tgt_langs[i] for i in batch_ids]
            # if self.src_atbs is not None:
            #     src_atbs_data = [self.src_atbs[i] for i in batch_ids]
            # if self.tgt_atbs is not None:
            #     tgt_atbs_data = [self.tgt_atbs[i] for i in batch_ids]
            src_atbs_data = None
            tgt_atbs_data = None

        if self.use_past_src:
            past_src = [self.past_src[i] for i in batch_ids]
        else:
            past_src = None

        batch = rewrap(collate_fn(src_data, tgt_data=tgt_data,
                                  src_lang_data=src_lang_data, tgt_lang_data=tgt_lang_data,
                                  src_atbs_data=src_atbs_data, tgt_atbs_data=tgt_atbs_data,
                                  src_align_right=self.src_align_right, tgt_align_right=self.tgt_align_right,
                                  src_type=self._type,
                                  augmenter=self.augmenter, upsampling=self.upsampling, vocab_mask=self.vocab_mask,
                                  past_src_data=past_src,
                                  src_pad=self.src_pad,
                                  tgt_pad=self.tgt_pad,
                                  feature_size=self.input_size,
                                  use_memory=self.use_memory)
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
            src_atbs_data, tgt_atbs_data = None, None
            past_src_data = None

            if self.src and len(samples)>0 and "src" in samples[0] and samples[0]['src'] is not None:
                src_data = [sample['src'] for sample in samples]

            if self.tgt:
                tgt_data = [sample['tgt'] for sample in samples]

            if self.bilingual:
                if self.src_langs is not None:
                    src_lang_data = [self.src_langs[0]]  # should be a tensor [0]
                if self.tgt_langs is not None:
                    tgt_lang_data = [self.tgt_langs[0]]  # should be a tensor [1]
                # if self.src_atbs is not None:
                #     src_atbs_data = [self.src_atbs[0]]
                # if self.tgt_atbs is not None:
                #     tgt_atbs_data = [self.tgt_atbs[0]]
                src_atbs_data = None
                tgt_atbs_data = None
            else:
                if self.src_langs is not None:
                    src_lang_data = [sample['src_lang'] for sample in samples]  # should be a tensor [0]
                if self.tgt_langs is not None:
                    tgt_lang_data = [sample['tgt_lang'] for sample in samples]  # should be a tensor [1]
                # if self.src_atbs is not None:
                #     src_atbs_data = [self.src_atbs[i] for i in batch_ids]
                # if self.tgt_atbs is not None:
                #     tgt_atbs_data = [self.tgt_atbs[i] for i in batch_ids]
                src_atbs_data = None
                tgt_atbs_data = None

            if self.use_past_src:
                past_src_data = [sample['past_src'] for sample in samples]

            # TODO:
            # src_data is now a [list of [list of Samples]]
            # tgt_data is either None or a [list of [list of Samples]]

            if len(samples) > 0 and "src_features" in samples[0] and samples[0]["src_features"] is not None:
                src_features = [sample["src_features"] for sample in samples]
            else:
                src_features = None

            batch = collate_fn(src_data, tgt_data=tgt_data,
                               src_lang_data=src_lang_data, tgt_lang_data=tgt_lang_data,
                               src_atbs_data=src_atbs_data, tgt_atbs_data=tgt_atbs_data,
                               src_align_right=self.src_align_right, tgt_align_right=self.tgt_align_right,
                               src_type=self._type,
                               augmenter=self.augmenter, upsampling=self.upsampling, vocab_mask=self.vocab_mask,
                               past_src_data=past_src_data, src_pad=self.src_pad, tgt_pad=self.tgt_pad,
                               feature_size=self.input_size, use_memory=self.use_memory, src_features=src_features, deterministic=self.validation)

            batches.append(batch)

        return batches


    def collater_concat(self, collected_samples):
        """
        Merge a list of samples into a Batch
        :param collected_samples: list of dicts (the output of the __getitem__)
        :return: batch
        """

        batch = list()

        # this feature only works with src and tgt in present
        assert self.src is not None
        assert self.tgt is not None

        max_retry = 20
        max_len = self.max_src_len

        sampled_ids = collected_samples

        for sample in  collected_samples:

            if not (type(sample) is dict):
                continue

            cur_sample = [sample]

            cur_src_len = sample['src'].size(0)

            # random samples from the data (fast)
            random_sample_ids = random.choices(self.filtered_samples, k=max_retry)

            lengths = [self.src_sizes[_id] for _id in random_sample_ids]

            random_sample_ids = list(zip(random_sample_ids, lengths))
            random_sample_ids.sort(key = lambda a: a[1])  # sort in ascending order or descending order?

            for random_sample_id, src_len in random_sample_ids:

                if cur_src_len + src_len <= max_len and random_sample_id not in sampled_ids:

                    # if the randomly next sample can be concatenated
                    random_sample = self.__getitem__(random_sample_id)
                    cur_sample.append(random_sample)

                    # then update the current length and update the sampled ids
                    cur_src_len = cur_src_len + src_len
                    sampled_ids.append(random_sample_id)

                if max_len - cur_src_len < min(lengths):
                    break

            batch.append(cur_sample)

        print(len(batch), cur_src_len)
        print([len(data) for data in batch])

        batch = collate_concat_fn(batch, src_type=self._type, bilingual=self.bilingual,
                                  src_pad=self.src_pad, tgt_pad=self.tgt_pad, feature_size=self.input_size)

        return [batch]

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

