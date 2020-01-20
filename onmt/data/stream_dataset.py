from __future__ import division

import math
import torch
import torch.utils.data
from collections import defaultdict
import onmt
from onmt.speech.Augmenter import Augmenter
from onmt.modules.dropout import switchout

"""
Data management for stream-to-stream models
Two basic classes: 
- Batch stores the input / output sequences, grouped into tensors with the same length (by padding)
- Dataset stores all of the data and 
"""


class Stream(object):
    # An object to manage the data within a stream
    def __init__(self, src_data, tgt_data=None,
                 src_lang_data=None, tgt_lang_data=None,
                 src_type='text',
                 src_align_right=False, tgt_align_right=False,
                 augmenter=None, upsampling=False,
                  **kwargs):
        """
        :param src_data: list of source tensors
        :param tgt_data: list of target tensors
        :param src_lang_data: list of language features for the source (TB finished)
        :param tgt_lang_data: list of language features for the target (TB finished)
        :param src_type: text or audio
        :param src_align_right: if the source sequences are aligned to the right
        :param tgt_align_right: if the target sequences are aligned to the right
        (default False and maybe never changed unless new models need)
        :param reshape_speech: the number of frames to be reshaped
        :param augmenter: using augmentation for speech
        :param merge: if the two sequences are going to be merged for Relative Transformer
        """

        self.tensors = defaultdict(lambda: None)
        self.has_target = False
        self.src_type = src_type
        self.upsampling = upsampling
        self.feature_size = kwargs.get('feature_size', 40)
        self.src_align_right = src_align_right


        if src_data is not None:
            self.tensors['source'], self.tensors['source_pos'], self.src_lengths = \
                                                                    self.collate(src_data,
                                                                                 align_right=self.src_align_right,
                                                                                 type=self.src_type,
                                                                                 augmenter=augmenter)
            self.tensors['source'] = self.tensors['source'].transpose(0, 1).contiguous()
            if self.tensors['source_pos'] is not None:
                self.tensors['source_pos'] = self.tensors['source_pos'].transpose(0, 1)
            self.tensors['src_length'] = torch.LongTensor(self.src_lengths)
            self.src_size = sum(self.src_lengths)

        else:
            self.src_size = 0

        if tgt_data is not None:
            target_full, target_pos, self.tgt_lengths = self.collate(tgt_data, align_right=self.tgt_align_right)
            target_full = target_full.t().contiguous()  # transpose BxT to TxB
            self.tensors['target'] = target_full
            self.tensors['target_input'] = target_full[:-1]
            self.tensors['target_output'] = target_full[1:]
            self.tensors['target_pos'] = target_pos.t().contiguous()[:-1]
            self.tensors['tgt_mask'] = self.tensors['target_output'].ne(onmt.constants.PAD)
            self.has_target = True
            self.tgt_size = sum([len(x) - 1 for x in tgt_data])

        else:
            self.tgt_size = 0

        self.size = len(src_data) if src_data is not None else len(tgt_data)

        if src_lang_data is not None:
            self.tensors['source_lang'] = torch.cat(src_lang_data).long()
        if tgt_lang_data is not None:
            self.tensors['target_lang'] = torch.cat(tgt_lang_data).long()

    def switchout(self, swrate, src_vocab_size, tgt_vocab_size):
        # Switch out function ... currently works with only source text data
        if self.src_type == 'text':
            self.tensors['source'] = switchout(self.tensors['source'], src_vocab_size, swrate, transpose=True)

        if self.has_target:
            self.tensors['target'] = switchout(self.tensors['target'], tgt_vocab_size, swrate, transpose=True, offset=1)
            target_full = self.tensors['target']
            self.tensors['target_input'] = target_full[:-1]
            self.tensors['target_output'] = target_full[1:]
            self.tensors['tgt_mask'] = self.tensors['target_output'].ne(onmt.constants.PAD)

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

    def collate(self, data, type="text", augmenter=None):

        """
        Assembling the individual sequences into one single tensor, included padding
        :param data: the list of sequences in chronological order
        :param type: text or audio
        :param augmenter: for augmentation in audio models
        :return:
        data (list of Torch.Tensor) size 1 x T
        """

        return data
        # # initialize with batch_size * length
        # if type == "text":
        #     lengths = [x.size(0) for x in data]
        #     positions = [torch.arange(length_) for length_ in lengths]
        #     max_length = max(lengths)
        #     tensor = data[0].new(len(data), max_length).fill_(onmt.constants.PAD)
        #     pos = tensor.new(*tensor.size()).fill_(0)
        #
        #     for i in range(len(data)):
        #         data_length = data[i].size(0)
        #         offset = max_length - data_length if align_right else 0
        #         tensor[i].narrow(0, offset, data_length).copy_(data[i])
        #         pos[i].narrow(0, offset, data_length).copy_(positions[i])
        #
        #     return tensor, pos, lengths
        #
        # elif type == "audio":
        #
        #     # First step: on-the-fly processing for the samples
        #     # Reshaping: either downsampling or upsampling
        #     # On the fly augmentation
        #     samples = []
        #
        #     for i in range(len(data)):
        #         sample = data[i]
        #
        #         if augmenter is not None:
        #             sample = augmenter.augment(sample)
        #
        #         if self.upsampling:
        #             sample = sample.view(-1, self.feature_size)
        #
        #         samples.append(sample)
        #
        #     # compute the lengths afte on-the-fly processing
        #     lengths = [x.size(0) for x in samples]
        #
        #     max_length = max(lengths)
        #
        #     # allocate data for the batch speech
        #     feature_size = samples[0].size(1)
        #     batch_size = len(data)
        #
        #     # feature size + 1 because the last dimension is created for padding
        #     tensor = data[0].float().new(batch_size, max_length, feature_size + 1).fill_(onmt.constants.PAD)
        #
        #     for i in range(len(samples)):
        #         sample = samples[i]
        #
        #         data_length = sample.size(0)
        #         offset = max_length - data_length if align_right else 0
        #
        #         tensor[i].narrow(0, offset, data_length).narrow(1, 1, sample.size(1)).copy_(sample)
        #         # in padding dimension: 0 is not padded, 1 is padded
        #         tensor[i].narrow(0, offset, data_length).narrow(1, 0, 1).fill_(1)
        #
        #     return tensor, None, lengths
        # else:
        #     raise NotImplementedError

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