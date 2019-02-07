# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
import io

import numpy as np

import torch.utils.data

from nmtg.data import data_utils
from nmtg.data.indexed_data import IndexedData


class Dataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def collate_samples(self, samples):
        """Collect the specified samples into a minibatch.
        :param samples: list of samples from this dataset
        :return: The samples, collected into a minibatch
        """
        raise NotImplementedError

    def get_iterator(self, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                     num_workers=0, cuda=False, drop_last=False):
        """See torch.utils.data.DataLoader"""
        return torch.utils.data.DataLoader(self,
                                           batch_size=1,
                                           shuffle=shuffle,
                                           sampler=sampler,
                                           batch_sampler=batch_sampler,
                                           num_workers=num_workers,
                                           collate_fn=self.collate_samples,
                                           pin_memory=cuda,
                                           drop_last=drop_last)


class RawDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @classmethod
    def load_into_memory(cls, filename, offsets_filename=None):
        if offsets_filename is None:
            samples = cls.load_raw(filename)
            return cls(samples)
        else:
            offsets = np.load(offsets_filename)
            data = IndexedData(filename, offsets, cls.decode)
            return cls(list(data))

    @classmethod
    def load_indexed(cls, filename, offsets_filename):
        offsets = np.load(offsets_filename)
        data = IndexedData(filename, offsets, cls.decode)
        return cls(data)

    def save(self, filename, offsets_filename=None):
        if offsets_filename is None:
            self.save_raw(filename, self)
        else:
            IndexedData.save(self, filename, offsets_filename, self.encode)

    @staticmethod
    def load_raw(filename):
        """Load the entire dataset in raw format.
        :param filename: Input filename
        :return An iterable of samples (not necessarily an instance of Dataset)"""
        raise NotImplementedError

    @staticmethod
    def save_raw(filename, samples):
        """Save the entire dataset in raw format.
        :param filename: Output filename
        :param samples: An iterable of samples"""
        raise NotImplementedError

    @staticmethod
    def decode(data_bytes):
        """Decode a single sample from raw bytes."""
        raise NotImplementedError

    @staticmethod
    def encode(sample):
        """Encode a single sample into bytes."""
        raise NotImplementedError


class TensorDataset(RawDataset):
    """A dataset consisting of a structure of tensors."""
    @staticmethod
    def load_raw(filename):
        return torch.load(filename, map_location='cpu')

    @staticmethod
    def save_raw(filename, samples):
        torch.save(samples, filename)

    @staticmethod
    def decode(data_bytes):
        buffer = io.BytesIO(data_bytes)
        return torch.load(buffer, map_location='cpu')

    @staticmethod
    def encode(sample):
        buffer = io.BytesIO()
        torch.save(buffer, sample)
        return buffer.getvalue()

    def collate_samples(self, samples):
        return data_utils.collate_tensor_structures(samples)


class TextLineDataset(RawDataset):
    """A dataset consisting of lines of text."""

    @staticmethod
    def load_raw(filename):
        with open(filename) as f:
            return [line[:-1] for line in f]

    @staticmethod
    def save_raw(filename, samples):
        with open(filename, 'w') as f:
            f.writelines(line + '\n' for line in samples)

    @staticmethod
    def decode(data_bytes):
        data_bytes.decode('utf-8')

    @staticmethod
    def encode(sample):
        sample.encode('utf-8')

    def collate_samples(self, samples):
        return samples
