# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
import io
from typing import Mapping, Sequence

import numpy as np

import torch.utils.data
from torch import Tensor

from nmtg.data import data_utils
from nmtg.data.indexed_data import IndexedData


class _CUDAIterator:
    def __init__(self, iterable):
        self.length = len(iterable)
        self.iterator = iter(iterable)

    def __len__(self):
        return self.length

    def __next__(self):
        return _batch_to_cuda(next(self.iterator))

    def __iter__(self):
        return self


def _batch_to_cuda(batch):
    if isinstance(batch, Tensor):
        return batch.cuda()
    elif isinstance(batch, Mapping):
        return {k: _batch_to_cuda(v) for k, v in batch.items()}
    elif isinstance(batch, Sequence):
        return [_batch_to_cuda(x) for x in batch]
    else:
        return batch


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
        dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=self.collate_samples,
            # pin_memory=cuda,
            drop_last=drop_last)
        if cuda:
            return _CUDAIterator(dataloader)
        else:
            return dataloader


class RawDataset(Dataset):
    def __init__(self, data, filename):
        self.data = data
        self.filename = filename

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @property
    def in_memory(self):
        return isinstance(self.data, IndexedData)

    @classmethod
    def load_into_memory(cls, filename, offsets_filename=None):
        if offsets_filename is None:
            samples = cls.load_raw(filename)
            return cls(samples, filename)
        else:
            offsets = np.load(offsets_filename)
            data = IndexedData(filename, offsets, cls.decode)
            return cls(list(data), filename)

    @classmethod
    def load_indexed(cls, filename, offsets_filename):
        offsets = np.load(offsets_filename)
        data = IndexedData(filename, offsets, cls.decode)
        return cls(data, filename)

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
        return data_bytes.decode('utf-8')[:-1]

    @staticmethod
    def encode(sample):
        return (sample + '\n').encode('utf-8')

    def collate_samples(self, samples):
        return samples


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.length = sum(len(d) for d in datasets)

    def __getitem__(self, index):
        for i, dataset in enumerate(self.datasets):
            if index >= len(dataset):
                index -= len(dataset)
            else:
                out = dataset[index]
                if isinstance(out, dict):
                    out['dataset_index'] = i
                return out

    def __len__(self):
        return self.length

    def collate_samples(self, samples):
        out = self.datasets[0].collate_samples(samples)
        if isinstance(out, dict):
            out['dataset_index'] = [sample['dataset_index'] for sample in samples]


class MultiDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.length = max(datasets, key=len)

    def __getitem__(self, index):
        assert len(index) == len(self.datasets)
        return [dataset[i] if i is not None else None for i, dataset in zip(index, self.datasets)]

    def __len__(self):
        return self.length

    def collate_samples(self, samples):
        # Unreadable version:
        # return [dataset.collate_samples(list(filter(lambda x: x is not None, s)))
        #         for s, dataset in zip(zip(*samples), self.datasets)]

        # Equivalent, but more readable version
        padded_samples = zip(*samples)
        real_samples = [[x for x in padded if x is not None] for padded in padded_samples]
        return [dataset.collate_samples(real) for dataset, real in zip(self.datasets, real_samples)]
