import itertools
import logging
import math
import operator
import os
import queue
import time
from threading import Thread
from .data_iterator import EpochBatchIterating, DataIterator

import numpy as np
import torch


class MultiEpochIterator(object):
    # this class stores N epoch iterators for N datasets
    # init is called at the beginning of the epoch

    def __init__(self, iterators, round_robin=True):
        """
        :param iterators: a list of CountingIterators
        :param round_robin: if the data is sampled iteratively 1 to N or randomly
        """

        self.iterators = iterators
        self.round_robin = round_robin
        self.n_iterators = len(iterators)
        # self.total = sum([len(iterator) for iterator in self.iterators])
        self.totals = [len(iterator) for iterator in self.iterators]
        self.total = sum(self.totals)
        self.itr = iter(self)

        if self.round_robin:
            self.itr_indices = torch.arange(self.n_iterators)
        else:
            self.itr_indices = torch.randperm(self.n_iterators)

        self.idx = -1
        self.n_yielded = 0

    def iterations_in_epoch(self):
        """
        :return: a list of iterations in epoch for each iterator
        """
        return [iterator.n for iterator in self.iterators]

    def load_iterations(self, iteration_in_epochs):

        for iterator, iter_in_epoch in zip(self.iterators, iteration_in_epochs):
            iterator.n = iter_in_epoch

    def __len__(self):

        return sum([len(iterator) for iterator in self.iterators])

    def __iter__(self):

        while True:

            if self.n_yielded >= self.total:
                return

            self.idx = self.idx + 1
            if self.idx >= self.n_iterators:
                self.idx = 0

            cur_iterator = self.iterators[self.itr_indices[self.idx]]

            # if the current iterator is not exhausted, then yield
            # otherwise go to the next one
            if cur_iterator.has_next():
                self.n_yielded += 1
                yield next(cur_iterator)
            else:
                continue

    def __next__(self):

        return next(self.itr)

    def has_next(self):

        return self.n_yielded < self.total

    def skip(self, num_to_skip):

        for iterator in self.iterators:
            iterator.skip(num_to_skip)

    def take(self, n):

        """
        Truncates the iterator to n elements at most.
        """
        for iterator in self.iterators:
            iterator.take(n)


class MultiDataIterator(EpochBatchIterating):

    def next_epoch_itr(self, shuffle=True, pin_memory=False):
        self.epoch = self.next_epoch_idx
        if self._next_epoch_itr is not None:
            self._cur_epoch_itr = self._next_epoch_itr
            self._next_epoch_itr = None
        else:
            self._cur_epoch_itr = self._get_iterator_for_epoch(
                self.epoch, shuffle, pin_memory=pin_memory
            )
        for dataset in self.datasets:
            dataset.set_epoch(self.epoch)
        self.shuffle = shuffle
        return self._cur_epoch_itr

    # each dataset = dataiterator > generate 1 epoch iterator
    # this class gen
    def __init__(self, datasets, seed=1., num_workers=0, epoch=1, buffer_size=0,
                 timeout=0, round_robin=True, num_shards=1, shard_id=0):

        self.datasets = datasets
        self.data_iterators = list()
        for dataset in datasets:
            self.data_iterators.append(DataIterator(dataset, dataset.collater, dataset.batches, seed=seed,
                                                    num_workers=num_workers, epoch=epoch, buffer_size=buffer_size,
                                                    timeout=timeout, num_shards=num_shards, shard_id=shard_id))

        self.shuffle = True
        self._cur_epoch_itr = None
        self._next_epoch_itr = None
        self._support_prefetch = False
        self.round_robin = round_robin
        self.epoch = max(epoch, 1)
        self.n_samples = sum([dataset.num_batches for dataset in self.datasets])

    def __len__(self):

        return sum([dataset.num_batches for dataset in self.datasets])

    @property
    def next_epoch_idx(self):
        """Return the epoch index after *next_epoch_itr* is called"""
        if self._next_epoch_itr is not None:
            return self.epoch
        elif self._cur_epoch_itr is not None and self.end_of_epoch():
            return self.epoch + 1
        else:
            return self.epoch

    def end_of_epoch(self) -> bool:
        return not self._cur_epoch_itr.has_next()

    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        return {
            'epoch': self.epoch,
            'iterations_in_epoch': self.iterations_in_epoch,
            'shuffle': self.shuffle,
        }

    @property
    def iterations_in_epoch(self):
        """ The number of consumed batches in the current epoch"""
        if self._cur_epoch_itr is not None:
            return self._cur_epoch_itr.iterations_in_epoch()
        elif self._next_epoch_itr is not None:
            return self._next_epoch_itr.iterations_in_epoch()
        return [0] * len(self.data_iterators)

    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        return {
            'epoch': self.epoch,
            'iterations_in_epoch': self.iterations_in_epoch,
            'shuffle': self.shuffle,
        }

    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        if state_dict is not None:
            self.epoch = state_dict['epoch']
            itr_pos = state_dict.get('iterations_in_epoch', [0] * len(self.data_iterators))

            if sum(itr_pos) > 0:
                # fast-forward epoch iterator
                self._next_epoch_itr = self._get_iterator_for_epoch(
                    self.epoch,
                    shuffle=state_dict.get('shuffle', True),
                    offsets=itr_pos
                )
                if self._next_epoch_itr is None:
                    # we finished the epoch, increment epoch counter
                    self.epoch += 1
            else:
                self._next_epoch_itr = None
        else:
            self.epoch = 1
            itr_pos = list()
            self._next_epoch_itr = None

    def _get_iterator_for_epoch(self, epoch, shuffle=False, offsets=None, pin_memory=False):

        epoch_iterators = list()

        if offsets is not None and sum(offsets) >= self.n_samples:
            return None

        if offsets is None:
            offsets = [0] * len(self.data_iterators)

        # first, generate an iterator for each data iterator
        for (data_iterator, offset) in zip(self.data_iterators, offsets):
            epoch_iterator =  data_iterator._get_iterator_for_epoch(epoch, shuffle, offset, pin_memory=pin_memory)
            epoch_iterators.append(epoch_iterator)

        # next, use an multi epoch iterator
        epoch_iterator = MultiEpochIterator(epoch_iterators, round_robin=self.round_robin)

        return epoch_iterator







