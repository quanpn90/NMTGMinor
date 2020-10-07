# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.

import itertools
import logging
import math
import operator
import os
import queue
import time
from threading import Thread

import numpy as np
import torch
from onmt.data.dataset import rewrap

from onmt.data import data_utils

_sentinel = object()


class CountingIterator(object):
    """Wrapper around an iterable that maintains the iteration count.
    Args:
        iterable (iterable): iterable to wrap
        start (int): starting iteration count. Note that this doesn't
            actually advance the iterator.
        total (int): override the iterator length returned by
            ``__len__``. This can be used to truncate *iterator*.
    Attributes:
        n (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable, start=None, total=None):
        self.iterable = iterable
        self.itr = iter(self)

        if start is None:
            self.n = getattr(iterable, 'n', 0)
        else:
            self.n = start

        if total is None:
            self.total = self.n + len(iterable)
        else:
            self.total = total

    def __len__(self):
        return self.total

    def __iter__(self):
        for x in self.iterable:
            if self.n >= self.total:
                return
            self.n += 1
            yield x

    def __next__(self):
        return next(self.itr)

    def has_next(self):
        """Whether the iterator has been exhausted."""
        return self.n < len(self)

    def skip(self, num_to_skip):
        """Fast-forward the iterator by skipping *num_to_skip* elements."""
        next(itertools.islice(self.itr, num_to_skip, num_to_skip), None)
        return self

    def take(self, n):
        """
        Truncates the iterator to n elements at most.
        """
        self.total = min(self.total, n)

        # Propagate this change to the underlying iterator
        if hasattr(self.iterable, "take"):
            self.iterable.take(n)


class EpochBatchIterating(object):
    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def next_epoch_idx(self):
        raise NotImplementedError

    def next_epoch_itr(self, shuffle=True, pin_memory=False):
        """Return a new iterator over the dataset.
        Args:
            :param shuffle: (bool, optional): shuffle batches before returning the
            iterator (default: True).
            :param pin_memory: bool
        """
        raise NotImplementedError

    def end_of_epoch(self) -> bool:
        """Returns whether the most recent epoch iterator has been exhausted"""
        raise NotImplementedError

    @property
    def iterations_in_epoch(self) -> int:
        """The number of consumed batches in the current epoch."""
        raise NotImplementedError

    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        raise NotImplementedError


"""A multi-epoch iterator over a :class:`torch.utils.data.Dataset`.
Compared to :class:`torch.utils.data.DataLoader`, this iterator:

dataset (~torch.utils.data.Dataset)
"""


class DataIterator(EpochBatchIterating):

    def __init__(self, dataset, collate_fn, batch_sampler, seed=1, num_workers=0,
                 epoch=1, buffer_size=0, timeout=0, num_shards=1, shard_id=0):
        """
        :param dataset:
        :param collate_fn:
        :param batch_sampler:
        :param seed:
        :param num_workers:
        :param epoch:
        :param buffer_size:
        :param timeout:
        :param shard_id: equivalent with rank
        :param num_shards: equivalent with world size
        """
        assert isinstance(dataset, torch.utils.data.Dataset)

        self.dataset = dataset
        self.collate_fn = collate_fn
        self.frozen_batches = tuple(batch_sampler)  # ??
        self.seed = seed
        self.num_workers = num_workers
        self.epoch = max(epoch, 1)
        self.buffer_size = buffer_size
        self.timeout = timeout

        self.shard_id = shard_id
        self.num_shards = num_shards

        self.shuffle = True
        self._cur_epoch_itr = None
        self._next_epoch_itr = None
        self._support_prefetch = False

    def __len__(self):
        # number of minibatches, or ???
        return len(self.frozen_batches)

    @property
    def next_epoch_idx(self):
        """Return the epoch index after *next_epoch_itr* is called"""
        if self._next_epoch_itr is not None:
            return self.epoch
        elif self._cur_epoch_itr is not None and self.end_of_epoch():
            return self.epoch + 1
        else:
            return self.epoch

    def next_epoch_itr(self, shuffle=True, pin_memory=False):
        """
        Return a new iterator over the dataset

        :param pin_memory:
        :param shuffle:
        :return:
        """
        self.epoch = self.next_epoch_idx
        if self._next_epoch_itr is not None:
            self._cur_epoch_itr = self._next_epoch_itr
            self._next_epoch_itr = None
        else:
            self._cur_epoch_itr = self._get_iterator_for_epoch(
                self.epoch, shuffle, pin_memory=pin_memory
            )
        self.dataset.set_epoch(self.epoch)
        self.shuffle = shuffle
        return self._cur_epoch_itr

    def end_of_epoch(self) -> bool:
        return not self._cur_epoch_itr.has_next()

    @property
    def iterations_in_epoch(self):
        """ The number of consumed batches in the current epoch"""
        if self._cur_epoch_itr is not None:
            return self._cur_epoch_itr.n
        elif self._next_epoch_itr is not None:
            return self._next_epoch_itr.n
        return 0

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
            itr_pos = state_dict.get('iterations_in_epoch', 0)
            if itr_pos > 0:
                # fast-forward epoch iterator
                self._next_epoch_itr = self._get_iterator_for_epoch(
                    self.epoch,
                    shuffle=state_dict.get('shuffle', True),
                    offset=itr_pos,
                )
                if self._next_epoch_itr is None:
                    # we finished the epoch, increment epoch counter
                    self.epoch += 1
            else:
                self._next_epoch_itr = None
        else:
            self.epoch = 1
            itr_pos = 0
            self._next_epoch_itr = None

    def _get_iterator_for_epoch(self, epoch, shuffle, offset=0, pin_memory=False):

        def shuffle_batches(batches_, seed):
            with data_utils.numpy_seed(seed):
                np.random.shuffle(batches_)

            return batches_

        if self._support_prefetch:
            raise NotImplementedError

        if shuffle:
            batches = shuffle_batches(list(self.frozen_batches), self.seed + epoch)
        else:
            batches = self.frozen_batches

        #
        num_shards = self.num_shards
        batches = list(ShardedIterator(batches, num_shards, self.shard_id, fill_value=None))

        if offset > 0 and offset >= len(batches):
            return None

        if self.num_workers > 0:
            os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

        # Create data loader
        itr = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_sampler=batches[offset:],
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            timeout=self.timeout,
        )

        # Wrap with a BufferedIterator if needed
        if self.buffer_size > 0:
            itr = BufferedIterator(self.buffer_size, itr)

        # Wrap with CoutingIterator
        itr = CountingIterator(itr, start=offset)
        return itr


class ShardedIterator(CountingIterator):
    """A sharded wrapper around an iterable, padded to length.
    Args:
        iterable (iterable): iterable to wrap
        num_shards (int): number of shards to split the iterable into
        shard_id (int): which shard to iterator over
        fill_value (Any, optional): padding value when the iterable doesn't
            evenly divide *num_shards* (default: None).
    Attributes:
        n (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable, num_shards, shard_id, fill_value=None):

        # assert num_shards == 1
        # assert shard_id == 0

        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError('shard_id must be between 0 and num_shards')
        sharded_len = int(math.ceil(len(iterable) / float(num_shards)))

        # first islice takes a list of minibatch-ids from shard_id to max, every num_shards
        # next, zip_longest takes the zip between (0, 1, ... n) and the minibatches (longest, fill the latter with [])
        # next, map will apply the function taking the minibatches to return the iterator
        itr = map(
            operator.itemgetter(1),
            itertools.zip_longest(
                range(sharded_len),
                itertools.islice(iterable, shard_id, len(iterable), num_shards),
                fillvalue=fill_value,
            ),
        )
        super().__init__(
            itr,
            start=int(math.ceil(getattr(iterable, 'n', 0) / float(num_shards))),
            total=sharded_len,
        )


class BackgroundConsumer(Thread):
    def __init__(self, queue, source, max_len):
        Thread.__init__(self)

        self._queue = queue
        self._source = source
        self._max_len = max_len
        self.count = 0

    def run(self):
        try:
            self._source_iter = iter(self._source)
            for _ in range(len(self._source)):
                item = next(self._source_iter)
                self._queue.put(item)

                # Stop if we reached the maximum length
                self.count += 1
                if self._max_len is not None and self.count >= self._max_len:
                    break

            # Signal the consumer we are done.
            self._queue.put(_sentinel)
        except Exception as e:
            self._queue.put(e)

        del self._source_iter


class BufferedIterator(object):
    def __init__(self, size, iterable):
        self._queue = queue.Queue(size)
        self._iterable = iterable
        self.max_len = None
        self._consumer = None

        self.start_time = time.time()
        self.warning_time = None

    def _create_consumer(self):
        self._consumer = BackgroundConsumer(
            self._queue,
            self._iterable,
            self.max_len
        )
        self._consumer.daemon = True
        self._consumer.start()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._iterable)

    def take(self, n):
        self.max_len = n

    def __next__(self):
        # Create consumer if not created yet
        if self._consumer is None:
            self._create_consumer()

        # Notify the user if there is a data loading bottleneck
        if self._queue.qsize() < max(1, self._queue.maxsize // 2):
            if time.time() - self.start_time > 5 * 60:
                if self.warning_time is None or time.time() - self.warning_time > 15 * 60:
                    # print(
                    #     "Data loading buffer is empty or nearly empty (%d). This may "
                    #     "indicate a data loading bottleneck, and increasing the "
                    #     "number of workers (--num-workers) may help." % self._queue.qsize()
                    # )
                    self.warning_time = time.time()

        # Get next example
        item = self._queue.get(True)
        if isinstance(item, Exception):
            raise item
        if item is _sentinel:
            raise StopIteration()
        return item
