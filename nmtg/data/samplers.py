import numpy as np
import torch.utils.data
from torch.utils.data import BatchSampler


class StatefulSampler(torch.utils.data.Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.index = -1

    def __next__(self):
        raise NotImplementedError

    def __iter__(self):
        return self

    def reset(self):
        """Restart the iterator and regenerate internal state.
        The next pass over this iterator is not guaranteed to be the same."""
        self.index = -1

    def soft_reset(self):
        """Restart the iterator without regenerating internal state.
        The next pass over this iterator is guaranteed to be the same."""
        self.index = -1

    def state_dict(self):
        return {'index': self.index}

    def load_state_dict(self, state_dict):
        self.index = state_dict['index']


class StatefulSequentialSampler(StatefulSampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.length = len(data_source)

    def __next__(self):
        if self.index >= self.length - 1:
            raise StopIteration
        self.index += 1
        return self.index

    def __len__(self):
        return self.length


class StatefulRandomSampler(StatefulSampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.length = len(data_source)
        self.order = None
        self.reset()

    def __next__(self):
        if self.index >= self.length - 1:
            raise StopIteration
        self.index += 1
        return self.order[self.index]

    def __len__(self):
        return self.length

    def reset(self):
        super().reset()
        self.order = np.random.permutation(self.length)

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['order'] = self.order
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.order = state_dict['order']


class PreGeneratedBatchSampler(StatefulSampler):
    """Iterates over a pre-generated set of batch indices, optionally in a random order."""

    def __init__(self, batches, shuffle=False):
        super().__init__(batches)
        self.batches = batches
        self.shuffle = shuffle
        self.batch_order = None
        self.reset()

    def reset(self):
        super().reset()
        if self.shuffle:
            self.batch_order = np.random.permutation(len(self.batches))

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['batch_order'] = self.batch_order
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.batch_order = state_dict['batch_order']
        self.shuffle = self.batch_order is not None

    def __len__(self):
        return len(self.batches)

    def __next__(self):
        if self.index >= len(self.batches) - 1:
            raise StopIteration
        self.index += 1
        index = self.index
        if self.shuffle:
            index = self.batch_order[index]
        return self.batches[index]


class StatefulBatchSampler(PreGeneratedBatchSampler):
    def __init__(self, sampler, batch_size, shuffle_batches=False, drop_incomplete=False):
        batches = [batch for batch in BatchSampler(sampler, batch_size, drop_incomplete)]
        super().__init__(batches, shuffle_batches)
