import torch
import random
import pickle
from collections import defaultdict

from onmt.data.batch_utils import _is_oversized

try:
    import zstd
    compressor = zstd
except ModuleNotFoundError:
    import zlib
    compressor = zlib


def compress(x):
    return compressor.compress(pickle.dumps(x))

def uncompress(z):
    return pickle.loads(compressor.decompress(z))


class Reservoir:

    def get_stats(self):

        return self.total_per_dataset

    def state_dict(self):

        state_dict = {'data': self.data,
                      'num_observed': self.num_observed,
                      'unit': self.unit,
                      'update_method': self.update_method,
                      'weights': self.weights}

        return state_dict

    def get_stats_per_dataset(self):

        print(len(self.data))
        for j in self.data:

            sample = self.data[j]
            if self.unit in ['batch', 'minibatch']:
                dataset_id, index = sample
            else:
                dataset_id, index, _, _ = sample

            if dataset_id not in self.stats:
                self.stats[dataset_id] = list()

            self.stats[dataset_id].append(index)
            # print(dataset_id, index, len(self.stats[dataset_id]))


    def load_state_dict(self, state_dict):

        self.data = state_dict['data']
        self.num_observed = state_dict['num_observed']
        self.unit = state_dict['unit']
        self.update_method = state_dict['update_method']

        weights = state_dict['weights'] if 'weights' in state_dict else None
        if weights is not None:
            self.weights = weights

        self.get_stats_per_dataset()

    def initialize_weights(self):
        if not self.weighting:
            return
        # lagrangian weights for each memory sample in the buffer
        self.weights = torch.nn.Parameter(torch.randn(self.max_samples))
        self.weights_ones = torch.nn.Parameter(torch.ones(self.max_samples))

    def cuda(self):
        if self.weights is not None:
            self.weights = self.weights.cuda()
            self.weights_ones = self.weights_ones.cuda()

    def parameters(self):

        return [self.weights]

    # note: samples here are measured by minibatches
    def __init__(self, max_samples=40000, update_method="reservoir_sampling",
                 unit="minibatch",
                 batch_size_frames=1,
                 batch_size_words=1,
                 batch_size_sents=1,
                 weighting=False):

        # default as 0
        self.total_per_dataset = defaultdict(int)
        self.max_samples = max_samples

        self.data = dict()

        self.update_method = update_method

        self.num_observed = 0

        # if unit is batch or minibatch, we can just store the
        self.unit = unit

        self.batch_size_frames = batch_size_frames
        self.batch_size_words = batch_size_words
        self.batch_size_sents = batch_size_sents
        self.stats = dict()

        # TODO: initialize weights per sample if needed
        self.weighting = weighting
        self.static = weighting
        self.weights = None
        self.batch_data = []
        self.initialize_weights()
        self.create_minidataset()

    # def add_sample(self, sample):
    def add_sample(self, batch, recurring=False):
        """
        Args:
            batch: a list of data samples
            recurring: if recurring: don't increase the observed (not sure)

        Returns:

        """

        # we store the dataset id and the samples
        (dataset_id, indices, src_lengths, tgt_lengths) = batch
        # sample = compress(sample)
        if self.unit in ['batch', 'minibatch']:

            if self.update_method == "reservoir_sampling":
                sample = (dataset_id, indices)

                if len(self.data) < self.max_samples:

                    self.data[len(self.data)] = sample
                    self.num_observed += 1

                    self.total_per_dataset[dataset_id] += len(indices)

                else:

                    # random from i to len(self.data) + 1
                    self.num_observed += 1

                    j = random.randint(0, self.num_observed)
                    if j < self.max_samples:

                        to_delete = self.data[j]
                        _dataset_id, _indices = to_delete
                        self.total_per_dataset[_dataset_id] -= len(_indices)

                        del self.data[j]
                        self.data[j] = sample

                        # reset weight for that one
                        if self.weights is not None:
                            self.weights.data[j].normal_(0, 1)

                        self.total_per_dataset[dataset_id] += len(indices)
            else:
                raise NotImplementedError

        elif self.unit in ['sample', 'samples']:

            if self.update_method == "reservoir_sampling":

                (dataset_id, indices, src_lengths, tgt_lengths) = batch
                assert len(src_lengths) == len(tgt_lengths)
                assert len(src_lengths) == len(indices)

                # add each index/dataset_id into the reservoir
                for j, index in enumerate(indices):

                    if len(self.data) < self.max_samples:

                        # we have to store more information in this case
                        self.data[len(self.data)] = (dataset_id, index, src_lengths[j],  tgt_lengths[j])
                        self.num_observed += 1

                        self.total_per_dataset[dataset_id] += 1

                    else:

                        # random from i to len(self.data) + 1
                        self.num_observed += 1
                        r = random.randint(0, self.num_observed)
                        if r < self.max_samples:

                            to_delete = self.data[r]
                            _dataset_id, _, _, _ = to_delete
                            self.total_per_dataset[_dataset_id] -= 1
                            del self.data[r]

                            self.data[r] = (dataset_id, index, src_lengths[j],  tgt_lengths[j])
                            self.total_per_dataset[dataset_id] += 1

                            assert len(self.data) == self.max_samples

            else:
                raise NotImplementedError

    def create_minidataset(self):

        # first we need to sort
        all_data = self.data.values()

        # sort in descending order
        sorted_data = sorted(enumerate(all_data), key=lambda x: x[1][2], reverse=True)

        # now we need to create the minibatches
        batches = list()
        current_batch = list()
        current_batch_sizes = list()
        for sample in sorted_data:
            reservoir_index = sample[0]
            dataset_id, index, src_length, tgt_length = sample[1]
            sample_data = (dataset_id, index, reservoir_index)

            if _is_oversized(current_batch, src_length, current_batch_sizes,
                             self.batch_size_frames, self.batch_size_sents):

                # if the current sample cannot be added to the minibatch
                batches.append(current_batch)
                current_batch = list()
                current_batch_sizes = list()

            else:
                current_batch.append(sample_data)
                current_batch_sizes.append(src_length)

        if len(current_batch) > 0:
            batches.append(current_batch)

        # convert to the data format that the dataset accepts (a list of dataset ids and a list of indices)
        batch_data = list()
        for sample in batches:
            _dataset_ids = [_x[0] for _x in sample]
            _indices = [_x[1] for _x in sample]
            _reservoir_ids = [_x[2] for _x in sample]

            with torch.no_grad():
                lagrangian_weights = self.weights[_reservoir_ids]

            batch_data.append((_dataset_ids, _indices, lagrangian_weights, _reservoir_ids))

        self.batch_data = batch_data
        return

    def get_samples(self, worker=0, num_workers=1):

        assert self.unit in ['sample', 'samples']

        # regenerate the minibatches if dynamic, otherwise reuse
        if not self.static:
            self.create_minidataset()

        if len(self.batch_data) == 0:
            self.create_minidataset()

        batch_data = self.batch_data

        # split the list into
        num_batches = len(batch_data)
        avg_chunk_size = num_batches // num_workers
        remainder = num_batches % num_workers

        sub_lists = []
        start = 0

        for i in range(num_workers):
            # Calculate the end index for this chunk
            end = start + avg_chunk_size + (1 if i < remainder else 0)
            sub_lists.append(batch_data[start:end])
            start = end  # Move start to the next chunk's beginning

        assert worker < len(sub_lists)
        return sub_lists[worker], num_batches

    def sample(self):

        if self.unit in ['batch', 'minibatch']:

            if self.update_method == "reservoir_sampling":
                j = random.randint(0, len(self.data) - 1)

                dataset_id, indices = self.data[j]

                # return uncompress(self.data[j])
                return [dataset_id] * len(indices), indices
            else:

                raise NotImplementedError

        elif self.unit in ['sample', 'samples']:

            if self.update_method == "reservoir_sampling":

                current_bsz_src = 0
                current_bsz_tgt = 0
                sampled_ids = dict()
                current_batch = list()
                current_batch_sizes = list()

                while True:

                    j = random.randint(0, len(self.data) - 1)

                    # maybe we need another option for non-overlapping?
                    if j in sampled_ids:
                        continue

                    dataset_id, index, src_length, tgt_length = self.data[j]
                    randomized_sample = (dataset_id, index)

                    # try to add more in current minibatch: if its full then break
                    # less efficient than the dataset minibatch sorting and grouping
                    if _is_oversized(current_batch, src_length, current_batch_sizes,
                                     self.batch_size_frames, self.batch_size_sents):

                        break

                    else:
                        current_batch.append(randomized_sample)
                        current_batch_sizes.append(src_length)
                        sampled_ids[j] = 1

                randomized_samples = current_batch

                dataset_ids = [sample[0] for sample in randomized_samples]
                indices = [sample[1] for sample in randomized_samples]

                # TODO: generate weights
                if self.weighting:

                    # not sure while the gradients don't flow back. TODO: fix this
                    lagrangian_weights = self.weights[list(sampled_ids.keys())]
                    return dataset_ids, indices, lagrangian_weights

                else:
                    lagrangian_weights = None
                    return dataset_ids, indices

            else:

                raise NotImplementedError

