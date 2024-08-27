import torch
import random
import pickle

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

    # note: samples here are measured by minibatches
    def __init__(self, max_samples=40000, update_method="reservoir_sampling",
                 unit="minibatch",
                 batch_size_frames=1,
                 batch_size_words=1,
                 batch_size_sents=1):

        self.max_samples = max_samples

        self.data = dict()

        self.update_method = update_method

        self.num_observed = 0

        # if unit is batch or minibatch, we can just store the
        self.unit = unit

        self.batch_size_frames = batch_size_frames
        self.batch_size_words = batch_size_words
        self.batch_size_sents = batch_size_sents

    # def add_sample(self, sample):
    def add_sample(self, batch):

        # we store the dataset id and the samples
        (dataset_id, indices, src_lengths, tgt_lengths) = batch
        # sample = compress(sample)
        if self.unit in ['batch', 'minibatch']:

            if self.update_method == "reservoir_sampling":
                sample = (dataset_id, indices)

                if len(self.data) < self.max_samples:

                    self.data[len(self.data)] = sample
                    self.num_observed += 1

                else:

                    # random from i to len(self.data) + 1
                    self.num_observed += 1

                    j = random.randint(0, self.num_observed)
                    if j < self.max_samples:
                        del self.data[j]
                        self.data[j] = sample
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

                    else:

                        # random from i to len(self.data) + 1
                        self.num_observed += 1
                        r = random.randint(0, self.num_observed)
                        if r < self.max_samples:
                            del self.data[r]
                            self.data[r] = (dataset_id, index, src_lengths[j],  tgt_lengths[j])

                # if len(self.data) < self.max_samples:
                #
                #     self.data[len(self.data)] = sample
                #     self.num_observed += 1
                #
                # else:
                #
                #     # random from i to len(self.data) + 1
                #     self.num_observed += 1
                #
                #     j = random.randint(0, self.num_observed)
                #     if j < self.max_samples:
                #         del self.data[j]
                #         self.data[j] = sample
            else:
                raise NotImplementedError


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

                    # _is_oversized(cur_batch, new_sent_size, cur_batch_sizes, batch_size_words, batch_size_sents):

                    # since we are working with audio
                    # batch by source
                    # its going to be inefficient but alright
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

                return dataset_ids, indices
            else:

                raise NotImplementedError

