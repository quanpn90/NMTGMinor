import logging
import os

import numpy as np

from nmtg.data import data_utils
from nmtg.data.dictionary import Dictionary
from nmtg.data.dataset import Dataset, TextLineDataset


logger = logging.getLogger(__name__)


class TextLookupDataset(Dataset):
    def __init__(self, text_dataset: TextLineDataset, dictionary: Dictionary, words=True,
                 lower=False, bos=True, eos=True, align_right=False, trunc_len=0):
        """
        A dataset which contains indices derived by splitting
        text lines and looking up indices in a dictionary
        :param text_dataset: The source text
        :param dictionary: The lookup
        :param words: Whether to split characters or words
        :param bos: Whether to include a beginning-of-sequence token
        :param eos: Whether to include an end-of-sequence token
        :param align_right: Whether to align the padded batches to the right
        """
        self.source = text_dataset
        self.dictionary = dictionary
        self.words = words
        self.lower = lower
        self.bos = bos
        self.eos = eos
        self.align_right = align_right
        self.trunc_len = trunc_len

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        line = self.source[index]
        if self.lower:
            line = line.lower()
        if self.words:
            line = line.split(' ')
        if self.trunc_len > 0:
            line = line[:self.trunc_len]
        if len(line) == 0:
            logger.warning('Zero-length input at {}'.format(index))
        return self.dictionary.to_indices(line, bos=self.bos, eos=self.eos)

    def collate_samples(self, samples):
        indices, lengths = data_utils.collate_sequences(samples,
                                                        self.dictionary.pad(),
                                                        self.align_right)
        return {'indices': indices, 'lengths': lengths, 'size': lengths.sum().item()}

    @classmethod
    def load(cls, filename, dictionary, data_dir, load_into_memory=False, words=True, *args, **kwargs):
        base_name = os.path.basename(filename)

        if load_into_memory:
            text_data = TextLineDataset.load_into_memory(filename)
            if words:
                lengths = np.array([len(sample.split()) for sample in text_data])
            else:
                lengths = np.array([len(sample) for sample in text_data])
        else:
            offsets_filename = os.path.join(data_dir, base_name + '.idx.npy')
            text_data = TextLineDataset.load_indexed(filename, offsets_filename)
            lengths_filename = os.path.join(data_dir, base_name + '.len.npy')
            lengths = np.load(lengths_filename)

        return cls(text_data, dictionary, words, *args, **kwargs), lengths
