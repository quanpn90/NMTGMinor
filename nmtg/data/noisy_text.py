import torch

import numpy as np

from nmtg.data import Dataset
from nmtg.data.parallel_dataset import MultiParallelDataset
from nmtg.data.text_lookup_dataset import TextLookupDataset


class NoisyTextDataset(Dataset):
    def __init__(self, lookup_dataset: TextLookupDataset, word_shuffle=3, word_dropout=0.1, word_blank=0.2,
                 bpe_symbol='@@ '):
        self.source = lookup_dataset
        self.word_shuffle = word_shuffle
        self.word_dropout = word_dropout
        self.word_blank = word_blank
        self.bpe_symbol = bpe_symbol
        self.dictionary = lookup_dataset.dictionary

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        sample = self.source[index]
        sample = self.shuffle(sample)
        sample = self.dropout(sample)
        sample = self.blank(sample)
        return sample

    def shuffle(self, sample):
        if self.word_shuffle == 0:
            return sample

        length = len(sample)
        if self.source.eos:
            length -= 1

        noise = np.random.uniform(0, self.word_shuffle, size=length)
        if self.source.bos:
            noise[0] = -1  # do not move the start of sentence symbol

        word_idx = self.get_word_idx(sample)

        scores = word_idx[:length] + noise[word_idx[:length]]
        scores += 1e-6 * np.arange(length)  # ensure no reordering inside a word
        permutation = scores.argsort()

        # shuffle words
        new_sample = sample.clone()
        new_sample[:length].copy_(new_sample[torch.from_numpy(permutation)])
        return new_sample

    def dropout(self, sample):
        if self.word_dropout == 0:
            return sample

        word_idx = self.get_word_idx(sample)
        keep = np.random.rand(word_idx.max() + 1) >= self.word_dropout
        keep = keep.astype(np.uint8)
        start = 0
        end = len(keep)
        if self.source.bos:
            keep[0] = 1
            start += 1
        if self.source.eos:
            keep[-1] = 1
            end -= 1

        # We need to keep at least one word
        if keep[start:end].sum() == 0:
            assert end - start > 0
            keep[np.random.randint(end)] = 1

        # drop the words
        sample = sample.masked_select(torch.from_numpy(keep[word_idx]))
        return sample

    def blank(self, sample):
        if self.word_blank == 0:
            return sample

        word_idx = self.get_word_idx(sample)
        drop = np.random.rand(word_idx.max() + 1) < self.word_blank
        drop = drop.astype(np.uint8)
        if self.source.bos:
            drop[0] = 1
        if self.source.eos:
            drop[-1] = 1

        new_sample = sample.clone()
        new_sample.masked_fill_(torch.from_numpy(drop[word_idx]), self.dictionary.unk())
        return new_sample

    def collate_samples(self, samples):
        return self.source.collate_samples(samples)

    def get_word_idx(self, sample):
        if self.bpe_symbol is not None:
            bpe_symbol = self.bpe_symbol.rstrip()
            bpe_end = np.array([not self.dictionary.symbol(ind).endswith(bpe_symbol) for ind in sample])
            word_idx = bpe_end[::-1].cumsum()[::-1]
            word_idx = word_idx.max() - word_idx
        else:
            word_idx = np.arange(len(sample))
        return word_idx


class NoisyMultiParallelDataset(MultiParallelDataset):

    def __init__(self, exclude_pairs=None, src_bos=False, tgt_lang_bos=False, word_shuffle=3,
                 word_dropout=0.1, word_blank=0.2, bpe_symbol='@@ ', noisy_languages=tuple(), **datasets):
        super().__init__(exclude_pairs, src_bos, tgt_lang_bos, **datasets)
        self.word_shuffle = word_shuffle
        self.word_dropout = word_dropout
        self.word_blank = word_blank
        self.bpe_symbol = bpe_symbol
        self.noisy_languages = set(noisy_languages)

    def _get_src(self, src_lang, tgt_lang, index):
        sample = super()._get_src(src_lang, tgt_lang, index)
        if src_lang not in self.noisy_languages:
            return sample

        self.source = self.datasets[src_lang]
        self.dictionary = self.source.dictionary
        sample = self.shuffle(sample)
        sample = self.dropout(sample)
        sample = self.blank(sample)
        return sample

    shuffle = NoisyTextDataset.shuffle
    dropout = NoisyTextDataset.dropout
    blank = NoisyTextDataset.blank
    get_word_idx = NoisyTextDataset.get_word_idx
