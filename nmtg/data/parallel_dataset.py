import numpy as np
import torch

from nmtg.data.dataset import Dataset
from nmtg.data.text_lookup_dataset import TextLookupDataset


class ParallelDataset(Dataset):
    def __init__(self, src_data: TextLookupDataset, tgt_data: TextLookupDataset=None, src_lang=None, tgt_lang=None):
        self.src_data = src_data  # technically a duplicate, but it's fine
        self.tgt_data = tgt_data
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __getitem__(self, index):
        source = self.src_data[index]
        res = {'id': index, 'src_indices': source, 'src_size': len(source)}
        if self.src_lang is not None:
            res['src_lang'] = self.src_lang

        if self.tgt_data is not None:
            target = self.tgt_data[index]
            target_input = target[:-1]
            target_output = target[1:]
            res['tgt_input'] = target_input
            res['tgt_output'] = target_output
            res['tgt_size'] = len(target_output)
            if self.tgt_lang is not None:
                res['tgt_lang'] = self.tgt_lang

        return res

    def __len__(self):
        return len(self.src_data)

    def collate_samples(self, samples):
        src_batch = self.src_data.collate_samples([x['src_indices'] for x in samples])
        res = {'src_indices': src_batch['indices'],
               'src_size': src_batch['size'],
               'src_lengths': src_batch['lengths'],
               'id': torch.tensor([x['id'] for x in samples])}
        if 'src_lang' in samples[0]:
            res['src_lang'] = [x['src_lang'] for x in samples]

        if self.tgt_data is not None:
            target_input = self.tgt_data.collate_samples([x['tgt_input'] for x in samples])
            target_output = self.tgt_data.collate_samples([x['tgt_output'] for x in samples])
            res['tgt_input'] = target_input['indices']
            res['tgt_output'] = target_output['indices']
            res['tgt_size'] = target_output['size']
            res['tgt_lengths'] = target_output['lengths']
            if 'tgt_lang' in samples[0]:
                res['tgt_lang'] = [x['tgt_lang'] for x in samples]

        return res


class MultiParallelDataset(Dataset):
    """
    A dataset containing a parallel text in two or more languages.
    When indexing, the indexing order is target language -> source language -> sentence index.
    """

    def __init__(self, exclude_pairs=None, src_bos=False, tgt_lang_bos=False, **datasets):
        assert len(datasets) > 1
        self.datasets = datasets
        self.src_bos = src_bos
        self.tgt_lang_bos = tgt_lang_bos
        self.languages = sorted(self.datasets.keys())
        self.num_sentences = len(self.datasets[self.languages[0]])
        pad_index = self.datasets[self.languages[0]].pad()
        align_right = self.datasets[self.languages[0]].align_right
        self.collate_fn = self.datasets[self.languages[0]].collate_samples

        assert all(len(dataset) == self.num_sentences for dataset in self.datasets)
        assert all(dataset.dictionary.pad() == pad_index for dataset in self.datasets)
        assert all(dataset.align_right == align_right for dataset in self.datasets)
        if exclude_pairs is not None:
            assert all(len(pair) == 2 for pair in exclude_pairs)
            assert all(lang in self.datasets for pair in exclude_pairs for lang in pair)

        self.exclude_pairs = exclude_pairs if exclude_pairs is not None else []
        self.pairs_per_sentence = len(self.datasets) * (len(self.datasets) - 1) - len(self.exclude_pairs)

        exclude_pairs = [(self.languages.index(s), self.languages.index(t)) for s, t in self.exclude_pairs]
        self._excluded_indices = sorted([s * (len(self.datasets) - 1) + (t - 1 if t >= s else t)
                                         for s, t in exclude_pairs])

    def __len__(self):
        return self.num_sentences * self.pairs_per_sentence

    def __getitem__(self, index):
        pair_index, sentence_index = divmod(index, self.num_sentences)
        for excluded in self._excluded_indices:
            if pair_index >= excluded:
                pair_index += 1
            else:
                break

        tgt_index, src_index = divmod(pair_index, len(self.languages) - 1)
        if src_index >= tgt_index:
            src_index += 1

        src_dataset = self.datasets[self.languages[src_index]]
        src_dataset.lang = self.languages[tgt_index]
        src_dataset.bos = self.src_bos
        source = src_dataset[sentence_index]

        tgt_dataset = self.datasets[self.languages[tgt_index]]
        tgt_dataset.lang = self.languages[tgt_index] if self.tgt_lang_bos else None
        tgt_dataset.bos, tgt_dataset.eos = True, True
        target = tgt_dataset[sentence_index]

        res = {'sentence_id': sentence_index,
               'src_lang': self.languages[src_index], 'tgt_lang': self.languages[tgt_index],
               'src_indices': source, 'src_size': len(source),
               'tgt_input': target[:-1], 'tgt_output': target[1:], 'tgt_size': len(target) - 1}
        return res

    def collate_samples(self, samples):
        src_batch = self.collate_fn([x['src_indices'] for x in samples])
        tgt_input = self.collate_fn([x['tgt_input'] for x in samples])
        tgt_output = self.collate_fn([x['tgt_output'] for x in samples])

        res = {'src_lang': [x['src_lang'] for x in samples],
               'tgt_lang': [x['tgt_lang'] for x in samples],
               'src_indices': src_batch['indices'], 'src_size': src_batch['size'], 'src_lengths': src_batch['lengths'],
               'tgt_input': tgt_input['indices'], 'tgt_output': tgt_output['indices'],
               'tgt_size': tgt_output['size'], 'tgt_lengths': tgt_output['lengths']}
        return res

    def concat_lengths(self, **lengths):
        assert all(lang in lengths for lang in self.languages)
        src_langs, tgt_langs = zip(*self.get_pairs())

        src_lengths = np.concatenate([lengths[lang] for lang in src_langs])
        tgt_lengths = np.concatenate([lengths[lang] for lang in tgt_langs])
        return src_lengths, tgt_lengths

    def get_pairs(self):
        pair_indices = np.arange(len(self.languages) * (len(self.languages) - 1) - len(self.exclude_pairs))
        for excluded in self._excluded_indices:
            pair_indices[pair_indices >= excluded] += 1

        src_indices, tgt_indices = np.divmod(pair_indices, len(self.languages) - 1)
        tgt_indices[tgt_indices >= src_indices] += 1
        src_langs = [self.langauges[i] for i in src_indices]
        tgt_langs = [self.languages[i] for i in tgt_indices]
        return list(zip(src_langs, tgt_langs))
