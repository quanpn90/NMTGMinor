from nmtg.data import data_utils
from nmtg.data.dictionary import Dictionary
from nmtg.data.dataset import Dataset, TextLineDataset


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
        return self.dictionary.to_indices(line, bos=self.bos, eos=self.eos)

    def collate_samples(self, samples):
        indices, lengths = data_utils.collate_sequences(samples,
                                                        self.dictionary.pad(),
                                                        self.align_right)
        return {'indices': indices, 'lengths': lengths, 'size': lengths.sum()}
