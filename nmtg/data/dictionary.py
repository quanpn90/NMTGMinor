import os
from collections import Counter

import torch
from tqdm import tqdm


class Dictionary:
    """A mapping from symbols to consecutive integers"""
    def __init__(self, pad='<pad>', unk='<unk>', bos='<s>', eos='</s>'):
        self.unk_word, self.pad_word, self.bos_word, self.eos_word = unk, pad, bos, eos
        self.symbols = []
        self.count = []
        self.indices = {}

        self.pad_index = self.add_symbol(pad)
        self.unk_index = self.add_symbol(unk)
        self.bos_index = self.add_symbol(bos)
        self.eos_index = self.add_symbol(eos)
        self.nspecial = len(self.symbols)

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.bos_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    def __eq__(self, other):
        return self.indices == other.indices

    def symbol(self, idx):
        """Returns the symbol with the specified index"""
        try:
            return self.symbols[idx]
        except IndexError:
            if idx >= 0:
                return self.unk_word
            else:
                raise

    def index(self, sym):
        """Returns the index of the specified symbol"""
        return self.indices.get(sym, self.unk_index)

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def string(self, tensor, join_str=' ', bpe_symbol=None, escape_unk=False):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() > 1:
            return [self.string(t) for t in tensor]

        unk_string = self.unk_string(escape_unk)

        def token_string(i):
            if i == self.unk():
                return unk_string
            else:
                return self.symbol(i)

        sent = join_str.join(token_string(i) for i in tensor if i not in (self.bos(), self.eos()))
        if bpe_symbol is not None:
            sent = (sent + ' ').replace(bpe_symbol, '').rstrip()
        return sent

    def to_indices(self, words, bos=True, eos=True):
        """Helper for converting a list of words to a tensor of token indices.
        """
        vec = []
        if bos:
            vec.append(self.bos())

        vec.extend(self.index(sym) for sym in words)

        if eos:
            vec.append(self.eos())

        return torch.tensor(vec, dtype=torch.int64)

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return '<{}>'.format(self.unk_word)
        else:
            return self.unk_word

    def add_symbol(self, word, n=1):
        """Adds a word to the dictionary"""
        if word in self.indices:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def update(self, new_dict):
        """Updates counts from new dictionary."""
        for word in new_dict.symbols:
            idx2 = new_dict.indices[word]
            if word in self.indices:
                idx = self.indices[word]
                self.count[idx] = self.count[idx] + new_dict.count[idx2]
            else:
                idx = len(self.symbols)
                self.indices[word] = idx
                self.symbols.append(word)
                self.count.append(new_dict.count[idx2])

    def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
        """Sort symbols by frequency in descending order and optionally truncate,
        ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        if nwords <= 0:
            nwords = len(self)

        new_indices = dict(zip(self.symbols[:self.nspecial], range(self.nspecial)))
        new_symbols = self.symbols[:self.nspecial]
        new_count = self.count[:self.nspecial]

        c = Counter(dict(zip(self.symbols[self.nspecial:], self.count[self.nspecial:])))
        for symbol, count in c.most_common(nwords - self.nspecial):
            if count >= threshold:
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(count)
            else:
                break

        threshold_nwords = len(new_symbols)
        if padding_factor > 1:
            i = 0
            while threshold_nwords % padding_factor != 0:
                symbol = 'madeupword{:04d}'.format(i)
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(0)
                i += 1
                threshold_nwords += 1

        assert len(new_symbols) % padding_factor == 0
        assert len(new_symbols) == len(new_indices)

        self.count = list(new_count)
        self.symbols = list(new_symbols)
        self.indices = new_indices

    @classmethod
    def load(cls, f, ignore_utf_errors=False, **kwargs):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        if isinstance(f, str):
            try:
                if not ignore_utf_errors:
                    with open(f, 'r', encoding='utf-8') as fd:
                        return cls.load(fd)
                else:
                    with open(f, 'r', encoding='utf-8', errors='ignore') as fd:
                        return cls.load(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception("Incorrect encoding detected in {}, please "
                                "rebuild the dataset".format(f))

        d = cls(**kwargs)
        for line in f.readlines():
            idx = line.rfind(' ')
            if idx == -1:
                raise ValueError("Incorrect dictionary format, expected '<token> <cnt>'")
            word = line[:idx]
            count = int(line[idx+1:])
            d.indices[word] = len(d.symbols)
            d.symbols.append(word)
            d.count.append(count)
        return d

    def state_dict(self):
        return {'dict': [(w, c) for w, c in zip(self.symbols, self.count)]}

    def load_state_dict(self, state_dict):
        symbols, count = zip(*state_dict['dict'])
        indices = {w: i for i, w in enumerate(symbols)}
        self.symbols = symbols
        self.count = count
        self.indices = indices

    @classmethod
    def infer_from_text(cls, lines, words=True, lower=False, progress_bar=True, **kwargs):
        """Infer a dictionary from an iterable of text lines (stripped of newlines)
        :param lines: Iterable of text
        :param words: Whether to create a word or a character dictionary
        :param lower: Lowercase the text before reading
        :param progress_bar: Display a progress bar
        """
        counter = Counter()
        for line in tqdm(lines, unit='lines', disable=not progress_bar):
            line = line.rstrip()
            if lower:
                line = line.lower()
            if words:
                line = line.split()
            counter.update(line)
        dictionary = cls(**kwargs)
        for w, c in counter.items():
            dictionary.add_symbol(w, c)
        return dictionary

    def save(self, f):
        """Stores dictionary into a text file"""
        if isinstance(f, str):
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, 'w', encoding='utf-8') as fd:
                return self.save(fd)
        for symbol, count in zip(self.symbols[self.nspecial:], self.count[self.nspecial:]):
            print('{} {}'.format(symbol, count), file=f)

    @staticmethod
    def convert(old_dict):
        pad_word = '<blank>'
        unk_word = '<unk>'
        bos_word = '<s>'
        eos_word = '</s>'
        new_dict = Dictionary(pad=pad_word, unk=unk_word, bos=bos_word, eos=eos_word)
        new_dict.indices = old_dict.labelToIdx
        new_dict.symbols = [old_dict.idxToLabel[i] for i in range(len(old_dict.idxToLabel))]
        new_dict.count = [old_dict.frequencies[old_dict.labelToIdx[w]] for w in new_dict.symbols]
        new_dict.pad_index = new_dict.indices[pad_word]
        new_dict.unk_index = new_dict.indices[unk_word]
        new_dict.bos_index = new_dict.indices[bos_word]
        new_dict.eos_index = new_dict.indices[eos_word]
        return new_dict

    @classmethod
    def from_counters(cls, *counters, **kwargs):
        dictionary = cls(**kwargs)
        for counter in counters:
            for word, count in counter:
                dictionary.add_symbol(word, count)
        return dictionary


class MultilingualDictionary(Dictionary):
    def __init__(self, languages, pad='<pad>', unk='<unk>', eos='</s>'):
        super().__init__(pad=pad, unk=unk, eos=eos)

        self.language_indices = {}

        for lang in languages:
            self.language_indices[lang] = self.add_symbol('##{}##'.format(lang.upper()))

        self.nspecial = len(self.symbols)

    def to_indices(self, words, bos=True, eos=True, lang=None):
        if not bos or lang is None:
            return super().to_indices(words, bos, eos)

        vec = [self.language_indices[lang]]
        vec.extend(self.index(sym) for sym in words)
        if eos:
            vec.append(self.eos())

        return torch.tensor(vec, dtype=torch.int64)
