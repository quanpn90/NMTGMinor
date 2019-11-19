import torch
import math
import random, string


class Dict(object):
    def __init__(self, data=None, lower=False):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.frequencies = {}
        self.lower = lower

        # Special entries will not be pruned.
        self.special = []

        if data is not None:
            if type(data) == str:
                self.loadFile(data)
            else:
                self.addSpecials(data)

    def size(self):
        return len(self.idxToLabel)

    def loadFile(self, filename):
        "Load entries from a file."
        for line in open(filename):

            # NOTE: a vocab entry might be a space
            # so we want to find the right most space index in the line
            # the left part is the label
            # the right part is the index

            right_space_idx = line.rfind(' ')
            label = line[:right_space_idx]
            idx = int(line[right_space_idx+1:])

            print(label, idx)
            self.add(label, idx)

    def writeFile(self, filename):
        "Write entries to a file."
        with open(filename, 'w') as file:
            for i in range(self.size()):
                label = self.idxToLabel[i]
                file.write('%s %d\n' % (label, i))

        file.close()

    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.labelToIdx[key]
        except KeyError:
            return default

    def getLabel(self, idx, default=None):
        try:
            return self.idxToLabel[idx]
        except KeyError:
            return default

    def addSpecial(self, label, idx=None):
        "Mark this `label` and `idx` as special (i.e. will not be pruned)."
        idx = self.add(label, idx)
        self.special += [idx]

    def addSpecials(self, labels):
        "Mark all labels in `labels` as specials (i.e. will not be pruned)."
        for label in labels:
            self.addSpecial(label)

    def add(self, label, idx=None, num=1):
        "Add `label` in the dictionary. Use `idx` as its index if given."
        label = label.lower() if self.lower else label
        if idx is not None:
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        else:
            if label in self.labelToIdx:
                idx = self.labelToIdx[label]
            else:
                idx = len(self.idxToLabel)
                self.idxToLabel[idx] = label
                self.labelToIdx[label] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = num
        else:
            self.frequencies[idx] += num

        return idx

    def prune(self, size):
        "Return a new dictionary with the `size` most frequent entries."
        if size >= self.size():
            return self

        # Only keep the `size` most frequent entries.
        freq = torch.Tensor(
                [self.frequencies[i] for i in range(len(self.frequencies))])
        _, idx = torch.sort(freq, 0, True)

        newDict = Dict()
        newDict.lower = self.lower
        
        count = 0
        # Add special entries in all cases.
        for i in self.special:
            newDict.addSpecial(self.idxToLabel[i])
            count = count + 1

        for i in idx.tolist():
            newDict.add(self.idxToLabel[i])
            count = count + 1
            
            if count >= size:
                break

        return newDict

    def convertToIdx(self, labels, unkWord, bos_word=None, eos_word=None, type='int64'):
        """
        Convert `labels` to indices. Use `unkWord` if not found.
        Optionally insert `bos_word` at the beginning and `eos_word` at the .
        """
        vec = []

        if bos_word is not None:
            vec += [self.lookup(bos_word)]

        unk = self.lookup(unkWord)
        vec += [self.lookup(label, default=unk) for label in labels]

        if eos_word is not None:
            vec += [self.lookup(eos_word)]

        if type == 'int64':
            return torch.LongTensor(vec)
        elif type == 'int32' or type == 'int':
            return torch.IntTensor(vec)
        elif type == 'int16':
            return torch.ShortTensor(vec)
        else:
            raise NotImplementedError

    def convertToIdx2(self, labels, unkWord, bos_word=None, eos_word=None):
        """
        Convert `labels` to indices. Use `unkWord` if not found.
        Optionally insert `bos_word` at the beginning and `eos_word` at the .
        """
        vec = []

        if bos_word is not None:
            vec += [self.lookup(bos_word)]

        unk = self.lookup(unkWord)
        vec += [self.lookup(label, default=unk) for label in labels]

        if eos_word is not None:
            vec += [self.lookup(eos_word)]

        return torch.LongTensor(vec)

    def convertToLabels(self, idx, stop):
        """
        Convert `idx` to labels.
        If index `stop` is reached, convert it and return.
        """
        #~ print(self.idxToLabel)
        labels = []

        for i in idx:

            word = self.getLabel(int(i))
            labels += [word]
            if i == stop:
                break

        return labels

    # Adding crap stuff so that the vocab size divides by the multiplier
    # Help computation with tensor cores
    # This may create bad effect with label smoothing
    # But who knows?
    def patch(self, multiplier=8):

        # self.idxToLabel = {}
        # self.labelToIdx = {}
        # self.frequencies = {}
        size = self.size()

        # number of words to be patched
        n_words = (math.ceil(size / multiplier) * multiplier) - size

        for i in range(n_words):

            while True:
                l_ = 6
                random_string = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(l_))

                if random_string in self.labelToIdx:
                    continue
                else:
                    self.add(random_string)
                    self.frequencies[random_string] = 0
                    break

        print("Vocabulary size after patching: %d" % self.size())

    # @staticmethod
    # def __read_from_single_thread():
    #
    # @staticmethod
    # def __read_from_text_file(self):



