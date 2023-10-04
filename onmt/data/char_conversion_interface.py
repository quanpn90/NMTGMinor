# this file defines different methods of converting a sequence of bpe to low level representations
# characters
# phonemes
# bytes (possible?)


class BPEConverter:

    def convert(self, bpe_seq):

        pass


class BPEtochar(BPEConverter):

    def __init__(self, bpe2char=None, bpeidto2charid=None, char2id=None):

        self.bpe2char = bpe2char
        self.bpeid2charid = bpeidto2charid
        self.char2id = char2id

    def convert(self, bpe_seq):

        char_seq = list()

        for bpe in bpe_seq:

            char_seq += self.bpeid2charid[bpe]

        return char_seq


class BPEtoPhonemes(BPEConverter):

    def __init__(self):

        return


    def convert(self, bpe_seq):

        pass