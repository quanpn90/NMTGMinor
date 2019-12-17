import onmt


def split_line_by_char(line, word_list=["<unk>"]):
    chars = list()

    words = line.strip().split()

    for i, word in enumerate(words):
        if word in word_list:
            chars.append(word)
        else:
            for c in word:
                chars.append(c)

        if i < (len(words) - 1):
            chars.append(' ')

    return chars


class Tokenizer(object):

    def __init__(self, input_type='word', lower=False):
        self.input_type = input_type
        self.lower = lower

    def tokenize(self, sentence):
        if self.input_type == "word":
            tokens = sentence.strip().split()
        elif self.input_type == "char":
            tokens = split_line_by_char(sentence)
        else:
            raise NotImplementedError("Input type not implemented")

        return tokens
