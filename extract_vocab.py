import torch

import onmt
import onmt.markdown
import torch
import argparse
import math
import numpy
import sys
import numpy as np


parser = argparse.ArgumentParser(description='translate.py')
onmt.markdown.add_md_help_argument(parser)

parser.add_argument('-dict', required=True,
                    help='Path to dict.pt file')

parser.add_argument('-text', required=True,
                    help='text files (separated by |)')

parser.add_argument('-external_tokenizer', default="",
                    help="External tokenizer from Huggingface. Currently supports barts.")

parser.add_argument('-lang', default="",
                    help="External tokenizer from Huggingface. Currently supports barts.")

parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")


def addone(f):
    for line in f:
        yield line
    yield None


def main():
    opt = parser.parse_args()
    dictionary = torch.load(opt.dict)

    data_files = opt.text.split("|")

    if "mbart-large-50" in opt.external_tokenizer.lower():
        print("[INFO] Using the external MBART50 tokenizer...")

        from transformers import MBart50TokenizerFast
        external_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang=opt.lang)
    elif "bart" in opt.external_tokenizer.lower():
        print("[INFO] Using the external BART tokenizer...")

        from transformers import BartTokenizer
        external_tokenizer = BartTokenizer.from_pretrained(opt.external_tokenizer)

    elif "m2m100" in opt.external_tokenizer.lower():
        print("[INFO] Using the external %s tokenizer..." % opt.external_tokenizer)
        from transformers import M2M100Tokenizer
        external_tokenizer = M2M100Tokenizer.from_pretrained(opt.external_tokenizer, src_lang=opt.lang)

    elif opt.external_tokenizer is None or len(opt.external_tokenizer) == 0:
        external_tokenizer = None
    else:
        raise NotImplementedError

    vocab_ids = dict()

    for data_file in data_files:

        reader = open(data_file)
        print("Loading from data %s ..." % data_file)

        for line in addone(reader):

            if line is not None:

                token_ids = list()

                if external_tokenizer is not None:
                    token_ids = external_tokenizer(line)['input_ids']
                else:
                    if opt.input_type == 'word':
                        tokens = line.split()
                    elif opt.input_type == 'char':
                        tokens = list(line.strip())
                    else:
                        raise NotImplementedError("Input type unknown")

                    for token in tokens:
                        token_ids.append(dictionary.lookup(token, default=dictionary.lookup("<unk>")))

                for id in token_ids:
                    if id not in vocab_ids:
                        vocab_ids[id] = 1
                    else:
                        vocab_ids[id] += 1

    vocab_ids = dict(sorted(vocab_ids.items(), key=lambda item: item[1]))

    print(vocab_ids)

    print('saving .... to %s' % opt.output)
    torch.save(vocab_ids, opt.output)
    print("Done")

if __name__ == "__main__":
    main()