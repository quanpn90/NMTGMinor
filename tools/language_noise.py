#!/usr/bin/env python3

"""
Randomly replaces words in the input file with similar (by edit distance) words from another language.
Apply before BPE
Requires editdistance (pip install editdistance)
"""
import argparse
import functools
import multiprocessing
import random
import sys

import editdistance
from tqdm import tqdm

from nmtg.data import Dictionary

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('other_language_dict')
parser.add_argument('output', default='-', nargs='?')
parser.add_argument('-threshold', type=int, default=0)
parser.add_argument('-prob', type=float, default=0.1)
parser.add_argument('-num_variants', type=int, default=1)

args = parser.parse_args()


with open(args.input) as f:
    main_dictionary = Dictionary.infer_from_text(f)
main_symbols = main_dictionary.symbols[main_dictionary.nspecial:]
del main_dictionary

dictionary = Dictionary.load(args.other_language_dict)
if args.threshold != 0:
    dictionary.finalize(threshold=args.threshold)
symbols = dictionary.symbols[dictionary.nspecial:]
del dictionary


def get_nearest(pool, symbol):
    return symbol, min(pool, key=lambda x: editdistance.eval(x, symbol))


partial = functools.partial(get_nearest, symbols)

nearest = {}
with multiprocessing.Pool(80) as pool:
    for symbol, closest in tqdm(pool.imap_unordered(partial, main_symbols, 50), total=len(main_symbols)):
        nearest[symbol] = closest

out_file = open(args.output, 'w') if args.output != '-' else sys.stdout
try:
    with open(args.input) as in_file:
        for line in tqdm(in_file):
            words = line.rstrip().split()

            for i in range(args.num_variants):
                new_words = list(words)
                for j in range(len(words)):
                    word = words[j]
                    if random.random() < args.prob:
                        new_words[j] = nearest.get(word, word)
                out_file.write(' '.join(new_words) + '\n')
finally:
    if args.output != '-':
        out_file.close()
