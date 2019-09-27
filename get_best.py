#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import onmt
import onmt.Markdown
import torch
import argparse
import math
import numpy
import sys
import h5py as h5
import numpy as np
import apex

parser = argparse.ArgumentParser(description='rescore.py')
onmt.Markdown.add_md_help_argument(parser)
#
parser.add_argument('-input', required=True,
                    help='Path to the nbest file')
parser.add_argument('-n_best', type=int, default=1,
                    help="""n_best value from decoding and rescoring""")

parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-coeff', default=[], nargs='+', type=float,
                        help="Use CUDA on the listed devices.")


def addone(f):
    for line in f:
        yield line
    yield None


def main():
    opt = parser.parse_args()
    reader = open(opt.input)
    out_writer = open(opt.output, 'w')

    count = 0
    all_sents, all_scores = [], []

    for line in addone(reader):

        if line is not None:
            count += 1
            parts = line.strip().split(" ||| ")
            text = parts[0]
            scores = parts[1].strip().split()

            # print(scores)
            # print(len(scores))
            # assert(len(scores) == len(opt.coeff))
            all_sents.append(text)

            score = 0
            print(count)
            for i, score_ in enumerate(scores):
                score += opt.coeff[i] * float(score_)

            all_scores.append(score)

            if count % opt.n_best == 0:
                all = zip(all_sents, all_scores)

                sorted_all = sorted(all,  key=lambda x: x[1], reverse=True)

                best_sent = sorted_all[0][0]
                out_writer.write(best_sent + "\n")
                all_sents = []
                all_scores = []
        else:
            break

    out_writer.close()


if __name__ == "__main__":
    main()