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

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-lm', required=False,
                    help='Path to language model .pt file. Used for cold fusion')
parser.add_argument('-autoencoder', required=False,
                    help='Path to autoencoder .pt file')
parser.add_argument('-input_type', default="word",
                    help="Input type: word/char")
parser.add_argument('-src', required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-attributes', default="",
                    help='Attributes for the decoder. Split them by |   ')
parser.add_argument('-stride', type=int, default=1,
                    help="Stride on input features")
parser.add_argument('-concat', type=int, default=1,
                    help="Concate sequential audio features to decrease sequence length")
parser.add_argument('-asr_format', default="h5", required=False,
                    help="Format of asr data h5 or scp")
parser.add_argument('-encoder_type', default='text',
                    help="Type of encoder to use. Options are [text|img|audio].")
parser.add_argument('-previous_context', type=int, default=0,
                    help="Number of previous sentence for context")

parser.add_argument('-tgt',
                    help='True target sequence (optional)')
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size', type=int, default=5,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=2048,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
parser.add_argument('-start_with_bos', action="store_true",
                    help="""Add BOS token to the top of the source sentence""")
# parser.add_argument('-phrase_table',
#                     help="""Path to source-target dictionary to replace UNK
#                     tokens. See README.md for the format of this file.""")
parser.add_argument('-verbose', action="store_true",
                    help='Print scores and predictions for each sentence')
parser.add_argument('-sampling', action="store_true",
                    help='Using multinomial sampling instead of beam search')
parser.add_argument('-dump_beam', type=str, default="",
                    help='File to dump beam information to.')
parser.add_argument('-bos_token', type=str, default="<s>",
                    help='BOS Token (used in multilingual model). Default is <s>.')
parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")
parser.add_argument('-alpha', type=float, default=0.6,
                    help="""Length Penalty coefficient""")
parser.add_argument('-beta', type=float, default=0.0,
                    help="""Coverage penalty coefficient""")
parser.add_argument('-print_nbest', action='store_true',
                    help='Output the n-best list instead of a single sentence')
parser.add_argument('-ensemble_op', default='mean', help="""Ensembling operator""")
parser.add_argument('-normalize', action='store_true',
                    help='To normalize the scores based on output length')
parser.add_argument('-fp16', action='store_true',
                    help='To use floating point 16 in decoding')
parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")


def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / (wordsTotal + 1e-9),
        name, math.exp(-scoreTotal / (wordsTotal + 1e-9))))


def addone(f):
    for line in f:
        yield line
    yield None


def lenPenalty(s, l, alpha):
    l_term = math.pow(l, alpha)
    return s / l_term


def getSentenceFromTokens(tokens, input_type):
    if input_type == 'word':
        sent = " ".join(tokens)
    elif input_type == 'char':
        sent = "".join(tokens)
    else:
        raise NotImplementedError
    return sent


def main():
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    # Always pick n_best
    opt.n_best = opt.beam_size

    if opt.output == "stdout":
        outF = sys.stdout
    else:
        outF = open(opt.output, 'w')

    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0

    srcBatch, tgtBatch, tgtScores = [], [], []

    count = 0

    tgtF = open(opt.tgt) if opt.tgt else None

    if opt.dump_beam != "":
        import json
        translator.initBeamAccum()

        # here we are trying to 
    inFile = None
    if opt.src == "stdin":
        inFile = sys.stdin
        opt.batch_size = 1
    elif opt.encoder_type == "audio" and opt.asr_format == "h5":
        inFile = h5.File(opt.src, 'r')
    elif opt.encoder_type == "audio" and opt.asr_format == "scp":
        import kaldiio
        from kaldiio import ReadHelper
        audio_data = iter(ReadHelper('scp:' + opt.src))
    else:
        inFile = open(opt.src)

    # initialize the rescorer (with models) and stuff
    rescorer = onmt.Rescorer(opt)

    if opt.encoder_type == "audio":
        s_prev_context = []
        t_prev_context = []

        i = 0
        while True:
            if opt.asr_format == "h5":
                if i == len(inFile):
                    break
                line = np.array(inFile[str(i)])
                i += 1
            elif opt.asr_format == "scp":
                try:
                    _, line = next(audio_data)
                except StopIteration:
                    break

            if opt.stride != 1:
                line = line[0::opt.stride]
            line = torch.from_numpy(line)
            if opt.concat != 1:
                add = (opt.concat - line.size()[0] % opt.concat) % opt.concat
                z = torch.FloatTensor(add, line.size()[1]).zero_()
                line = torch.cat((line, z), 0)
                line = line.reshape((line.size()[0] // opt.concat, line.size()[1] * opt.concat))

            if opt.previous_context > 0:
                s_prev_context.append(line)
                for i in range(1, opt.previous_context + 1):
                    if i < len(s_prev_context):
                        line = torch.cat((torch.cat((s_prev_context[-i - 1], torch.zeros(1, line.size()[1]))), line))
                if len(s_prev_context) > opt.previous_context:
                    s_prev_context = s_prev_context[-1 * opt.previous_context:]
            srcBatch += [line]

            if tgtF:
                # ~ tgt_tokens = tgtF.readline().split() if tgtF else None
                tline = tgtF.readline().strip()

                twords = tline.split("|||")[0].strip()

                if opt.input_type == 'word':
                    tgt_tokens = tline.split() if tgtF else None
                elif opt.input_type == 'char':
                    tgt_tokens = list(tline.strip()) if tgtF else None
                else:
                    raise NotImplementedError("Input type unknown")

                tgtBatch += [tgt_tokens]

            if len(srcBatch) < opt.batch_size:
                continue

            print("Batch size:", len(srcBatch), len(tgtBatch))
            goldScore, numGoldWords, allGoldScores = rescorer.rescore_asr(
                srcBatch, tgtBatch)

            print("Result:", len(predBatch))
            count = translateBatch(opt, tgtF, count, outF, translator,
                                                   srcBatch, tgtBatch, goldScore, numGoldWords,
                                                   allGoldScores, opt.input_type)
            srcBatch, tgtBatch, tgtScores = [], []

        if len(srcBatch) != 0:
            print("Batch size:", len(srcBatch), len(tgtBatch))
            goldScore, numGoldWords, allGoldScores = translator.rescore_asr(srcBatch, tgtBatch)
            print("Result:", len(predBatch))
            count = translateBatch(opt, tgtF, count, outF, srcBatch, tgtBatch, tgtScores,
                                                                               goldScore, numGoldWords,
                                                                               allGoldScores, opt.input_type)
            srcBatch, tgtBatch, tgtScores = [], []

    else:
        for line in addone(inFile):
            if line is not None:
                if opt.input_type == 'word':
                    srcTokens = line.split()
                elif opt.input_type == 'char':
                    srcTokens = list(line.strip())
                else:
                    raise NotImplementedError("Input type unknown")

                # for each source sentence, we read in n target
                for n in range(opt.n_best):
                    # duplicate the srcTokens
                    srcBatch += [srcTokens]

                    tgtline = tgtF.readline()
                    tgt_text = tgtline.strip().split(' ||| ')[0]
                    tgt_score = tgtline.strip().split(' ||| ')[1]

                    if opt.input_type == 'word':
                        tgt_tokens = tgt_text.split() if tgtF else None
                    elif opt.input_type == 'char':
                        tgt_tokens = list(tgt_text.strip()) if tgtF else None
                    else:
                        raise NotImplementedError("Input type unknown")
                    tgtBatch += [tgt_tokens]
                    tgtScores += [tgt_score]

                    if len(srcBatch) < opt.batch_size * opt.n_best:
                        continue
            else:
                # at the end of file, check last batch
                if len(srcBatch) == 0:
                    break

            goldScore, numGoldWords, allGoldScores = rescorer.rescore(srcBatch, tgtBatch)

            # convert output tensor to words
            count = translateBatch(opt, tgtF, count, outF, srcBatch, tgtBatch, tgtScores,
                                                                       goldScore, numGoldWords,
                                                                       allGoldScores, opt.input_type)
            srcBatch, tgtBatch = [], []

    if tgtF:
        tgtF.close()

def translateBatch(opt, tgtF, count, outF, srcBatch, tgtBatch, tgtScores, goldScore,
                   numGoldWords, allGoldScores, input_type):

    for b in range(len(tgtBatch)):


        # if not opt.print_nbest:
        #     outF.write(getSentenceFromTokens(predBatch[b][0], input_type) + '\n')
        #     outF.flush()
        # else:
        #     for n in range(opt.n_best):
        #         idx = n
        #         output_sent = getSentenceFromTokens(predBatch[b][idx], input_type)
        #         out_str = "%s ||| %.4f" % (output_sent, predScore[b][idx])
        #
        #         print(out_str)
        #         outF.write(out_str + 'n')
        #         outF.flush()

        tgtSent = getSentenceFromTokens(tgtBatch[b], input_type)
        gold_score = goldScore[b]
        prev_score = tgtScores[b]  # string
        outstr = "%s ||| %s %.4f" % (tgtSent, prev_score, gold_score)
        outF.write(outstr + '\n')
        outF.flush()

        if opt.verbose:
            if count % opt.beam_size == 0:
                srcSent = getSentenceFromTokens(srcBatch[b], input_type)
                print('SRC SENT %d: %s ' % (count // opt.beam_size + 1, srcSent))
                print('')
            print(outstr)
            # if tgtF is not None:
            #     tgtSent = getSentenceFromTokens(tgtBatch[b], input_type)
            #     print('GOLD %d: %s ' % (count, tgtSent))
            #     print("GOLD SCORE: %.4f" % goldScore[b])
            #     # print("Single GOLD Scores:",end=" ")
            #     # for j in range(len(tgtBatch[b])):
            #     #     print(allGoldScores[j][b].item(),end =" ")
            #     print ()
            # if opt.print_nbest:
            #     print('\n BEST HYP:')
            #     for n in range(opt.n_best):
            #         idx = n
            #         out_str = "%s ||| %.4f" % (" ".join(predBatch[b][idx]), predScore[b][idx])
            #         print(out_str)
            print('')

        count += 1

    return count


if __name__ == "__main__":
    main()
