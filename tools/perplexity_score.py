#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import onmt
import onmt.markdown
import torch
import argparse
import math
import numpy
import sys
import h5py as h5
import numpy as np
import apex

parser = argparse.ArgumentParser(description='translate.py')
onmt.markdown.add_md_help_argument(parser)

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
parser.add_argument('-src_lang', default='src',
                    help='Source language')
parser.add_argument('-tgt_lang', default='tgt',
                    help='Target language')
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
parser.add_argument('-max_sent_length', type=int, default=256,
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
parser.add_argument('-no_bos_gold', action="store_true",
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
parser.add_argument('-fast_translate', action='store_true',
                    help='Using the fast decoder')

def reportScore(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / (words_total + 1e-9),
        name, math.exp(-score_total / (words_total + 1e-9))))
    

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

    all_scores = torch.empty(0)

    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    # Always pick n_best
    opt.n_best = opt.beam_size

    if opt.output == "stdout":
        outF = sys.stdout
    else:
        outF = open(opt.output, 'w')

    gold_score_total, gold_words_total = 0, 0,
    src_batch, tgt_batch = [], []
    cur_batch_sizes = []

    count = 0

    tgtF = open(opt.tgt) if opt.tgt else None

    in_file = None

    if opt.src == "stdin":
        in_file = sys.stdin
        opt.batch_size = 1
    elif opt.encoder_type == "audio" and opt.asr_format  == "h5":
        in_file = h5.File(opt.src, 'r')
    elif opt.encoder_type == "audio" and opt.asr_format == "scp":
        import kaldiio
        from kaldiio import ReadHelper
        audio_data = iter(ReadHelper('scp:' + opt.src))
    else:
        in_file = open(opt.src)

    from onmt.inference.perplexity_scorer import PerplexityScorer
    translator = PerplexityScorer(opt)

    # Audio processing for the source batch
    if opt.encoder_type == "audio":

        s_prev_context = []
        t_prev_context = []

        i = 0
        while True:
            if opt.asr_format == "h5":
                if i == len(in_file):
                    break
                line = np.array(in_file[str(i)])
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
            src_batch += [line]

            if tgtF:
                tline = tgtF.readline().strip()
                if opt.previous_context > 0:
                    t_prev_context.append(tline)
                    for i in range(1, opt.previous_context + 1):
                        if i < len(s_prev_context):
                            tline = t_prev_context[-i - 1] + " # " + tline
                    if len(t_prev_context) > opt.previous_context:
                        t_prev_context = t_prev_context[-1 * opt.previous_context:]

                if opt.input_type == 'word':
                    tgt_tokens = tline.split() if tgtF else None
                elif opt.input_type == 'char':
                    tgt_tokens = list(tline.strip()) if tgtF else None
                else:
                    raise NotImplementedError("Input type unknown")

                tgt_batch += [tgt_tokens]

            if len(src_batch) < opt.batch_size:
                continue

            print("Batch size:", len(src_batch), len(tgt_batch))
            gold_score, num_gold_words, all_gold_scores = translator.translate(src_batch, tgt_batch, type='asr')


            count, gold_score, goldWords, all_scores = translateBatch(opt, tgtF, count, outF, translator,
                                                          src_batch, tgt_batch,
                                                          gold_score, num_gold_words,
                                                          all_gold_scores, all_scores,
                                                          opt.input_type)

            gold_score_total += gold_score
            gold_words_total += goldWords
            src_batch, tgt_batch = [], []

        # catch the last batch
        if len(src_batch) != 0:
            print("Batch size:", len(src_batch), len(tgt_batch))
            gold_score, num_gold_words, all_gold_scores = translator.translate(
                src_batch,
                tgt_batch, type='asr')
            count, gold_score, goldWords, all_scores = translateBatch(opt, tgtF, count, outF, translator,
                                                                   src_batch, tgt_batch,
                                                                   gold_score, num_gold_words,
                                                                   all_gold_scores, all_scores, opt.input_type)
            gold_score_total += gold_score
            gold_words_total += goldWords
            src_batch, tgt_batch = [], []
    # Text processing
    else:
        for line in addone(in_file):
            if line is not None:
                if opt.input_type == 'word':
                    src_tokens = line.split()
                elif opt.input_type == 'char':
                    src_tokens = list(line.strip())
                else:
                    raise NotImplementedError("Input type unknown")
                src_batch += [src_tokens]
                if tgtF:
                    # ~ tgt_tokens = tgtF.readline().split() if tgtF else None
                    if opt.input_type == 'word':
                        tgt_tokens = tgtF.readline().split() if tgtF else None
                    elif opt.input_type == 'char':
                        tgt_tokens = list(tgtF.readline().strip()) if tgtF else None
                    else:
                        raise NotImplementedError("Input type unknown")
                    tgt_batch += [tgt_tokens]

                # cur_batch_sizes.append(max([len(src_tokens),len(tgt_tokens)]))
                # if len(src_batch) == 0 or (max(cur_batch_sizes) * len(src_batch)) < opt.batch_size:
                if len(src_batch) < opt.batch_size:
                    continue
            else:
                # at the end of file, check last batch
                if len(src_batch) == 0:
                    break

            # actually done beam search from the model
            gold_score, num_gold_words, all_gold_scores = translator.translate(src_batch, tgt_batch)

            # convert output tensor to words
            count, gold_score, goldWords, all_scores = translateBatch(opt, tgtF, count, outF, translator,
                                                                               src_batch, tgt_batch,
                                                                               gold_score, num_gold_words,
                                                                               all_gold_scores, all_scores,
                                                                               opt.input_type)
            gold_score_total += gold_score
            gold_words_total += goldWords
            src_batch, tgt_batch = [], []
            cur_batch_sizes = []

    if opt.verbose:
        if tgtF: reportScore('GOLD', gold_score_total, gold_words_total)

    if tgtF:
        tgtF.close()

    if opt.dump_beam:
        json.dump(translator.beam_accum, open(opt.dump_beam, 'w'))

    outF.close()

    print(all_scores.size())
    all_scores_numpy = all_scores.numpy()
    np.savetxt(opt.output, all_scores_numpy, delimiter="\n")

def translateBatch(opt, tgtF, count, outF, translator, src_batch, tgt_batch, gold_score,
                   num_gold_words, all_gold_scores, all_scores, input_type):

    gold_score_total = 0
    gold_words_total = 0
    if tgtF is not None:
        gold_score_total = sum(gold_score).item()
        gold_words_total = num_gold_words

    batch_size = len(src_batch)
    scores = torch.Tensor(batch_size).zero_()

    for b in range(len(src_batch)):

        count += 1

        if opt.normalize:
            gold_score_ = gold_score[b] / len(tgt_batch[b])
        else:
            gold_score_ = gold_score[b]

        if opt.verbose:
            if opt.encoder_type == "text":
                src_sent = " ".join(src_batch[b])
                print('SRC %d: %s' % (count, src_sent))

            if tgtF is not None:
                tgt_sent = getSentenceFromTokens(tgt_batch[b], input_type)
                if translator.tgt_dict.lower:
                    tgt_sent = tgt_sent.lower()
                print('GOLD %d: %s ' % (count, tgt_sent))

                print("GOLD SCORE: %.4f" % gold_score_)
                print()

            print('')

        else:
            if count % 100000 == 0:
                print("Finished %d sentences ... " % count)

        scores[b].fill_(gold_score_)

    all_scores = torch.cat([all_scores, scores], dim=0)

    return count, gold_score_total, gold_words_total, all_scores


if __name__ == "__main__":
    main()
