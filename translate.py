from __future__ import division

import onmt
import onmt.Markdown
import torch
import argparse
import math
import numpy
import sys

parser = argparse.ArgumentParser(description='translate.py')
onmt.Markdown.add_md_help_argument(parser)

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-version', type=float, default=1.0,
                    help="""Decoder version. The 2.0 one uses diverse decoding""")                    
parser.add_argument('-diverse_beam_strength', type=float, default=0.5,
                    help="""Diverse beam strength in decoding""")                    
parser.add_argument('-input_type', default="word",
                    help="Input type: word/char")                    
parser.add_argument('-src',   required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-tgt',
                    help='True target sequence (optional)')
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size',  type=int, default=5,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=512,
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
parser.add_argument('-dump_beam', type=str, default="",
                    help='File to dump beam information to.')
parser.add_argument('-bos_token', type=str, default="<s>",
                    help='BOS Token to start BEAM search. Default is <s>.')

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


def reportScore(name, score_total, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / wordsTotal,
        name, math.exp(-score_total/wordsTotal)))


def addone(f):
    for line in f:
        yield line
    yield None

    
def len_penalty(s, l, alpha):
    
    # ~ l_term = math.pow(l, alpha)
    
    l_term = math.pow(l + 5.0, alpha) / math.pow(6, alpha)
    return s / l_term


def get_sent_from_tokens(tokens, input_type):
    
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

    pred_score_total, pred_words_total, gold_scores_total, gold_words_total = 0, 0, 0, 0

    src_batch, tgt_batch = [], []

    count = 0

    tgtF = open(opt.tgt) if opt.tgt else None

    # here we are trying to check if online decoding is used
    in_file = None
    if(opt.src == "stdin"):
        in_file = sys.stdin
        opt.batch_size = 1
    else:
        in_file = open(opt.src)

    translator = onmt.EnsembleTranslator(opt)

    if opt.dump_beam != "":
        import json
        translator.initBeamAccum()
        
    for line in addone(in_file):
        if line is not None:

            if opt.input_type == 'word':
                src_tokens = line.split()
                src_batch += [src_tokens]
                if tgtF:
                    tgt_tokens = tgtF.readline().split() if tgtF else None
                    tgt_batch += [tgt_tokens]

            elif opt.input_type == 'char':
                src_tokens = list(line.strip())
                src_batch += [src_tokens]
                if tgtF:
                    # tgt_tokens = tgtF.readline().split() if tgtF else None
                    tgt_tokens = list(tgtF.readline().strip()) if tgtF else None
                    tgt_batch += [tgt_tokens]
            else:
                raise NotImplementedError("Input type unknown")

            if len(src_batch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(src_batch) == 0:
                break

        pred_batch, pred_score, pred_length, gold_scores, num_gold_words  = translator.translate(src_batch,
                                                                                    tgt_batch)
        if opt.normalize and opt.version == 1.0:
            pred_batch_ = []
            pred_score_ = []
            for bb, ss, ll in zip(pred_batch, pred_score, pred_length):
                # ss_ = [s_/numpy.maximum(1.,len(b_)) for b_,s_,l_ in zip(bb,ss,ll)]
                ss_ = [len_penalty(s_, l_, opt.alpha) for b_,s_,l_ in zip(bb,ss,ll)]
                # ss_origin = [(s_, len(b_)) for b_,s_,l_ in zip(bb,ss,ll)]
                sidx = numpy.argsort(ss_)[::-1]
                # print(ss_, sidx, ss_origin)
                pred_batch_.append([bb[s] for s in sidx])
                pred_score_.append([ss_[s] for s in sidx])
            pred_batch = pred_batch_
            pred_score = pred_score_    
                                                              
        pred_score_total += sum(score[0] for score in pred_score)
        pred_words_total += sum(len(x[0]) for x in pred_batch)
        if tgtF is not None:
            gold_scores_total += sum(gold_scores).item()
            gold_words_total += num_gold_words
            
        for b in range(len(pred_batch)):
                        
            count += 1
            
            best_hyp =  get_sent_from_tokens(pred_batch[b][0], opt.input_type)            
            if not opt.print_nbest:
                # print(pred_batch[b][0])
                outF.write(best_hyp + '\n')
                outF.flush()

            if opt.verbose:
                src_sent = get_sent_from_tokens(src_batch[b], opt.input_type)
                if translator.tgt_dict.lower:
                    src_sent = src_sent.lower()
                print('SENT %d: %s' % (count, src_sent))
                print('PRED %d: %s' % (count, best_hyp))
                print("PRED SCORE: %.4f" %  pred_score[b][0])

                if tgtF is not None:
                    tgtSent = get_sent_from_tokens(tgt_batch[b], opt.input_type)
                    if translator.tgt_dict.lower:
                        tgtSent = tgtSent.lower()
                    print('GOLD %d: %s ' % (count, tgtSent))
                    print("GOLD SCORE: %.4f" % gold_scores[b])

                if opt.print_nbest:
                    print('\nBEST HYP:')
                    for n in range(opt.n_best):
                        idx = n
                        sent = get_sent_from_tokens(pred_batch[b][idx], opt.input_type)
                        print("[%.4f] %s" % (pred_score[b][idx], sent))
                            
                print('')

        src_batch, tgt_batch = [], []
        
    if opt.verbose:
        reportScore('PRED', pred_score_total, pred_words_total)
        if tgtF: reportScore('GOLD', gold_scores_total, gold_words_total)
                

    if tgtF:
        tgtF.close()

    if opt.dump_beam:
        json.dump(translator.beam_accum, open(opt.dump_beam, 'w'))
    
    


if __name__ == "__main__":
    main()

