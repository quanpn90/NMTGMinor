from __future__ import division

import onmt
import onmt.Markdown
import torch
import argparse
import math
import numpy
import sys

from onmt.metrics.gleu import sentence_gleu

parser = argparse.ArgumentParser(description='generate_distill.py')
onmt.Markdown.add_md_help_argument(parser)

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src',   required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-src_img_dir',   default="",
                    help='Source image directory')
parser.add_argument('-tgt', required=True,
                    help='True target sequence')
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

parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")
parser.add_argument('-print_nbest', action='store_true',
                    help='Output the n-best list instead of a single sentence')
parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")


def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal/wordsTotal)))


def addone(f):
    for line in f:
        yield line
    yield None
    
def len_penalty(s, l, alpha):
    
    l_term = math.pow(l, alpha)
    return s / l_term

def main():
    opt = parser.parse_args()
    opt.alpha = 0.0
    opt.beta = 0.0
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

    srcBatch, tgtBatch = [], []

    count = 0

    tgtF = open(opt.tgt) if opt.tgt else None

    if opt.dump_beam != "":
        import json
        translator.initBeamAccum()
        
        # here we are trying to 
    inFile = None
    if(opt.src == "stdin"):
            inFile = sys.stdin
            opt.batch_size = 1
    else:
      inFile = open(opt.src)
        
    translator = onmt.Translator(opt)
        
    for line in addone(inFile):
        if line is not None:
            srcTokens = line.split()
            srcBatch += [srcTokens]
            if tgtF:
                tgtTokens = tgtF.readline().split() if tgtF else None
                tgtBatch += [tgtTokens]

            if len(srcBatch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(srcBatch) == 0:
                break

        predBatch, predScore, predLength, goldScore, numGoldWords  = translator.translate(srcBatch,
                                                                                    tgtBatch)
                                                              
        predScoreTotal += sum(score[0] for score in predScore)
        predWordsTotal += sum(len(x[0]) for x in predBatch)
        if tgtF is not None:
            goldScoreTotal += sum(goldScore)
            goldWordsTotal += numGoldWords
        
        transScores = list()
        
        for b in range(len(predBatch)):
            
            count += 1 
            
            current_scores = list()
            
            # compute the gleu scores for each sentence in the beam
            for n in range(opt.beam_size):
                
                pred = predBatch[b][n]
                gold = tgtBatch[b]
                
                sent_score = sentence_gleu(gold, pred)[0]
                
                current_scores.append(sent_score)
                
            
            
            # sort the scores descending 
            sidx = numpy.argsort(current_scores)[::-1]
            #~ 
            #~ print(current_scores)
            #~ print(sidx)
            
            # choose the one with highest score
            best_id = sidx[0]
            
            #~ print(best_id)
            
            outF.write(" ".join(predBatch[b][best_id]) + '\n')
            outF.flush()
            

            if opt.verbose:
                srcSent = ' '.join(srcBatch[b])
                if translator.tgt_dict.lower:
                    srcSent = srcSent.lower()
                print('SENT %d: %s' % (count, srcSent))
                print('PRED %d: %s' % (count, " ".join(predBatch[b][best_id])))
                print("PRED SCORE: %.4f" %  predScore[b][best_id])


                tgtSent = ' '.join(tgtBatch[b])
                if translator.tgt_dict.lower:
                    tgtSent = tgtSent.lower()
                print('GOLD %d: %s ' % (count, tgtSent))
                print("GOLD SCORE: %.4f" % goldScore[b])
                
                print('')

        srcBatch, tgtBatch = [], []
        
    if opt.verbose:
        reportScore('PRED', predScoreTotal, predWordsTotal)
        if tgtF: reportScore('GOLD', goldScoreTotal, goldWordsTotal)
                

    if tgtF:
        tgtF.close()

    if opt.dump_beam:
        json.dump(translator.beam_accum, open(opt.dump_beam, 'w'))
    
    


if __name__ == "__main__":
    main()
