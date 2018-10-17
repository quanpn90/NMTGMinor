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

parser = argparse.ArgumentParser(description='translate.py')
onmt.Markdown.add_md_help_argument(parser)

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-input_type', default="word",
                    help="Input type: word/char")
parser.add_argument('-src',   required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-src_img_dir',   default="",
                    help='Source image directory')
parser.add_argument('-stride', type=int, default=1,
                    help="Stride on input features")
parser.add_argument('-concat', type=int, default=1,
                    help="Concate sequential audio features to decrease sequence length")
parser.add_argument('-encoder_type', default='text',
                    help="Type of encoder to use. Options are [text|img|audio].")
parser.add_argument('-previous_context', type=int, default=0,
                    help="Number of previous sentence for context")

parser.add_argument('-tgt',
                    help='True target sequence (optional)')
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size',  type=int, default=5,
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
parser.add_argument('-dump_beam', type=str, default="",
                    help='File to dump beam information to.')

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
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal/wordsTotal)))


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

    srcBatch, tgtBatch = [], []

    count = 0

    tgtF = open(opt.tgt) if opt.tgt else None

    if opt.dump_beam != "":
        import json
        translator.initBeamAccum()
        
        # here we are trying to 
    inFile = None

    translator = onmt.EnsembleTranslator(opt)

    if(opt.src == "stdin"):
            inFile = sys.stdin
            opt.batch_size = 1
    elif(opt.encoder_type == "audio"):
        inFile = h5.File(opt.src,'r')
    else:
      inFile = open(opt.src)

    if(opt.encoder_type == "audio"):

        s_prev_context = []
        t_prev_context = []


        for i in range(len(inFile)):
            if(opt.stride == 1):
                line = torch.from_numpy(np.array(inFile[str(i)]))
            else:
                line = torch.from_numpy(np.array(inFile[str(i)])[0::opt.stride])
            if(opt.concat != 1):
                add = (opt.concat-line.size()[0]%opt.concat)%opt.concat
                z= torch.FloatTensor(add, line.size()[1]).zero_()
                line = torch.cat((line,z),0)
                line = line.reshape((line.size()[0]/opt.concat,line.size()[1]*opt.concat))

            #~ srcTokens = line.split()
            if(opt.previous_context > 0):
                s_prev_context.append(line)
                for i in range(1,opt.previous_context+1):
                    if(i < len(s_prev_context)):
                        line = torch.cat((torch.cat((s_prev_context[-i-1],torch.zeros(1,line.size()[1]))),line))
                if(len(s_prev_context) > opt.previous_context):
                    s_prev_context = s_prev_context[-1*opt.previous_context:]
            srcBatch += [line]

            if tgtF:
                #~ tgtTokens = tgtF.readline().split() if tgtF else None
                tline = tgtF.readline().strip()
                if(opt.previous_context > 0):
                    t_prev_context.append(tline)
                    for i in range(1,opt.previous_context+1):
                        if(i < len(s_prev_context)):
                            tline = t_prev_context[-i-1]+" # "+tline
                    if(len(t_prev_context) > opt.previous_context):
                        t_prev_context = t_prev_context[-1*opt.previous_context:]


                if opt.input_type == 'word':
                    tgtTokens = tline.split() if tgtF else None
                elif opt.input_type == 'char':
                    tgtTokens = list(tline.strip()) if tgtF else None
                else:
                    raise NotImplementedError("Input type unknown")

                tgtBatch += [tgtTokens]

            if len(srcBatch) < opt.batch_size:
                continue

            print("Batch size:",len(srcBatch),len(tgtBatch))
            predBatch, predScore, predLength, goldScore, numGoldWords  = translator.translateASR(srcBatch,
                                                                                    tgtBatch)
            print("Result:",len(predBatch))
            count,predScore,predWords,goldScore,goldWords = translateBatch(opt,tgtF,count,outF,translator,srcBatch,tgtBatch,predBatch, predScore, predLength, goldScore, numGoldWords,opt.input_type)
            predScoreTotal += predScore
            predWordsTotal += predWords
            goldScoreTotal += goldScore
            goldWordsTotal += goldWords
            srcBatch, tgtBatch = [], []

        if len(srcBatch) != 0:
            print("Batch size:",len(srcBatch),len(tgtBatch))
            predBatch, predScore, predLength, goldScore, numGoldWords  = translator.translateASR(srcBatch,
                                                                                    tgtBatch)
            print("Result:",len(predBatch))
            count,predScore,predWords,goldScore,goldWords = translateBatch(opt,tgtF,count,outF,translator,srcBatch,tgtBatch,predBatch, predScore, predLength, goldScore, numGoldWords,opt.input_type)
            predScoreTotal += predScore
            predWordsTotal += predWords
            goldScoreTotal += goldScore
            goldWordsTotal += goldWords
            srcBatch, tgtBatch = [], []
            


    else:
        
        for line in addone(inFile):
            if line is not None:
                if opt.input_type == 'word':
                    srcTokens = line.split()
                elif opt.input_type == 'char':
                    srcTokens = list(line.strip())
                else:
                    raise NotImplementedError("Input type unknown")
                srcBatch += [srcTokens]
                if tgtF:
                    #~ tgtTokens = tgtF.readline().split() if tgtF else None
                    if opt.input_type == 'word':
                        tgtTokens = tgtF.readline().split() if tgtF else None
                    elif opt.input_type == 'char':
                        tgtTokens = list(tgtF.readline().strip()) if tgtF else None
                    else:
                        raise NotImplementedError("Input type unknown")
                    tgtBatch += [tgtTokens]

                if len(srcBatch) < opt.batch_size:
                    continue
            else:
                # at the end of file, check last batch
                if len(srcBatch) == 0:
                    break
                    
            
            predBatch, predScore, predLength, goldScore, numGoldWords  = translator.translate(srcBatch,
                                                                                    tgtBatch)

            count,predScore,predWords,goldScore,goldWords = translateBatch(opt,tgtF,count,outF,translator,srcBatch,tgtBatch,predBatch, predScore, predLength, goldScore, numGoldWords,opt.input_type)
            predScoreTotal += predScore
            predWordsTotal += predWords
            goldScoreTotal += goldScore
            goldWordsTotal += goldWords
            srcBatch, tgtBatch = [], []

    if opt.verbose:
        reportScore('PRED', predScoreTotal, predWordsTotal)
        if tgtF: reportScore('GOLD', goldScoreTotal, goldWordsTotal)


    if tgtF:
        tgtF.close()

    if opt.dump_beam:
        json.dump(translator.beam_accum, open(opt.dump_beam, 'w'))

def translateBatch(opt,tgtF,count,outF,translator,srcBatch,tgtBatch,predBatch, predScore, predLength, goldScore, numGoldWords,input_type):
    if opt.normalize:
        predBatch_ = []
        predScore_ = []
        for bb, ss, ll in zip(predBatch, predScore, predLength):
            #~ ss_ = [s_/numpy.maximum(1.,len(b_)) for b_,s_,l_ in zip(bb,ss,ll)]
            length = [len(i) for i in [''.join(b_)  for b_ in bb]]
            ss_ = [lenPenalty(s_, max(l_,1), opt.alpha) for b_,s_,l_ in zip(bb,ss,length)]
            ss_origin = [(s_, len(b_)) for b_,s_,l_ in zip(bb,ss,ll)]
            sidx = numpy.argsort(ss_)[::-1]
            #~ print(ss_, sidx, ss_origin)
            predBatch_.append([bb[s] for s in sidx])
            predScore_.append([ss_[s] for s in sidx])
        predBatch = predBatch_
        predScore = predScore_
                                                              
    predScoreTotal = sum(score[0].item() for score in predScore)
    predWordsTotal = sum(len(x[0]) for x in predBatch)
    goldScoreTotal = 0
    goldWordsTotal = 0
    if tgtF is not None:
        goldScoreTotal = sum(goldScore).item()
        goldWordsTotal = numGoldWords
            
    for b in range(len(predBatch)):
                        
        count += 1

        if not opt.print_nbest:
            #~ print(predBatch[b][0])
            outF.write(getSentenceFromTokens(predBatch[b][0], input_type) + '\n')
            outF.flush()

        if opt.verbose:
            #~ srcSent = ' '.join(srcBatch[b])
            #srcSent = ''.join(srcBatch[b])
            #if translator.tgt_dict.lower:
            #    srcSent = srcSent.lower()
            #print('SENT %d: %s' % (count, srcSent))
            print('PRED %d: %s' % (count, getSentenceFromTokens(predBatch[b][0], input_type)))
            print("PRED SCORE: %.4f" %  predScore[b][0])

            if tgtF is not None:
                tgtSent = getSentenceFromTokens( tgtBatch[b], input_type)
                if translator.tgt_dict.lower:
                    tgtSent = tgtSent.lower()
                print('GOLD %d: %s ' % (count, tgtSent))
                print("GOLD SCORE: %.4f" % goldScore[b])

            if opt.print_nbest:
                print('\nBEST HYP:')
                for n in range(opt.n_best):
                    idx = n
                    print("[%.4f] %s" % (predScore[b][idx],
                                         " ".join(predBatch[b][idx])))

            print('')

    return count,predScoreTotal,predWordsTotal,goldScoreTotal,goldWordsTotal
    
    


if __name__ == "__main__":
    main()

