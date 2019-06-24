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
from ae.Evaluator import Evaluator

parser = argparse.ArgumentParser(description='translate.py')
onmt.Markdown.add_md_help_argument(parser)

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-autoencoder', required=True,
                    help='Path to model .pt file')
parser.add_argument('-input_type', default="word",
                    help="Input type: word/char")
parser.add_argument('-src', required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-src_img_dir', default="",
                    help='Source image directory')
parser.add_argument('-stride', type=int, default=1,
                    help="Stride on input features")
parser.add_argument('-concat', type=int, default=1,
                    help="Concate sequential audio features to decrease sequence length")
parser.add_argument('-encoder_type', default='text',
                    help="Type of encoder to use. Options are [text|img|audio].")

parser.add_argument('-tgt',
                    help='True target sequence (optional)')
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
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
parser.add_argument('-fp16', action='store_true',
                    help='To use floating point 16 in decoding')
parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")
parser.add_argument('-representation', type=str, default="EncoderHiddenState",
                    help="Representation for Autoencoder")
parser.add_argument('-auto_encoder_hidden_size', type=int, default=100,
                    help="Hidden size of autoencoder")
parser.add_argument('-auto_encoder_drop_out', type=float, default=0,
                    help="Use drop_out in autoencoder")


def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal / wordsTotal)))


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


    if opt.output == "stdout":
        outF = sys.stdout
    else:
        outF = open(opt.output, 'w')


    srcBatch, tgtBatch = [], []

    count = 0

    tgtF = open(opt.tgt) if opt.tgt else None


    evaluator = Evaluator(opt)

    if (opt.src == "stdin"):
        inFile = sys.stdin
        opt.batch_size = 1
    elif (opt.encoder_type == "audio"):
        inFile = h5.File(opt.src, 'r')
    else:
        inFile = open(opt.src)

    if (opt.encoder_type == "audio"):
        for i in range(len(inFile)):
            if (opt.stride == 1):
                line = torch.from_numpy(np.array(inFile[str(i)]))
            else:
                line = torch.from_numpy(np.array(inFile[str(i)])[0::opt.stride])
            if (opt.concat != 1):
                add = (opt.concat - line.size()[0] % opt.concat) % opt.concat
                z = torch.FloatTensor(add, line.size()[1]).zero_()
                line = torch.cat((line, z), 0)
                line = line.reshape((line.size()[0] / opt.concat, line.size()[1] * opt.concat))

            if line is not None:
                # ~ srcTokens = line.split()
                srcBatch += [line]
                if tgtF:
                    # ~ tgtTokens = tgtF.readline().split() if tgtF else None
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

            r = evaluator.evalASR(srcBatch,tgtBatch)
            if(opt.representation == "EncoderHiddenState"):
                outputResults(srcBatch,r,outF)
            elif(opt.representation == "DecoderHiddenState" or opt.representation == "Probabilities"):
                for i in range(len(tgtBatch)):
                    tgtBatch[i].append("EOS");
                outputResults(tgtBatch,r,outF)
            elif(opt.representation == "EncoderDecoderHiddenState"):
                for i in range(len(tgtBatch)):
                    tgtBatch[i].append("EOS");
                outputAlignment(srcBatch,tgtBatch,r,outF)
            srcBatch, tgtBatch = [], []
        if len(srcBatch) != 0:
            r = evaluator.evalASR(srcBatch,tgtBatch)
            if(opt.representation == "EncoderHiddenState"):
                outputResults(srcBatch,r,outF)
            elif(opt.representation == "DecoderHiddenState" or opt.representation == "Probabilities"):
                for i in range(len(tgtBatch)):
                    tgtBatch[i].append("EOS");
                outputResults(tgtBatch,r,outF)
            elif(opt.representation == "EncoderDecoderHiddenState"):
                for i in range(len(tgtBatch)):
                    tgtBatch[i].append("EOS");
                outputAlignment(srcBatch,tgtBatch,r,outF)

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
                    # ~ tgtTokens = tgtF.readline().split() if tgtF else None
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

            r = evaluator.eval(srcBatch,tgtBatch)
            if(opt.representation == "EncoderHiddenState"):
                outputResults(srcBatch,r,outF)
            elif(opt.representation == "DecoderHiddenState" or opt.representation == "Probabilities"):
                for i in range(len(tgtBatch)):
                    tgtBatch[i].append("EOS");
                outputResults(tgtBatch,r,outF)
            elif(opt.representation == "EncoderDecoderHiddenState"):
                for i in range(len(tgtBatch)):
                    tgtBatch[i].append("EOS");
                outputAlignment(srcBatch,tgtBatch,r,outF)
            srcBatch, tgtBatch = [], []


    if tgtF:
        tgtF.close()



def outputResults(srcBatch,r,outF):


    x=0
    j=0
    out= []
    for i in range(len(srcBatch)):
        out.append([])
    while(x < r.size(0)):
        for i in range(len(srcBatch)):
            if(j < len(srcBatch[i])):
               out[i].append(str(r[x].item()))
               x+=1
        j += 1
    for i in range(len(out)):
        for j in range(len(out[i])):
            outF.write(out[i][j])
            outF.write(' ')
        outF.write("\n")
        outF.flush()

def outputAlignment(srcBatch,tgtBatch,r,outF):



    for b in range(len(srcBatch)):
        for i in range(len(srcBatch[b])):
            for j in range (len(tgtBatch[b])):
                outF.write("%i-%i#%f " % (i,j,r[i,j,b]))
        outF.write("\n")


if __name__ == "__main__":
    main()

