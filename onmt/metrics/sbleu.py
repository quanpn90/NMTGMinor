import sys
import math
ngramLength = 4;

smoothingConstant=0.1
bpSmoothingConstant=1.5

def getCounts(words):
    counts = {}
    for i in range(len(words)):
        ngram = []
        for j in range(ngramLength):
            if(i+j < len(words)):
                ngram.append(words[i+j])
                if(" ".join(ngram) in counts):
                    counts[" ".join(ngram)] += 1
                else:
                    counts[" ".join(ngram)] = 1
    return counts

def getRefCounts(ref):
	
	count = getCounts(ref)
	length = len(ref)
	
	return count, length

    #~ file = open(filename)
#~ 
    #~ line = file.readline()
#~ 
    #~ counts = []
    #~ length = []
#~ 
    #~ while(line):
#~ 
        #~ counts.append(getCounts(line.split()))
        #~ length.append(len(line.split()))
        #~ line = file.readline()
    #~ return counts,length

def countMatches(hyp, ref):
    counts = [0] * ngramLength
    found = {}
    for i in range(len(hyp)):
        ngram = []
        for j in range(ngramLength):
            if(i+j < len(hyp)):
                ngram.append(hyp[i+j])
                if(" ".join(ngram) in ref and (" ".join(ngram) not in found or found[" ".join(ngram)] < ref[" ".join(ngram)])):
                    counts[j] += 1
                    if(" ".join(ngram) in found):
                        found[" ".join(ngram)] += 1;
                    else:
                        found[" ".join(ngram)] = 1;

    return counts


def calcBLEU(counts,length,referenceLength):
        result = 1;
        for i in range(ngramLength):
            if(length -i > 0):
                #cannot calculte 4-gram precision for sentence length 3
                result *= 1.0*(counts[i]+smoothingConstant)/(length-i+smoothingConstant)
        result = pow(result,1.0/ngramLength);
        if(length > referenceLength):
            return result
        else:
            if(length == 0):
                return math.exp(1.0-(referenceLength+bpSmoothingConstant)/1)*result
            return math.exp(1.0-(referenceLength+bpSmoothingConstant)/length)*result

def calc(refCounts,refLength,hyp):
	
	target = hyp
	count = countMatches(target, refCounts)
	s = calcBLEU(count, len(target), refLength)
	
	return s

    #~ file = open(filename)
#~ 
    #~ out = open(outname,'w')
    #~ line = file.readline()
#~ 
    #~ bestScores = []
    #~ firstScores = []
#~ 
    #~ while(line):
        #~ number = int(line.split("|||")[0])
        #~ target = line.split("|||")[1].split()
        #~ count = countMatches(target,refCounts[number])
        #~ s = calcBLEU(count,len(target),refLength[number])
        #~ print >>out,s
        #~ if(number < len(bestScores)):
            #~ if(bestScores[number] < s):
                #~ bestScores[number] = s;
        #~ else:
            #~ firstScores.append(s)
            #~ bestScores.append(s)
        #~ line = file.readline()
#~ 
    #~ avg = sum(firstScores)/len(firstScores)
    #~ oracle = sum(bestScores)/len(bestScores)
    #~ print "First hypothesis: ",avg
    #~ print "Oracle score: ",oracle

# inputs are lists of words
def sentence_bleu(ref, hyp):
	
	refCounts, refLength = getRefCounts(ref)
	sbleu = calc(refCounts,refLength,hyp)
	return (sbleu,)
	
	
