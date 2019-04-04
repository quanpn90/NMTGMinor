from onmt.metrics.gleu import sentence_gleu
import math

# hit is the metrics for getting rare words copied 

class HitMetrics(object):
    
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def hit(self, reference, hypothesis):

        index = -1;
        alpha=self.alpha
        
        for i in range(len(reference) - 3):
            if(index < 0 and reference[i] == "." and reference[i+1] == ";" and reference[i+2] == "."):
                index = i;
        pureRef= reference[:index]+[reference[-1]]
        refWords = reference[index+3:-1]
        gleu = sentence_gleu(pureRef,hypothesis)[0]
        hit = calculateHits(refWords,hypothesis)
        
        combined_score = alpha * max(hit, 0) + (1.0-alpha) * gleu
        return (combined_score, gleu, hit)
    
def calculateHits(reference, hypothesis):
    
    phrases = " ".join(reference).split(";")
    hit = 0;
    count = 0;
    for p in phrases:
        pattern = p.strip().split();
        if (len(pattern) > 0):
            count +=1;
            for i in range(len(hypothesis)):
                j = 0;
                while(j < len(pattern) and i+j < len(hypothesis) and pattern[j] == hypothesis[i+j]):
                    j +=1;
                if(j == len(pattern)):
                    hit +=1;
                    break;


    if(count == 0):
        return -1 
    else:
        return 1.0*hit/count;
