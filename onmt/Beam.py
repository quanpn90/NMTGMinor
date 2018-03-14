from __future__ import division
import torch
import onmt

"""
 Class for managing the internals of the beam search process.


         hyp1-hyp1---hyp1 -hyp1
                 \             /
         hyp2 \-hyp2 /-hyp2hyp2
                               /      \
         hyp3-hyp3---hyp3 -hyp3
         ========================

 Takes care of beams, back pointers, and scores.
"""


class Beam(object):
    def __init__(self, size, cuda=False):

        self.size = size
        #~ self.done = False
        self.n_best = self.size

        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.allScores = []

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(onmt.Constants.PAD)]
        self.nextYs[0][0] = onmt.Constants.BOS

        # The attentions (matrix) for each time.
        self.attn = []
        self.finished = []
        
        self._eos = onmt.Constants.EOS
        self.eos_top = False

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        return self.nextYs[-1]

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk, attnOut):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)
            
            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == onmt.Constants.EOS:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]

        flatBeamLk = beamLk.view(-1)

        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)
        self.allScores.append(self.scores)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords
        self.prevKs.append(prevK)
        self.nextYs.append(bestScoresId - prevK * numWords)
        self.attn.append(attnOut.index_select(0, prevK))
        
        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                #~ global_scores = self.global_scorer.score(self, self.scores)
                global_scores = self.scores
                s = global_scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == onmt.Constants.EOS:
            #~ self.done = True
            self.eos_top = True
            self.allScores.append(self.scores)

        return self.done
        
    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    #~ def sortBest(self):
        #~ return torch.sort(self.scores, 0, True)
#~ 
    #~ def getBest(self):
        #~ "Get the score of the best in the beam."
        #~ scores, ids = self.sortBest()
        #~ return scores[1], ids[1]
        
    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                global_scores = self.scores
                s = global_scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))
                i += 1

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.

        Parameters.

             * `k` - the position in the beam to construct.

         Returns.

            1. The hypothesis
            2. The attention at each time step.
        """
        hyp, attn = [], []
        lengths = []
        for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
            hyp.append(self.nextYs[j+1][k])
            attn.append(self.attn[j][k])
            k = self.prevKs[j][k]
        
        length = len(hyp)

        return hyp[::-1], torch.stack(attn[::-1]), length
