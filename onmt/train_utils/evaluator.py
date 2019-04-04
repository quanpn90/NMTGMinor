from __future__ import division

import sys, tempfile
import onmt
import onmt.modules
#~ from onmt.metrics.gleu import sentence_gleu
#~ from onmt.metrics.sbleu import sentence_bleu
from onmt.metrics.bleu import moses_multi_bleu
#~ from onmt.utils import compute_score
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math

class Evaluator(object):
    
    def __init__(self, model, dataset, opt, cuda=False):
        
        # some properties
        self.dataset = dataset
        self.dicts = dataset['dicts']
        
        self.setIDs = dataset['dicts']['setIDs']
        
        self.model = model
        
        self.cuda = cuda
        
        # self.translator = onmt.InplaceTranslator(self.model, self.dicts, 
                                            # beam_size=1, 
                                            # cuda=self.cuda)
            
        
    def setScore(self, score):
        self.score = score
    
    def setCriterion(self, criterion):
        self.criterion = criterion
    
    
    # Compute perplexity of a data given the model
    # For a multilingual dataset, we may need the setIDs of the desired languages
    # data is a dictionary with key = setid and value = DataSet object
    def eval_perplexity(self, data, loss_function):
        
        total_loss = 0
        total_words = 0
       

        self.model.eval()
        with torch.no_grad():
            for i in range(len(data)):
                batch = data[i]
                _, predictions = model(batch)
                # exclude <s> from targets
                targets = batch[1][1:]
                # loss, _ = memoryEfficientLoss(
                        # outputs, targets, model.generator, criterion, eval=True)
                total_loss += loss
                total_words += targets.data.ne(onmt.Constants.PAD).sum()

        model.train()
        return total_loss / total_words
        
    #~ def eval_reinforce(self, data, score, verbose=False):
        #~ 
        #~ total_score = 0
        #~ total_sentences = 0
        #~ 
        #~ total_hit = 0
        #~ total_hit_sentences = 0
        #~ total_gleu = 0
        #~ 
        #~ model = self.model
        #~ model.eval()
        #~ tgtDict = self.dicts['tgt']
        #~ srcDict = self.dicts['src']
        #~ 
        #~ for i in range(len(data)):
            #~ batch = data[i][:-1]
            #~ src = batch[0]
            #~ ref = batch[1][1:]
            #~ # we need to sample
            #~ sampled_sequence = model.sample(src, max_length=100, argmax=True)
            #~ batch_size = ref.size(1)
            #~ 
            #~ for idx in xrange(batch_size):
            #~ 
                #~ tgtIds = sampled_sequence.data[:,idx]
                #~ 
                #~ tgtWords = tgtDict.convertTensorToLabels(tgtIds, onmt.Constants.EOS)        
                                #~ 
                #~ refIds = ref.data[:,idx]
                #~ 
                #~ refWords = tgtDict.convertTensorToLabels(refIds, onmt.Constants.EOS)
                #~ 
                #~ # return a single score value
                #~ s = score(refWords, tgtWords)
                #~ 
                #~ if len(s) > 2:
                    #~ gleu = s[1]
                    #~ hit = s[2]
                    #~ 
                    #~ if hit >= 0:
                        #~ total_hit_sentences += 1
                        #~ total_hit += hit
                #~ 
                #~ if verbose:
                    #~ sampledSent = " ".join(tgtWords)
                    #~ refSent = " ".join(refWords)
                    #~ 
                    #~ if s[0] > 0:
                        #~ print "SAMPLE :", sampledSent
                        #~ print "   REF :", refSent
                        #~ print "Score =", s
#~ 
                #~ # bleu is scaled by 100, probably because improvement by .01 is hard ?
                #~ total_score += s[0] * 100 
                #~ 
            #~ total_sentences += batch_size
        #~ 
        #~ if total_hit_sentences > 0:
            #~ average_hit = total_hit / total_hit_sentences
            #~ print("Average HIT : %.2f" % (average_hit * 100))
        #~ 
        #~ average_score = total_score / total_sentences
        #~ model.train()
        #~ return average_score
    
    
    # Compute translation quality of a data given the model
    # def eval_translate(self, data, beam_size=1, batch_size=16, bpe=True, bpe_token="@"):
        
        # model = self.model
        # setIDs = self.setIDs
        
        # count = 0

        # one score for each language pair
        # bleu_scores = dict()
        
        # for sid in data: # sid = setid
            
            # if self.adapt:
                # if sid != self.adapt_pair:
                    # continue
            
            # dset = data[sid]
            # model.switchLangID(setIDs[sid][0], setIDs[sid][1])
            # model.switchPairID(sid)
            
            # tgt_lang = self.dicts['tgtLangs'][setIDs[sid][1]]
            # src_lang = self.dicts['srcLangs'][setIDs[sid][0]]
            # tgt_dict = self.dicts['vocabs'][tgt_lang]
            # src_dict = self.dicts['vocabs'][src_lang]
            
            # we print translations into temp files
            # outF = tempfile.NamedTemporaryFile()
            # outRef = tempfile.NamedTemporaryFile()
                
            # for i in range(len(dset)):
                # exclude original indices
                # batch = dset[i][:-1]
                
                # src = batch[0]
                
                # exclude <s> from targets
                # targets = batch[1][1:]
                
                # transposed_targets = targets.data.transpose(0, 1) # bsize x nwords
                
                # pred = self.translator.translate(src)
                
                # bpe_string = bpe_token + bpe_token + " "
                
                # for b in range(len(pred)):
                    
                    # ref_tensor = transposed_targets[b].tolist()
                    
                    # decodedSent = tgt_dict.convertToLabels(pred[b], onmt.Constants.EOS)
                    # decodedSent = " ".join(decodedSent)
                    # decodedSent = decodedSent.replace(bpe_string, '')
                    
                    # refSent = tgt_dict.convertToLabels(ref_tensor, onmt.Constants.EOS)
                    # refSent = " ".join(refSent)
                    # refSent = refSent.replace(bpe_string, '')
                    
                    
                    # Flush the pred and reference sentences to temp files 
                    # outF.write(decodedSent + "\n")
                    # outF.flush()
                    # outRef.write(refSent + "\n")
                    # outRef.flush()
                    
            # compute bleu using external script
            # bleu = moses_multi_bleu(outF.name, outRef.name)
            # outF.close()
            # outRef.close()    
            
            # bleu_scores[sid] = bleu

        # after decoding, switch model back to training mode
        # self.model.train()
        
        # return bleu_scores