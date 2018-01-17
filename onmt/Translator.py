import onmt
import onmt.modules
import torch.nn as nn
import torch
import math
from torch.autograd import Variable
from onmt.ModelConstructor import build_model


def loadImageLibs():
    "Conditional import of torch image libs."
    global Image, transforms
    from PIL import Image
    from torchvision import transforms


class Translator(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch
        self.beam_accum = None
        self.beta = opt.beta
        self.alpha = opt.alpha
        
        if opt.verbose:
            print('Loading model from %s' % opt.model)
        checkpoint = torch.load(opt.model,
                               map_location=lambda storage, loc: storage)
        
        

        model_opt = checkpoint['opt']
        print(model_opt)
        self.src_dict = checkpoint['dicts']['src']
        self.tgt_dict = checkpoint['dicts']['tgt']
        onmt.Constants.weight_norm = model_opt.weight_norm
        self._type = model_opt.encoder_type \
            if "encoder_type" in model_opt else "text"

        # Build model from the saved option
        model = build_model(model_opt, checkpoint['dicts'])

        
        model.load_state_dict(checkpoint['model'])
        
        if model_opt.model == 'transformer':
            if model.decoder.positional_encoder.len_max < self.opt.max_sent_length:
                print("Not enough len to decode. Renewing .. ")    
                model.decoder.renew_buffer(self.opt.max_sent_length)
           
        if opt.cuda:
            model.cuda()
        else:
            model.cpu()
        
        self.cuda = opt.cuda
            
        self.model_type = model_opt.model


        self.model = model
        self.model.eval()
        if opt.verbose:
            print('Done')

    def initBeamAccum(self):
        self.beam_accum = {
            "predicted_ids": [],
            "beam_parent_ids": [],
            "scores": [],
            "log_probs": []}

    def _getBatchSize(self, batch):
        if self._type == "text":
            return batch.size(1)
        else:
            return batch.size(0)
            
    def to_variable(self, data):
        
        for i, t in enumerate(data):
            if self.cuda:
                data[i] = Variable(data[i].cuda())
            else:
                data[i] = Variable(data[i])

        return data

    def buildData(self, srcBatch, goldBatch):
        # This needs to be the same as preprocess.py.
        if self._type == "text":
            srcData = [self.src_dict.convertToIdx(b,
                              onmt.Constants.UNK_WORD)
                       for b in srcBatch]
        elif self._type == "img":
            srcData = [transforms.ToTensor()(
                Image.open(self.opt.src_img_dir + "/" + b[0]))
                       for b in srcBatch]

        tgtData = None
        if goldBatch:
            tgtData = [self.tgt_dict.convertToIdx(b,
                       onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD,
                       onmt.Constants.EOS_WORD) for b in goldBatch]

        return onmt.Dataset(srcData, tgtData, 9999,
                            [self.opt.gpu], volatile=True,
                            data_type=self._type, balance=False, max_seq_num =self.opt.batch_size)

    def buildTargetTokens(self, pred, src, attn):
        tokens = self.tgt_dict.convertToLabels(pred, onmt.Constants.EOS)
        tokens = tokens[:-1]  # EOS
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == onmt.Constants.UNK_WORD:
                    _, maxIndex = attn[i].max(0)
                    tokens[i] = src[maxIndex[0]]
        return tokens

    def translateBatch(self, srcBatch, tgtBatch):
        
        torch.set_grad_enabled(False)
        # Batch size is in different location depending on data.

        beamSize = self.opt.beam_size
        batchSize = self._getBatchSize(srcBatch)
        
        if self.model_type == 'recurrent':

            #  (1) run the encoder on the src
            encStates, context = self.model.encoder(srcBatch)

            rnnSize = context.size(2)
            
            decoder = self.model.decoder
            attentionLayer = decoder.attn
            useMasking = ( self._type == "text" and batchSize > 1 )

            #  This mask is applied to the attention model inside the decoder
            #  so that the attention ignores source padding
            attn_mask = srcBatch.eq(onmt.Constants.PAD).t()

            #  (2) if a target is specified, compute the 'goldScore'
            #  (i.e. log likelihood) of the target under the model
            goldScores = context.data.new(batchSize).zero_()
            goldWords = 0
            if tgtBatch is not None:
                decStates = encStates
                decOut = self.model.make_init_decoder_output(context)
                initOutput = self.model.make_init_decoder_output(context)
                decOut, decStates, attn = self.model.decoder(
                    tgtBatch[:-1], decStates, context, initOutput, attn_mask=attn_mask)
                for dec_t, tgt_t in zip(decOut, tgtBatch[1:].data):
                    gen_t = self.model.generator.forward(dec_t)
                    tgt_t = tgt_t.unsqueeze(1)
                    scores = gen_t.data.gather(1, tgt_t)
                    scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
                    goldScores += scores
                    goldWords += tgt_t.ne(onmt.Constants.PAD).sum()

            #  (3) run the decoder to generate sentences, using beam search

            # Expand tensors for each beam.
            context = Variable(context.data.repeat(1, beamSize, 1))

            decStates = (Variable(encStates[0].data.repeat(1, beamSize, 1)),
                         Variable(encStates[1].data.repeat(1, beamSize, 1)))

            beam = [onmt.Beam(beamSize, self.opt.cuda) for k in range(batchSize)]

            decOut = self.model.make_init_decoder_output(context)

            
            attn_mask = srcBatch.eq(
                onmt.Constants.PAD).t() \
                                   .unsqueeze(0) \
                                   .repeat(beamSize, 1, 1)
            
            
            batchIdx = list(range(batchSize))
            remainingSents = batchSize
            for i in range(self.opt.max_sent_length):
                # Prepare decoder input.
                input = torch.stack([b.getCurrentState() for b in beam
                                     if not b.done]).t().contiguous().view(1, -1)
                decOut, decStates, attn = self.model.decoder(
                    Variable(input), decStates, context, decOut, attn_mask=attn_mask)
                # decOut: 1 x (beam*batch) x numWords
                decOut = decOut.squeeze(0)
                out = self.model.generator.forward(decOut)

                # batch x beam x numWords
                wordLk = out.view(beamSize, remainingSents, -1) \
                            .transpose(0, 1).contiguous()
                attn = attn.view(beamSize, remainingSents, -1) \
                           .transpose(0, 1).contiguous()

                active = []
                for b in range(batchSize):
                    if beam[b].done:
                        continue

                    idx = batchIdx[b]
                    if not beam[b].advance(wordLk.data[idx], attn.data[idx]):
                        active += [b]

                    for decState in decStates:  # iterate over h, c
                        # layers x beam*sent x dim
                        sentStates = decState.view(-1, beamSize,
                                                   remainingSents,
                                                   decState.size(2))[:, :, idx]
                        sentStates.data.copy_(
                            sentStates.data.index_select(
                                1, beam[b].getCurrentOrigin()))

                if not active:
                    break

                # in this section, the sentences that are still active are
                # compacted so that the decoder is not run on completed sentences
                activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
                batchIdx = {beam: idx for idx, beam in enumerate(active)}

                def updateActive(t):
                    # select only the remaining active sentences
                    view = t.data.view(-1, remainingSents, rnnSize)
                    newSize = list(t.size())
                    newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                    return Variable(view.index_select(1, activeIdx)
                                    .view(*newSize))

                decStates = (updateActive(decStates[0]),
                             updateActive(decStates[1]))
                decOut = updateActive(decOut)
                context = updateActive(context)
               
                attn_mask_data = attn_mask.data.index_select(1, activeIdx)
                attn_mask = Variable(attn_mask_data)

                remainingSents = len(active)

            #  (4) package everything up
            allHyp, allScores, allAttn = [], [], []
            n_best = self.opt.n_best
            allLengths = []

            for b in range(batchSize):
                scores, ks = beam[b].sortBest()

                allScores += [scores[:n_best]]
                hyps, attn, length = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
                allHyp += [hyps]
                allLengths += [length]
                valid_attn = srcBatch.data[:, b].ne(onmt.Constants.PAD) \
                                                .nonzero().squeeze(1)
                attn = [a.index_select(1, valid_attn) for a in attn]
                allAttn += [attn]

                if self.beam_accum:
                    self.beam_accum["beam_parent_ids"].append(
                        [t.tolist()
                         for t in beam[b].prevKs])
                    self.beam_accum["scores"].append([
                        ["%4f" % s for s in t.tolist()]
                        for t in beam[b].allScores][1:])
                    self.beam_accum["predicted_ids"].append(
                        [[self.tgt_dict.getLabel(id)
                          for id in t.tolist()]
                         for t in beam[b].nextYs][1:])
            
            torch.set_grad_enabled(True)
            
            return allHyp, allScores, allAttn, allLengths, goldScores, goldWords
        elif self.model_type == 'transformer':

            assert self.opt.batch_size == 1, "Transformer only works with batch_size 1 atm"
            vocab_size = self.tgt_dict.size()
            allHyp, allScores, allAttn, allLengths = [], [], [], []
            
            # srcBatch should have size len x batch
            # tgtBatch should have size len x batch
            
            src = srcBatch.transpose(0, 1)
            context, src_mask = self.model.encoder(src)
            
            goldScores = context.data.new(batchSize).zero_()
            
            goldWords = 0
            if tgtBatch is not None:
            
                tgtBatchInput = tgtBatch[:-1]
                tgtBatchOutput = tgtBatch[1:]
                tgtBatchInput = tgtBatchInput.transpose(0,1)
                
                output, coverage = self.model.decoder(tgtBatchInput, context, src_mask)
                output = output.transpose(0, 1) # transpose to have time first, like RNN models
                
                
                #  (2) if a target is specified, compute the 'goldScore'
                #  (i.e. log likelihood) of the target under the model
                for dec_t, tgt_t in zip(output, tgtBatchOutput.data):
                    gen_t = self.model.generator(dec_t)
                    tgt_t = tgt_t.unsqueeze(1)
                    scores = gen_t.data.gather(1, tgt_t)
                    scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
                    goldScores += scores
                    goldWords += tgt_t.ne(onmt.Constants.PAD).sum()
            
                #  (3) Start decoding
                                                
                # print(src_mask.size())
                mask_src = src_mask 
                remaining_beams = beamSize
                logLikelihoods = []
                preds = []
                atten_probs = []
                coverage_penalties = []
                lengths = []
                
                # predict the first word
                decode_input = torch.LongTensor([onmt.Constants.BOS]).unsqueeze(1)
                if self.opt.cuda: decode_input = decode_input.cuda()
                decode_input = Variable(decode_input)
                decoder_hidden, coverage = self.model.decoder(decode_input, context, mask_src)
                dist = self.model.generator(decoder_hidden.view(-1, decoder_hidden.size(-1)))
                scores, scores_id = dist.view(-1).topk(beamSize)
                beam_index = scores_id / vocab_size
                pred_id = (scores_id - beam_index*vocab_size).view(beamSize, -1)
                decode_input = torch.cat((decode_input.repeat(beamSize ,1), pred_id), 1)
                context = context.repeat(beamSize, 1, 1)
                
                # continus to predict next work until <EOS>
                step = 1
                while step < self.opt.max_sent_length and remaining_beams>0:
                    step += 1
                    decoder_hidden, coverage = self.model.decoder(decode_input, context, mask_src)
                    out = self.model.generator(decoder_hidden.view(-1, decoder_hidden.size(-1)))
                    out = out.view(remaining_beams, -1, vocab_size)
                    out = scores.unsqueeze(1) + out[:, -1, :] # Add up the scores from previous steps
                    scores, scores_id = out.view(-1).topk(remaining_beams)
                    beam_id = scores_id / vocab_size
                    pred_id = (scores_id - beam_id*vocab_size).view(remaining_beams, -1)
                    decode_input = torch.cat((decode_input[beam_id], pred_id), 1) 
                    # remove finished beams
                    check = decode_input[:, -1].eq(onmt.Constants.EOS).data
                    
                    if step == self.opt.max_sent_length -1:
                        check.fill_(1)
                    
                    finished_index = check.nonzero().squeeze()
                    continue_index = (1-check).nonzero().squeeze()
                    # continue_index = 1 - finished_index
                    
                    # length_
                    for idx in finished_index:
                        logLikelihoods.append(scores[idx].data[0])
                        pred = decode_input[idx,:].data.tolist()
                        pred = pred[1:] # remove BOS
                        preds.append(pred)
                        lengths.append(len(preds[-1]))
                        
                        atten_prob = torch.sum(coverage[idx,:,:], dim=0)
                        atten_probs.append(coverage[idx,:,:])
                        coverage_penalty = torch.log(atten_prob.masked_select(atten_prob.le(1)))
                        coverage_penalty = self.beta * torch.sum(coverage_penalty).data[0]
                        coverage_penalties.append(coverage_penalty)       
                        remaining_beams -= 1
                    if len(continue_index) > 0:
                        # var = Variable(continue_index)
                        scores = Variable(scores.data.index_select(0, continue_index))
                        decode_input = Variable(decode_input.data.index_select(0, continue_index))
                        context = Variable(context.data.index_select(0, continue_index))
            # normalize the final scores by length and coverage 
            len_penalties = [math.pow(len(pred), self.alpha) for pred in preds]
            final_scores = [logLikelihoods[i]/len_penalties[i] + coverage_penalties[i] for i in range(len(preds))]
            sorted_scores_arg = sorted(range(len(preds)), key=lambda i:-final_scores[i])
            
            sorted_preds = [ preds[sorted_scores_arg[i]] for i in range(beamSize) ]
            sorted_logLikelihoods = [ logLikelihoods[sorted_scores_arg[i]] for i in range(beamSize) ]
            sorted_attn_probs = [ atten_probs[sorted_scores_arg[i]] for i in range(beamSize) ]
            sorted_lengths = [ lengths[sorted_scores_arg[i]] for i in range(beamSize) ]
            #~ best_beam = sorted_scores_arg[0]
            # tgt_id = ' '.join(map(str, preds[best_beam]))
            # target = self.id2word(tgt_id)
            # tgt_pieces = [self.id2word(str(i)) for i in preds[best_beam]]
            # attn = [atten_probs[best_beam], src_pieces, tgt_pieces]
            # return target, attn                         
            allHyp += [sorted_preds]
            allScores += [sorted_logLikelihoods]
            allAttn.append(sorted_attn_probs)

            allLengths.append(sorted_lengths)
            
            
            
            torch.set_grad_enabled(True)
            
            return allHyp, allScores, allAttn, allLengths, goldScores, goldWords
                                
                    
            # raise NotImplementedError
            

    def translate(self, srcBatch, goldBatch):
        #  (1) convert words to indexes
        dataset = self.buildData(srcBatch, goldBatch)
        batch = self.to_variable(dataset.next()[0])
        src, tgt = batch
        batchSize = self._getBatchSize(src)

        #  (2) translate
        pred, predScore, attn, predLength, goldScore, goldWords = self.translateBatch(src, tgt)
        

        #  (3) convert indexes to words
        predBatch = []
        for b in range(batchSize):
            predBatch.append(
                [self.buildTargetTokens(pred[b][n], srcBatch[b], attn[b][n])
                 for n in range(self.opt.n_best)]
            )

        return predBatch, predScore, predLength, goldScore, goldWords
