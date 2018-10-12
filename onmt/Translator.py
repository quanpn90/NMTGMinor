import onmt
import onmt.modules
import torch.nn as nn
import torch
import math
from torch.autograd import Variable
from onmt.ModelConstructor import build_model
import torch.nn.functional as F
from onmt.Search import BeamSearch, DiverseBeamSearch


model_list = ['transformer', 'stochastic_transformer']

class Translator(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch
        self.beam_accum = None
        self.beta = opt.beta
        self.alpha = opt.alpha
        self.start_with_bos = opt.start_with_bos
        self.fp16 = opt.fp16
        self.stop_early = True
        self.normalize_scores = opt.normalize
        self.len_penalty = opt.alpha
        
        self.models = list()
        self.model_types = list()
        
        # models are string with | as delimiter
        models = opt.model.split("|")
        
        print(models)
        self.n_models = len(models)
        self._type = 'text'
        
        
        for i, model in enumerate(models):
            if opt.verbose:
                print('Loading model from %s' % model)
            checkpoint = torch.load(model,
                               map_location=lambda storage, loc: storage)
                               
            model_opt = checkpoint['opt']
            
            if i == 0:
                self.src_dict = checkpoint['dicts']['src']
                self.tgt_dict = checkpoint['dicts']['tgt']
            
            # Build model from the saved option
            model = build_model(model_opt, checkpoint['dicts'])
            
            model.load_state_dict(checkpoint['model'])
            
            if model_opt.model in model_list:
                if model.decoder.positional_encoder.len_max < self.opt.max_sent_length + 1:
                    print("Not enough len to decode. Renewing .. ")    
                    model.decoder.renew_buffer(self.opt.max_sent_length + 1)
            
            if opt.fp16:
                model = model.half()
            
            if opt.cuda:
                model = model.cuda()
            else:
                model = model.cpu()
                
            
            
            model.eval()
            
            self.models.append(model)
            self.model_types.append(model_opt.model)
            
        self.cuda = opt.cuda
        self.ensemble_op = opt.ensemble_op
        # ~ # self.search = BeamSearch(self.tgt_dict)
        # 1 will give the same result as BeamSearch
        self.search = DiverseBeamSearch(self.tgt_dict, 1, self.opt.diverse_beam_strength)
        self.eos = onmt.Constants.EOS
        self.pad = onmt.Constants.PAD
        self.bos = onmt.Constants.BOS
        self.unk = onmt.Constants.UNK
        self.vocab_size = self.tgt_dict.size()
        self.minlen = 1
        
        if opt.verbose:
            print('Done')

    def initBeamAccum(self):
        self.beam_accum = {
            "predicted_ids": [],
            "beam_parent_ids": [],
            "scores": [],
            "log_probs": []}
    
    # Combine distributions from different models
    def _combineOutputs(self, outputs):
        
        if len(outputs) == 1:
            return outputs[0]
        
        if self.ensemble_op == "logSum":
            output = (outputs[0])
            
            # sum the log prob
            for i in range(1, len(outputs)):
                output += (outputs[i])
                
            output.div(len(outputs))
            
            #~ output = torch.log(output)
            output = F.log_softmax(output, dim=-1)
        elif self.ensemble_op == "mean":
            output = torch.exp(outputs[0])
            
            # sum the log prob
            for i in range(1, len(outputs)):
                output += torch.exp(outputs[i])
                
            output.div(len(outputs))
            
            #~ output = torch.log(output)
            output = torch.log(output)
        elif self.ensemble_op == 'gmean':
            output = torch.exp(outputs[0])
            
            # geometric mean of the probabilities
            for i in range(1, len(outputs)):
                output *= torch.exp(outputs[i])
                
            # have to normalize
            output.pow_(1.0 / float(len(outputs)))
            norm_ = torch.norm(output, p=1, dim=-1)
            output.div_(norm_.unsqueeze(-1))

            
            output = torch.log(output)
        else:
            raise ValueError('Emsemble operator needs to be "mean" or "logSum", the current value is %s' % self.ensemble_op)
        
        return output
    
    # Take the average of attention scores
    def _combineAttention(self, attns):
        
        attn = attns[0]
        
        for i in range(1, len(attns)):
            attn += attns[i]
        
        attn.div(len(attns))
        
        return attn

    def _getbsz(self, batch):
        return batch.size(1)

    def buildData(self, srcBatch, goldBatch):
        # This needs to be the same as preprocess.py.
        
        if self.start_with_bos:
            srcData = [self.src_dict.convertToIdx(b,
                              onmt.Constants.UNK_WORD,
                              onmt.Constants.BOS_WORD)
                       for b in srcBatch]
        else:
            srcData = [self.src_dict.convertToIdx(b,
                              onmt.Constants.UNK_WORD)
                       for b in srcBatch]

        tgtData = None
        if goldBatch:
            tgtData = [self.tgt_dict.convertToIdx(b,
                       onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD,
                       onmt.Constants.EOS_WORD) for b in goldBatch]

        return onmt.Dataset(srcData, tgtData, 9999,
                            [self.opt.gpu], 
                            max_seq_num =self.opt.batch_size)

    def buildTargetTokens(self, pred, src, attn):
        tokens = self.tgt_dict.convertToLabels(pred, onmt.Constants.EOS)
        if tokens[-1] == onmt.Constants.EOS_WORD:
            tokens = tokens[:-1]  # EOS
        length = len(pred)
        
        return tokens

    def translateBatch(self, srcBatch, tgtBatch):
        
        with torch.no_grad():
            return self._translateBatch(srcBatch, tgtBatch)
        
    def _translateBatch(self, srcBatch, tgtBatch):
        
        # Batch size is in different location depending on data.

        beam_size = self.opt.beam_size
        bsz = self._getbsz(srcBatch)
        vocab_size = self.tgt_dict.size()
        max_len = self.opt.max_sent_length
        
        # srcBatch should have size len x batch
        # tgtBatch should have size len x batch
        
        contexts = dict()
        
        src = srcBatch.transpose(0, 1)
        
        #  (1) run the encoders on the src
        #  note: here we assume that the src_mask is all the same accross encoders
        for i in range(self.n_models):
            contexts[i], src_mask = self.models[i].encoder(src)
        
                
        goldScores = contexts[0].data.new(bsz).zero_()
        goldWords = 0
        
        if tgtBatch is not None:
            # Use the first model to decode
            model_ = self.models[0]
        
            tgtBatchInput = tgtBatch[:-1]
            tgtBatchOutput = tgtBatch[1:]
            tgtBatchInput = tgtBatchInput.transpose(0,1)
            
            output, coverage = model_.decoder(tgtBatchInput, contexts[0], src)
            # output should have size time x batch x dim
            
            
            #  (2) if a target is specified, compute the 'goldScore'
            #  (i.e. log likelihood) of the target under the model
            for dec_t, tgt_t in zip(output, tgtBatchOutput.data):
                gen_t = model_.generator(dec_t)
                tgt_t = tgt_t.unsqueeze(1)
                scores = gen_t.data.gather(1, tgt_t)
                scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
                goldScores += scores.squeeze(1).type_as(goldScores)
                goldWords += tgt_t.ne(onmt.Constants.PAD).sum().item()
            
            
        #  (3) Start decoding
        
        # initialize buffers
        scores = src.new(bsz * beam_size, max_len + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = src.new(bsz * beam_size, max_len + 2).fill_(self.pad)
        tokens_buf = tokens.clone()
        tokens[:, 0] = self.bos # first token is bos
        attn, attn_buf = None, None
        nonpad_idxs = None
        src_tokens = src # batch x time
        
        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        worst_finalized = [{'idx': None, 'score': -math.inf} for i in range(bsz)]
        num_remaining_sent = bsz
        
        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS   
        
        # offset arrays for converting between different indexing schemes ????
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)
        
        # helper function for allocating buffers on the fly
        buffers = {}
        
        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]
            
        def is_finished(sent, step, unfinalized_scores=None):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size:
                if self.stop_early or step == max_len or unfinalized_scores is None:
                    return True
                # stop if the best unfinalized score is worse than the worst
                # finalized one
                best_unfinalized_score = unfinalized_scores[sent].max()
                if self.normalize_scores:
                    best_unfinalized_score /= max_len ** self.len_penalty
                if worst_finalized[sent]['score'] >= best_unfinalized_score:
                    return True
            return False
            
        def finalize_hypos(step, bbsz_idx, eos_scores, unfinalized_scores=None):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.
            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.
            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
                unfinalized_scores: A vector containing scores for all
                    unfinalized hypotheses
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
            tokens_clone[:, step] = self.eos
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step+2] if attn is not None else None

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step+1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.len_penalty

            cum_unfin = []
            prev = 0
            for f in finished:
                if f:
                    prev += 1
                else:
                    cum_unfin.append(prev)

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                unfin_idx = idx // beam_size
                sent = unfin_idx + cum_unfin[unfin_idx]

                sents_seen.add((sent, unfin_idx))

                def get_hypo():

                    if attn_clone is not None:
                        # remove padding tokens from attn scores
                        hypo_attn = attn_clone[i][nonpad_idxs[sent]]
                        # ~ _, alignment = hypo_attn.max(dim=0)
                    else:
                        hypo_attn = None
                        alignment = None

                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'attention': hypo_attn,  # src_len x tgt_len
                        # ~ 'alignment': alignment,
                        # ~ 'positional_scores': pos_scores[i],
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())
                elif not self.stop_early and score > worst_finalized[sent]['score']:
                    # replace worst hypo for this sentence with new/better one
                    worst_idx = worst_finalized[sent]['idx']
                    if worst_idx is not None:
                        finalized[sent][worst_idx] = get_hypo()

                    # find new worst finalized hypo for this sentence
                    idx, s = min(enumerate(finalized[sent]), key=lambda r: r[1]['score'])
                    worst_finalized[sent] = {
                        'score': s['score'],
                        'idx': idx,
                    }

            newly_finished = []
            for sent, unfin_idx in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfinalized_scores):
                    finished[sent] = True
                    newly_finished.append(unfin_idx)
            return newly_finished
        
        reorder_state = None
        batch_idxs = None
        
        # initialize the decoder state, including:
        # - expanding the context over the batch dimension len_src x (B*beam) x H
        # - expanding the mask over the batch dimension    (B*beam) x len_src 
        decoder_states = dict()
        for i in range(self.n_models):
            decoder_states[i] = self.models[i].create_decoder_state(src, contexts[i], src_mask, beam_size, type='new')
        
        # Start decoding
        for step in range(max_len + 1):
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)

                for i, model in enumerate(self.models):
                    decoder_states[i]._reorder_incremental_state(reorder_state)
            
            lprobs, avg_attn_scores = self._decode(tokens[:, :step + 1], decoder_states)
            # ~ print(lprobs.size())
            lprobs[:, self.pad] = -math.inf  # never select pad
            # ~ lprobs[:, self.unk] = -math.inf  # never select pad
            
            # Record attention scores
            if avg_attn_scores is not None:
                if attn is None:
                    attn = scores.new(bsz * beam_size, src_tokens.size(1), max_len + 2)
                    attn_buf = attn.clone()
                    nonpad_idxs = src_tokens.ne(self.pad)
                attn[:, :, step + 1].copy_(avg_attn_scores)
            
            scores = scores.type_as(lprobs)
            scores_buf = scores_buf.type_as(lprobs)
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)
            if step < max_len:
                cand_scores, cand_indices, cand_beams = self.search.step(
                    step,
                    lprobs.view(bsz, -1, self.vocab_size),
                    scores.view(bsz, beam_size, -1)[:, :, :step],
                )
            else:
                 # make probs contain cumulative scores for each hypothesis
                lprobs.add_(scores[:, step - 1].unsqueeze(-1))
                
                # finalize all active hypotheses once we hit max_len
                # pick the hypothesis with the highest prob of EOS right now
                torch.sort(
                    lprobs[:, self.eos],
                    descending=True,
                    out=(eos_scores, eos_bbsz_idx),
                )
                num_remaining_sent -= len(finalize_hypos(
                    step, eos_bbsz_idx, eos_scores))
                assert num_remaining_sent == 0
                break
                
            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)
            
            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.eos)
            
            finalized_sents = set()
            
            if step >= self.minlen:
                # only consider eos when it's among the top beam_size indices
                torch.masked_select(
                    cand_bbsz_idx[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_bbsz_idx,
                )
                if eos_bbsz_idx.numel() > 0:
                    torch.masked_select(
                        cand_scores[:, :beam_size],
                        mask=eos_mask[:, :beam_size],
                        out=eos_scores,
                    )
                    finalized_sents = finalize_hypos(
                        step, eos_bbsz_idx, eos_scores, cand_scores)
                    num_remaining_sent -= len(finalized_sents)
            
            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
                
            assert step < max_len
            
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)
                
                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = cand_indices.new_ones(bsz)
                batch_mask[cand_indices.new(finalized_sents)] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)
                
                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
            
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                
                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                scores_buf.resize_as_(scores)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_buf.resize_as_(tokens)
                
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                    attn_buf.resize_as_(attn)
                bsz = new_bsz
                
            else:
                batch_idxs = None
                
            active_mask = buffer('active_mask')
            torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_mask.size(1)],
                out=active_mask,
            )
                
            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, _ignore = buffer('active_hypos'), buffer('_ignore')
            torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,
                out=(_ignore, active_hypos)
            )
            
            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            )
            
            active_scores = torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),
            )

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)
            
            # copy tokens and scores for active hypotheses
            torch.index_select(
                tokens[:, :step + 1], dim=0, index=active_bbsz_idx,
                out=tokens_buf[:, :step + 1],
            )
            torch.gather(
                cand_indices, dim=1, index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1],
            )
            if step > 0:
                torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
            )

            # copy attention for active hypotheses
            if attn is not None:
                torch.index_select(
                    attn[:, :, :step + 2], dim=0, index=active_bbsz_idx,
                    out=attn_buf[:, :, :step + 2],
                )

            # swap buffers
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            if attn is not None:
                attn, attn_buf = attn_buf, attn

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx
            
        # sort by score descending
        for sent in range(len(finalized)):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)
            
        return finalized, goldScores, goldWords
    
    def _decode(self, tokens, decoder_states):
    
        # require batch first for everything
        outs = dict()
        attns = dict()
        
        for i in range(self.n_models):
            decoder_hidden, coverage = self.models[i].decoder.step(tokens, decoder_states[i])
            # ~ print(decoder_hidden.size())
            
            # take the last decoder state
            decoder_hidden = decoder_hidden.transpose(0, 1).squeeze(1)
            attns[i] = coverage[:, -1, :].squeeze(1) # batch * beam x src_len
            
            # batch * beam x vocab_size 
            outs[i] = self.models[i].generator(decoder_hidden)
            
        out = self._combineOutputs(outs)
        attn = self._combineAttention(attns)
        # attn = attn[:, -1, :]
        attn = None
        return out, attn
        
        

    def translate(self, srcBatch, goldBatch):
        #  (1) convert words to indexes
        dataset = self.buildData(srcBatch, goldBatch)
        batch = dataset.next()[0]
        batch.cuda()
        # ~ batch = self.to_variable(dataset.next()[0])
        src = batch.get('source')
        tgt = batch.get('target_input')
        bsz = batch.size

        #  (2) translate
        # ~ pred, predScore, attn, predLength, goldScore, goldWords = self.translateBatch(src, tgt)
        finalized, goldScore, goldWords = self.translateBatch(src, tgt)
        
        

        #  (3) convert indexes to words
        predBatch = []
        predScore = []
        predLength = []
        for b in range(bsz):
            
            a = 0
            pred = finalized[b]
            # ~ print(finalized[b][0]['tokens'])
            
            predBatch.append(
                [self.buildTargetTokens(pred[n]['tokens'], srcBatch[b], pred[n]['attention'])
                 for n in range(self.opt.n_best)]
            )
            predLength.append(
                [(pred[n]['tokens'].size(0), srcBatch[b], pred[n]['attention'])
                 for n in range(self.opt.n_best)]
            )
            
            predScore.append([pred[n]['score'] for n in range(self.opt.n_best)])
        
        return predBatch, predScore, predLength, goldScore, goldWords


