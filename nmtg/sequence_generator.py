import math
from typing import List, Sequence, Mapping

import torch

from nmtg import search
from nmtg.models.encoder_decoder import IncrementalDecoder, EncoderDecoderModel


class SequenceGenerator:
    def __init__(self, models: List[EncoderDecoderModel], dictionary, batch_first=False,
                 beam_size=1, minlen=1, maxlen_a=1.0, maxlen_b=0, stop_early=True,
                 normalize_scores=True, len_penalty=1., unk_penalty=0., retain_dropout=False,
                 sampling=False, sampling_topk=-1, sampling_temperature=1.,
                 diverse_beam_groups=-1, diverse_beam_strength=0.5, no_repeat_ngram_size=0):
        """Stores parameters for generating a target sequence.

        :param models: List of EncoderDecoderModel, ensemble of models to generate for
        :param dictionary: For getting indices of sos/eos/pad/unk
        :param beam_size: (int, optional): beam width (default: 1)
        :param min/maxlen: (int, optional): the length of the generated output will
                be bounded by minlen and maxlen (not including end-of-sentence)
        :param maxlen_a/b: (int, optional): generate sequences of maximum length
                ``ax + b``, where ``x`` is the source sentence length.
        :param stop_early: (bool, optional): stop generation immediately after we
                finalize beam_size hypotheses, even though longer hypotheses
                might have better normalized scores (default: True)
        :param normalize_scores: (bool, optional): normalize scores by the length
                of the output (default: True)
        :param len_penalty: (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
        :param unk_penalty: (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
        :param retain_dropout: (bool, optional): use dropout when generating
                (default: False)
        :param sampling: (bool, optional): sample outputs instead of beam search
                (default: False)
        :param sampling_topk: (int, optional): only sample among the top-k choices
                at each step (default: -1)
        :param sampling_temperature: (float, optional): temperature for sampling,
                where values >1.0 produces more uniform sampling and values
                <1.0 produces sharper sampling (default: 1.0)
        :param diverse_beam_groups/strength: (float, optional): parameters for
                Diverse Beam Search sampling
        """
        self.models = models
        assert all(isinstance(model.decoder, IncrementalDecoder) for model in models)
        self.batch_first = batch_first
        self.dictionary = dictionary
        self.beam_size = beam_size
        self.minlen = minlen
        self.maxlen_a = maxlen_a
        self.maxlen_b = maxlen_b
        self.stop_early = stop_early
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        self.no_repeat_ngram_size = no_repeat_ngram_size

        assert sampling_topk < 0 or sampling, '--sampling-topk requires --sampling'

        if sampling:
            self.search = search.Sampling(self.dictionary.eos(), sampling_topk, sampling_temperature)
        elif diverse_beam_groups > 0:
            self.search = search.DiverseBeamSearch(self.dictionary.eos(), diverse_beam_groups, diverse_beam_strength)
        else:
            self.search = search.BeamSearch(self.dictionary.eos())

    def generate(self, encoder_input, source_lengths, encoder_mask=None, gold_prefix=None):
        """Generate a batch of translations.

        :param encoder_input: (FloatTensor) Inputs to the encoder
        :param source_lengths: (LongTensor) Lengths of the input sequences
        :param encoder_mask: (ByteTensor) Optional mask for invalid encoder input (i.e. padding)
        :param gold_prefix: (LongTensor, optional): force decoder to begin with these tokens
        """
        with torch.no_grad():
            return self._generate(encoder_input, source_lengths, encoder_mask, gold_prefix)

    def _generate(self, encoder_input, source_lengths, encoder_mask=None, gold_prefix=None):
        """See generate"""
        if self.batch_first:
            batch_size, srclen = encoder_input.size()[:2]
        else:
            srclen, batch_size = encoder_input.size()[:2]

        maxlen = int(self.maxlen_a * srclen + self.maxlen_b)

        encoder_outs = []
        incremental_states = []
        new_order = torch.arange(batch_size).view(-1, 1).repeat(1, self.beam_size).view(-1)
        new_order = new_order.to(encoder_input.device).long()
        for model in self.models:
            if not self.retain_dropout:
                model.eval()
            incremental_states.append(IncrementalState())

            # compute the encoder output for each beam
            encoder_out = model.encoder(encoder_input, encoder_mask)
            encoder_out = self._reorder_encoder_out(encoder_out, new_order)
            encoder_outs.append(encoder_out)

        encoder_input = encoder_input.index_select(0 if self.batch_first else 1, new_order)
        if encoder_mask is not None:
            encoder_mask = encoder_mask.index_select(0 if self.batch_first else 1, new_order)

        # initialize buffers
        scores = encoder_input.data.new(batch_size * self.beam_size, maxlen + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = encoder_input.data.new(batch_size * self.beam_size, maxlen + 2).fill_(self.dictionary.pad())
        tokens_buf = tokens.clone()
        tokens[:, 0] = self.dictionary.bos()
        attn, attn_buf = None, None
        nonpad_idxs = encoder_mask if self.batch_first else encoder_mask.transpose(0, 1)

        # list of completed sentences
        finalized = [[] for i in range(batch_size)]
        finished = [False for i in range(batch_size)]
        worst_finalized = [{'idx': None, 'score': -math.inf} for i in range(batch_size)]
        num_remaining_sent = batch_size

        # number of candidate hypos per step
        cand_size = 2 * self.beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, batch_size) * self.beam_size).unsqueeze(1).type_as(tokens)
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
            assert len(finalized[sent]) <= self.beam_size
            if len(finalized[sent]) == self.beam_size:
                if self.stop_early or step == maxlen or unfinalized_scores is None:
                    return True
                # stop if the best unfinalized score is worse than the worst
                # finalized one
                best_unfinalized_score = unfinalized_scores[sent].max()
                if self.normalize_scores:
                    best_unfinalized_score /= maxlen ** self.len_penalty
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
                bbsz_idx: A vector of indices in the range [0, batch_size*beam_size),
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
            tokens_clone[:, step] = self.dictionary.eos()
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step + 2] if attn is not None else None

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step + 1]
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
                unfin_idx = idx // self.beam_size
                sent = unfin_idx + cum_unfin[unfin_idx]

                sents_seen.add((sent, unfin_idx))

                # if self.match_source_len and step > source_lengths[unfin_idx]:
                #    score = -math.inf

                def get_hypo():

                    if attn_clone is not None:
                        # remove padding tokens from attn scores
                        hypo_attn = attn_clone[i][nonpad_idxs[sent]]
                        _, alignment = hypo_attn.max(dim=0)
                    else:
                        hypo_attn = None
                        alignment = None

                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'attention': hypo_attn,  # src_len x tgt_len
                        'alignment': alignment,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized[sent]) < self.beam_size:
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
        for step in range(maxlen + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, self.beam_size).add_(corr.unsqueeze(-1) * self.beam_size)
                for i, model in enumerate(self.models):
                    model.decoder.reorder_incremental_state(incremental_states[i], reorder_state)
                    encoder_outs[i] = self._reorder_encoder_out(encoder_outs[i], reorder_state)
                encoder_input = encoder_input.index_select(0 if self.batch_first else 1, reorder_state)
                if encoder_mask is not None:
                    encoder_mask = encoder_mask.index_select(0 if self.batch_first else 1, reorder_state)

            lprobs, avg_attn_scores = self._decode(tokens[:, :step + 1], encoder_input, encoder_outs,
                                                   incremental_states, encoder_mask)

            lprobs[:, self.dictionary.pad()] = -math.inf  # never select pad
            if self.unk_penalty > 0:
                lprobs[:, self.dictionary.unk()] -= self.unk_penalty  # apply unk penalty

            if self.no_repeat_ngram_size > 0:
                # for each beam and batch sentence, generate a list of previous ngrams
                gen_ngrams = [{} for bbsz_idx in range(batch_size * self.beam_size)]
                for bbsz_idx in range(batch_size * self.beam_size):
                    gen_tokens = tokens[bbsz_idx].tolist()
                    for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                        gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
                            gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

            # Record attention scores
            if avg_attn_scores is not None:
                if attn is None:
                    attn = scores.new(batch_size * self.beam_size, srclen, maxlen + 2)
                    attn_buf = attn.clone()
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            scores_buf = scores_buf.type_as(lprobs)
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)
            if step < maxlen:
                self.search.set_src_lengths(source_lengths)

                if self.no_repeat_ngram_size > 0:
                    def calculate_banned_tokens(bbsz_idx):
                        # before decoding the next token, prevent decoding of ngrams that have already appeared
                        ngram_index = tuple(tokens[bbsz_idx, step + 2 - self.no_repeat_ngram_size:step + 1].tolist())
                        return gen_ngrams[bbsz_idx].get(ngram_index, [])

                    if step + 2 - self.no_repeat_ngram_size >= 0:
                        # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
                        banned_tokens = [calculate_banned_tokens(bbsz_idx) for bbsz_idx in
                                         range(batch_size * self.beam_size)]
                    else:
                        banned_tokens = [[] for bbsz_idx in range(batch_size * self.beam_size)]

                    for bbsz_idx in range(batch_size * self.beam_size):
                        lprobs[bbsz_idx, banned_tokens[bbsz_idx]] = float('-Inf')

                if gold_prefix is not None and step < gold_prefix.size(1):
                    probs_slice = lprobs.view(batch_size, -1, lprobs.size(-1))[:, 0, :]
                    cand_scores = torch.gather(
                        probs_slice, dim=1,
                        index=gold_prefix[:, step].view(-1, 1).data
                    ).expand(-1, cand_size)
                    cand_indices = gold_prefix[:, step].view(-1, 1).expand(batch_size, cand_size).data
                    cand_beams = torch.zeros_like(cand_indices)
                else:
                    cand_scores, cand_indices, cand_beams = self.search.step(
                        step,
                        lprobs.view(batch_size, self.beam_size, -1),
                        scores.view(batch_size, self.beam_size, -1)[:, :, :step],
                    )
            else:
                # make probs contain cumulative scores for each hypothesis
                lprobs.add_(scores[:, step - 1].unsqueeze(-1))

                # finalize all active hypotheses once we hit maxlen
                # pick the hypothesis with the highest prob of EOS right now
                torch.sort(
                    lprobs[:, self.dictionary.eos()],
                    descending=True,
                    out=(eos_scores, eos_bbsz_idx),
                )
                num_remaining_sent -= len(finalize_hypos(step, eos_bbsz_idx, eos_scores))
                assert num_remaining_sent == 0
                break

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, batch_size*beam_size),
            # and dimensions: [batch_size, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.dictionary.eos())

            finalized_sents = set()
            if step >= self.minlen:
                # only consider eos when it's among the top beam_size indices
                torch.masked_select(
                    cand_bbsz_idx[:, :self.beam_size],
                    mask=eos_mask[:, :self.beam_size],
                    out=eos_bbsz_idx,
                )
                if eos_bbsz_idx.numel() > 0:
                    torch.masked_select(
                        cand_scores[:, :self.beam_size],
                        mask=eos_mask[:, :self.beam_size],
                        out=eos_scores,
                    )
                    finalized_sents = finalize_hypos(step, eos_bbsz_idx, eos_scores, cand_scores)
                    num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < maxlen

            if len(finalized_sents) > 0:
                new_bsz = batch_size - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = cand_indices.new_ones(batch_size)
                batch_mask[cand_indices.new(finalized_sents)] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                if gold_prefix is not None:
                    gold_prefix = gold_prefix[batch_idxs]
                source_lengths = source_lengths[batch_idxs]

                scores = scores.view(batch_size, -1)[batch_idxs].view(new_bsz * self.beam_size, -1)
                scores_buf.resize_as_(scores)
                tokens = tokens.view(batch_size, -1)[batch_idxs].view(new_bsz * self.beam_size, -1)
                tokens_buf.resize_as_(tokens)
                if attn is not None:
                    attn = attn.view(batch_size, -1)[batch_idxs].view(new_bsz * self.beam_size, attn.size(1), -1)
                    attn_buf.resize_as_(attn)
                batch_size = new_bsz
            else:
                batch_idxs = None

            # set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos
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
                active_mask, k=self.beam_size, dim=1, largest=False,
                out=(_ignore, active_hypos)
            )

            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            )
            active_scores = torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores[:, step].view(batch_size, self.beam_size),
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
                out=tokens_buf.view(batch_size, self.beam_size, -1)[:, :, step + 1],
            )
            if step > 0:
                torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores_buf.view(batch_size, self.beam_size, -1)[:, :, step],
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

        return finalized

    def _decode(self, tokens, encoder_inputs, encoder_outs, incremental_states, encoder_mask=None):
        if len(self.models) == 1:
            return self._decode_one(tokens, self.models[0], encoder_inputs, encoder_outs[0], incremental_states[0],
                                    encoder_mask, log_probs=True)

        log_probs = []
        avg_attn = None
        for model, encoder_out, incremental_state, in zip(self.models, encoder_outs, incremental_states):
            probs, attn = self._decode_one(tokens, model, encoder_inputs, encoder_out, incremental_state,
                                           encoder_mask, log_probs=True)
            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        if len(self.models) > 1:
            avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(len(self.models))
            if avg_attn is not None:
                avg_attn.div_(len(self.models))
        else:
            avg_probs = log_probs[0]
        return avg_probs, avg_attn

    def _decode_one(self, tokens, model: EncoderDecoderModel, encoder_inputs, encoder_out, incremental_state,
                    encoder_mask, log_probs):
        if incremental_state is not None:
            tokens = tokens[:, -1:]
            if not self.batch_first:
                tokens = tokens.transpose(0, 1)
            decoder_out, attn_weights = model.decoder.step(tokens, encoder_out, encoder_mask=encoder_mask,
                                                           incremental_state=incremental_state)
        else:
            decoder_out, attn_weights = model.decoder(tokens, encoder_out)
        probs = model.get_normalized_probs(decoder_out, attn_weights, encoder_inputs,
                                           encoder_mask=encoder_mask, log_probs=log_probs)
        probs = probs.squeeze(1 if self.batch_first else 0)

        if attn_weights is not None:
            if isinstance(attn_weights, dict):
                attn_weights = attn_weights['attn']
            attn_weights = attn_weights.squeeze(1 if self.batch_first else 0)
        return probs, attn_weights

    def _reorder_encoder_out(self, encoder_out, new_order):
        if isinstance(encoder_out, Sequence):
            return [self._reorder_encoder_out(a, new_order) for a in encoder_out]
        elif isinstance(encoder_out, Mapping):
            return {k: self._reorder_encoder_out(v, new_order) for k, v in encoder_out.items()}
        elif isinstance(encoder_out, torch.Tensor):
            return encoder_out.index_select(0 if self.batch_first else 1, new_order)
        else:
            return encoder_out


class IncrementalState:
    def __init__(self):
        self.items = {}

    def get(self, instance, key, default=None):
        return self.items.get((id(instance), key), default)

    def set(self, instance, key, value):
        self.items[(id(instance), key)] = value

    def __str__(self):
        return str(self.items)
