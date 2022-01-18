# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import onmt


class Search(object):

    def __init__(self, tgt_dict):
        # self.pad = onmt.constants.PAD
        # self.unk = onmt.constants.UNK
        # self.eos = onmt.constants.EOS
        # self.bos = onmt.constants.BOS
        self.vocab_size = tgt_dict.size()
        self.scores_buf = None
        self.indices_buf = None
        self.beams_buf = None

    def _init_buffers(self, t):
        if self.scores_buf is None:
            self.scores_buf = t.new()
            self.indices_buf = torch.LongTensor().to(device=t.device)

    def step(self, step, lprobs, scores, beam_size):
        """Take a single search step.

        Args:
            step: the current search step, starting at 0
            lprobs: (bsz x input_beam_size x vocab_size)
                the model's log-probabilities over the vocabulary at the current step
            scores: (bsz x input_beam_size x step)
                the historical model scores of each hypothesis up to this point

        Return: A tuple of (scores, indices, beams) where:
            scores: (bsz x output_beam_size)
                the scores of the chosen elements; output_beam_size can be
                larger than input_beam_size, e.g., we may return
                2*input_beam_size to account for EOS
            indices: (bsz x output_beam_size)
                the indices of the chosen elements
            beams: (bsz x output_beam_size)
                the hypothesis ids of the chosen elements, in the range [0, input_beam_size)
                :param lprobs:
                :param step:
                :param scores:
                :param beam_size:
        """
        raise NotImplementedError

    def set_src_lengths(self, src_lengths):
        self.src_lengths = src_lengths


class BeamSearch(Search):

    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)

    def step(self, step, lprobs, scores, initial_score=None, **kwargs):
        super()._init_buffers(lprobs)

        # batch size first, then beam size
        bsz, beam_size, vocab_size = lprobs.size()

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            if initial_score is None or torch.sum(initial_score).item() == 0:
                lprobs = lprobs[:, ::beam_size, :].contiguous()
            else:
                lprobs.add_(initial_score.unsqueeze(-1))
            # if we don't do this, the first beam will contain top K of exactly the same thing ...
        else:

            # make probs contain cumulative scores for each hypothesis
            lprobs.add_(scores[:, :, step - 1].unsqueeze(-1))

        # here lprobs should be (bsz, beam_size, V) (in streaming, bsz should be 1)

        torch.topk(
            lprobs.view(bsz, -1),  # after view, it should be (bsz, beam_size x V)
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 2,
                lprobs.view(bsz, -1).size(1) - beam_size,  # -beam_size so we never select pad (beam_size times)
            ),
            out=(self.scores_buf.resize_(0), self.indices_buf.resize_(0)),
        )

        # torch.div(self.indices_buf, vocab_size, out=self.beams_buf)
        # beams_buf helps us know where the origin of each
        self.beams_buf = torch.true_divide(self.indices_buf, vocab_size).long()

        # indices: the word indices in the vocabulary
        self.indices_buf.fmod_(vocab_size)
        return self.scores_buf, self.indices_buf, self.beams_buf


class DiverseBeamSearch(Search):
    """Diverse Beam Search.

    See "Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence
    Models" for details.

    We only implement the Hamming Diversity penalty here, which performed best
    in the original paper.
    """

    def __init__(self, tgt_dict, num_groups, diversity_strength):
        super().__init__(tgt_dict)
        self.num_groups = num_groups
        self.diversity_strength = -diversity_strength
        self.diversity_buf = None
        self.beam = BeamSearch(tgt_dict)

    def step(self, step, lprobs, scores):
        super()._init_buffers(lprobs)
        bsz, beam_size, vocab_size = lprobs.size()
        if beam_size % self.num_groups != 0:
            raise ValueError(
                'DiverseBeamSearch requires --beam to be divisible by the number of groups'
            )
        group_size = beam_size // self.num_groups

        # initialize diversity penalty
        if self.diversity_buf is None:
            self.diversity_buf = lprobs.new()
        torch.zeros(lprobs[:, 0, :].size(), out=self.diversity_buf)

        scores_G, indices_G, beams_G = [], [], []
        for g in range(self.num_groups):
            lprobs_g = lprobs[:, g::self.num_groups, :]
            scores_g = scores[:, g::self.num_groups, :] if step > 0 else None

            # apply diversity penalty
            if g > 0:
                lprobs_g = torch.add(lprobs_g, self.diversity_strength, self.diversity_buf.unsqueeze(1))
            else:
                lprobs_g = lprobs_g.contiguous()

            scores_buf, indices_buf, beams_buf = self.beam.step(step, lprobs_g, scores_g)
            beams_buf.mul_(self.num_groups).add_(g)

            scores_G.append(scores_buf.clone())
            indices_G.append(indices_buf.clone())
            beams_G.append(beams_buf.clone())

            # update diversity penalty
            self.diversity_buf.scatter_add_(
                1,
                indices_buf,
                self.diversity_buf.new_ones(indices_buf.size())
            )

        # interleave results from different groups
        self.scores_buf = torch.stack(scores_G, dim=2, out=self.scores_buf).view(bsz, -1)
        self.indices_buf = torch.stack(indices_G, dim=2, out=self.indices_buf).view(bsz, -1)
        self.beams_buf = torch.stack(beams_G, dim=2, out=self.beams_buf).view(bsz, -1)
        return self.scores_buf, self.indices_buf, self.beams_buf


class Sampling(Search):
    sampling_topk: int
    sampling_topp: float

    def __init__(self, tgt_dict, sampling_topk=-1, sampling_topp=-1.0):
        super().__init__(tgt_dict)
        self.sampling_topk = sampling_topk
        self.sampling_topp = sampling_topp

    def _sample_topp(self, lprobs):
        """Sample among the smallest set of elements whose cumulative probability mass exceeds p.

        See `"The Curious Case of Neural Text Degeneration"
        (Holtzman et al., 2019) <https://arxiv.org/abs/1904.09751>`_.

        Args:
            lprobs: (bsz x input_beam_size x vocab_size)
                the model's log-probabilities over the vocabulary at the current step

        Return: A tuple of (trimed_probs, truncated_indices) where:
            trimed_probs: (bsz x input_beam_size x ?)
                the model's probabilities over the elements selected to sample from. The
                width of the third dimension is determined by top-P.
            truncated_indices: (bsz x input_beam_size x ?)
                the indices of the chosen elements.
        """
        probs = lprobs.exp_()

        # sort the last dimension (vocab dimension) in descending order
        sorted_probs, sorted_indices = probs.sort(descending=True)

        # compute a mask to indicate the words to be included in the top-P set.
        cumsum_probs = sorted_probs.cumsum(dim=2)
        mask = cumsum_probs.lt(self.sampling_topp)

        # note that mask was computed by 'lt'. One more word needs to be included
        # so that the cumulative probability mass can exceed p.
        cumsum_mask = mask.cumsum(dim=2)
        last_included = cumsum_mask[:, :, -1:]
        last_included.clamp_(0, mask.size()[2] - 1)
        mask = mask.scatter_(2, last_included, 1)

        # truncate unnecessary dims.
        max_dim = last_included.max()
        truncated_mask = mask[:, :, : max_dim + 1]
        truncated_probs = sorted_probs[:, :, : max_dim + 1]
        truncated_indices = sorted_indices[:, :, : max_dim + 1]

        # trim the words that are not in top-P by setting their probabilities
        # to 0, so that they would not be sampled later.
        trim_mask = ~truncated_mask
        trimed_probs = truncated_probs.masked_fill_(trim_mask, 0)
        return trimed_probs, truncated_indices

    @torch.jit.export
    def step(
        self,
        step: int,
        lprobs,
        scores,
        prev_output_tokens = None,
        original_batch_idxs = None,
    ):
        bsz, beam_size, vocab_size = lprobs.size()

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()

        if self.sampling_topp > 0:
            # only sample from the smallest set of words whose cumulative probability mass exceeds p
            probs, top_indices = self._sample_topp(lprobs)
        elif self.sampling_topk > 0:
            # only sample from top-k candidates
            lprobs, top_indices = lprobs.topk(self.sampling_topk)
            probs = lprobs.exp_()
        else:
            probs = lprobs.exp_()

            # dummy data to be consistent with true branch for type check
            top_indices = torch.empty(0).to(probs)
        # sample
        if step == 0:
            indices_buf = torch.multinomial(
                probs.view(bsz, -1),
                beam_size,
                replacement=True,
            ).view(bsz, beam_size)
        else:
            indices_buf = torch.multinomial(
                probs.view(bsz * beam_size, -1),
                1,
                replacement=True,
            ).view(bsz, beam_size)

        if step == 0:
            # expand to beam size
            probs = probs.expand(bsz, beam_size, -1)

        # gather scores
        scores_buf = torch.gather(probs, dim=2, index=indices_buf.unsqueeze(-1))
        scores_buf = scores_buf.log_().view(bsz, -1)

        # remap indices if using top-k or top-P sampling
        if self.sampling_topk > 0 or self.sampling_topp > 0:
            indices_buf = torch.gather(
                top_indices.expand(bsz, beam_size, -1),
                dim=2,
                index=indices_buf.unsqueeze(-1),
            ).squeeze(2)

        if step == 0:
            beams_buf = indices_buf.new_zeros(bsz, beam_size)
        else:
            beams_buf = torch.arange(0, beam_size).to(indices_buf).repeat(bsz, 1)
            # make scores cumulative
            scores_buf.add_(
                torch.gather(scores[:, :, step - 1], dim=1, index=beams_buf)
            )

        return scores_buf, indices_buf, beams_buf
