import sys
import onmt
import onmt.modules
import torch.nn as nn
import torch
import math
from onmt.model_factory import build_model, optimize_model
import torch.nn.functional as F
from onmt.inference.search import BeamSearch, DiverseBeamSearch, Sampling
from onmt.inference.translator import Translator
from onmt.constants import add_tokenidx
from onmt.options import backward_compatible

# buggy lines: 392, 442, 384
model_list = ['transformer', 'stochastic_transformer', 'fusion_network']


class FastTranslator(Translator):
    """
    A fast implementation of the Beam Search based translator
    Based on Fairseq implementation
    """

    def __init__(self, opt):

        super().__init__(opt)

        self.src_bos = onmt.constants.SRC_BOS
        self.src_eos = onmt.constants.SRC_EOS
        self.src_pad = onmt.constants.SRC_PAD
        self.src_unk = onmt.constants.SRC_UNK

        self.tgt_bos = self.bos_id
        self.tgt_pad = onmt.constants.TGT_PAD
        self.tgt_eos = onmt.constants.TGT_EOS
        self.tgt_unk = onmt.constants.TGT_UNK

        if opt.sampling:
            self.search = Sampling(self.tgt_dict)
        else:
            self.search = BeamSearch(self.tgt_dict)

        self.vocab_size = self.tgt_dict.size()
        self.min_len = 1
        self.normalize_scores = opt.normalize
        self.len_penalty = opt.alpha
        self.buffering = not opt.no_buffering

        if hasattr(opt, 'no_repeat_ngram_size'):
            self.no_repeat_ngram_size = opt.no_repeat_ngram_size
        else:
            self.no_repeat_ngram_size = 0

        if hasattr(opt, 'dynamic_max_len'):
            self.dynamic_max_len = opt.dynamic_max_len
        else:
            self.dynamic_max_len = False

        if hasattr(opt, 'dynamic_max_len_scale'):
            self.dynamic_max_len_scale = opt.dynamic_max_len_scale
        else:
            self.dynamic_max_len_scale = 1.2

        if opt.verbose:
            # print('* Current bos id is: %d, default bos id is: %d' % (self.tgt_bos, onmt.constants.BOS))
            print("src bos id is %d; src eos id is %d;  src pad id is %d; src unk id is %d"
                  % (self.src_bos, self.src_eos, self.src_pad, self.src_unk))
            print("tgt bos id is %d; tgt eos id is %d;  tgt_pad id is %d; tgt unk id is %d"
                  % (self.tgt_bos, self.tgt_eos, self.tgt_pad, self.tgt_unk))

            print('* Using fast beam search implementation')

        if opt.vocab_list:
            word_list = list()
            for line in open(opt.vocab_list).readlines():
                word = line.strip()
                word_list.append(word)

                self.filter = torch.Tensor(self.tgt_dict.size()).zero_()
            for word_idx in [self.tgt_eos, self.tgt_unk]:
                self.filter[word_idx] = 1

            for word in word_list:
                idx = self.tgt_dict.lookup(word)
                if idx is not None:
                    self.filter[idx] = 1

            self.filter = self.filter.bool()
            if opt.cuda:
                self.filter = self.filter.cuda()

            self.use_filter = True
        elif opt.vocab_id_list:
            ids = torch.load(opt.vocab_id_list)

            print('[INFO] Loaded word list with %d ids' % len(ids))
            self.filter = torch.Tensor(self.tgt_dict.size()).zero_()
            for id in ids:
                self.filter[id] = 1

            self.filter = self.filter.bool()
            if opt.cuda:
                self.filter = self.filter.cuda()

            self.use_filter = True
        else:
            self.use_filter = False

        # Sub-model is used for ensembling Speech and Text models
        if opt.sub_model:
            self.sub_models = list()
            self.sub_model_types = list()

            # models are string with | as delimiter
            sub_models = opt.sub_model.split("|")

            print("Loading sub models ... ")
            self.n_sub_models = len(sub_models)
            self.sub_type = 'text'

            for i, model_path in enumerate(sub_models):
                checkpoint = torch.load(model_path,
                                        map_location=lambda storage, loc: storage)

                model_opt = checkpoint['opt']
                model_opt = backward_compatible(model_opt)
                if hasattr(model_opt, "enc_not_load_state"):
                    model_opt.enc_not_load_state = True
                    model_opt.dec_not_load_state = True

                dicts = checkpoint['dicts']

                # update special tokens
                onmt.constants = add_tokenidx(model_opt, onmt.constants, dicts)
                # self.bos_token = model_opt.tgt_bos_word

                """"BE CAREFUL: the sub-models might mismatch with the main models in terms of language dict"""
                """"REQUIRE RE-matching"""

                if i == 0:
                    if "src" in checkpoint['dicts']:
                        self.src_dict = checkpoint['dicts']['src']
                if opt.verbose:
                    print('Loading sub-model from %s' % model_path)

                model = build_model (model_opt, checkpoint['dicts'], remove_pretrain=True)
                optimize_model(model)
                model.load_state_dict(checkpoint['model'])

                if model_opt.model in model_list:
                    # if model.decoder.positional_encoder.len_max < self.opt.max_sent_length:
                    #     print("Not enough len to decode. Renewing .. ")
                    #     model.decoder.renew_buffer(self.opt.max_sent_length)
                    model.renew_buffer(self.opt.max_sent_length)

                if opt.fp16:
                    model = model.half()

                if opt.cuda:
                    model = model.cuda()
                else:
                    model = model.cpu()

                if opt.dynamic_quantile == 1:

                    engines = torch.backends.quantized.supported_engines
                    if 'fbgemm' in engines:
                        torch.backends.quantized.engine = 'fbgemm'
                    else:
                        print(
                            "[INFO] fbgemm is not found in the available engines. "
                            " Possibly the CPU does not support AVX2."
                            " It is recommended to disable Quantization (set to 0).")
                        torch.backends.quantized.engine = 'qnnpack'

                    model = torch.quantization.quantize_dynamic(
                        model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
                    )

                model.eval()

                self.sub_models.append(model)
                self.sub_model_types.append(model_opt.model)
        else:
            self.n_sub_models = 0
            self.sub_models = []

        if opt.ensemble_weight:
            ensemble_weight = [float(item) for item in opt.ensemble_weight.split("|")]
            assert len(ensemble_weight) == self.n_models

            if opt.sub_ensemble_weight:
                sub_ensemble_weight = [float(item) for item in opt.sub_ensemble_weight.split("|")]
                assert len(sub_ensemble_weight) == self.n_sub_models
                ensemble_weight = ensemble_weight + sub_ensemble_weight

            total = sum(ensemble_weight)
            self.ensemble_weight = [ item / total for item in ensemble_weight]
        else:
            self.ensemble_weight = None

        # Pretrained Classifier is used for combining classifier and speech models
        if opt.pretrained_classifier:
            self.pretrained_clfs = list()

            # models are string with | as delimiter
            clfs_models = opt.pretrained_classifier.split("|")

            self.n_clfs = len(clfs_models)

            for i, model_path in enumerate(clfs_models):
                checkpoint = torch.load(model_path,
                                        map_location=lambda storage, loc: storage)

                model_opt = checkpoint['opt']
                model_opt = backward_compatible(model_opt)
                clf_dicts = checkpoint['dicts']

                if opt.verbose:
                    print('Loading pretrained classifier from %s' % model_path)

                from onmt.model_factory import build_classifier
                model = build_classifier(model_opt, clf_dicts)
                optimize_model(model)
                model.load_state_dict(checkpoint['model'])

                if opt.fp16:
                    model = model.half()

                if opt.cuda:
                    model = model.cuda()
                else:
                    model = model.cpu()

                if opt.dynamic_quantile == 1:

                    engines = torch.backends.quantized.supported_engines
                    if 'fbgemm' in engines:
                        torch.backends.quantized.engine = 'fbgemm'
                    else:
                        print(
                            "[INFO] fbgemm is not found in the available engines. "
                            " Possibly the CPU does not support AVX2."
                            " It is recommended to disable Quantization (set to 0).")
                        torch.backends.quantized.engine = 'qnnpack'

                    model = torch.quantization.quantize_dynamic(
                        model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
                    )

                model.eval()

                self.pretrained_clfs.append(model)
        else:
            self.n_clfs = 0
            self.pretrained_clfs = list()

        if "mbart-large-50" in opt.external_tokenizer.lower():
            print("[INFO] Using the external MBART50 tokenizer...")

            from transformers import MBart50TokenizerFast
            try:
                self.external_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50",
                                                                               src_lang=opt.src_lang)
            except KeyError as e:
                self.external_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50",
                                                                               src_lang="en_XX")

            try:
                self.tgt_external_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50",
                                                                               src_lang=opt.tgt_lang)
            except KeyError as e:
                self.tgt_external_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50",
                                                                                   src_lang="en_XX")
        elif "m2m100" in opt.external_tokenizer.lower():
            print("[INFO] Using the external %s tokenizer..." % opt.external_tokenizer)
            from transformers import M2M100Tokenizer
            self.external_tokenizer = M2M100Tokenizer.from_pretrained(opt.external_tokenizer, src_lang=opt.src_lang)

            self.tgt_external_tokenizer = M2M100Tokenizer.from_pretrained(opt.external_tokenizer, src_lang=opt.tgt_lang)
        elif "deltalm" in opt.external_tokenizer.lower():
            print("[INFO] Using the external %s tokenizer..." % opt.external_tokenizer)
            lang_list = sorted(list(self.lang_dict.keys()))
            from pretrain_module.tokenization_deltalm import MultilingualDeltaLMTokenizer
            self.external_tokenizer = MultilingualDeltaLMTokenizer.from_pretrained("facebook/mbart-large-50", lang_list=lang_list, src_lang=opt.src_lang)
            self.tgt_external_tokenizer = MultilingualDeltaLMTokenizer.from_pretrained("facebook/mbart-large-50", lang_list=lang_list, src_lang=opt.tgt_lang)
        else:
            self.external_tokenizer = None
            self.tgt_external_tokenizer = None

    def change_language(self, new_src_lang=None, new_tgt_lang=None, use_srclang_as_bos=True):
        if new_src_lang is not None:
            self.src_lang = new_src_lang

        if new_tgt_lang is not None:
            self.tgt_lang = new_tgt_lang

        if use_srclang_as_bos:
            self.bos_token = self.src_lang
            self.bos_id = self.tgt_dict.labelToIdx[self.bos_token]
            print("[INFO] New Bos Token: %s Bos_ID: %d" % (self.bos_token, self.bos_id))

    def translate_batch(self, batches, sub_batches=None, prefix_tokens=None):

        with torch.no_grad():
            return self._translate_batch(batches, sub_batches=sub_batches, prefix_tokens=prefix_tokens)

    def _translate_batch(self, batches, sub_batches, prefix_tokens=None):
        batch = batches[0]
        # Batch size is in different location depending on data.

        beam_size = self.opt.beam_size
        bsz = batch_size = batch.size

        max_len = self.opt.max_sent_length

        gold_scores = batch.get('source').data.new(batch_size).float().zero_()
        gold_words = 0
        allgold_scores = []

        if batch.has_target:
            # Use the first model to decode (also batches[0])
            model_ = self.models[0]

            gold_words, gold_scores, allgold_scores = model_.decode(batch)

        #  (3) Start decoding

        # initialize buffers
        src = batch.get('source')
        scores = src.new(bsz * beam_size, max_len + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = src.new(bsz * beam_size, max_len + 2).long().fill_(self.tgt_pad)
        tokens_buf = tokens.clone()
        tokens[:, 0].fill_(self.tgt_bos)  # first token is bos
        attn, attn_buf = None, None
        nonpad_idxs = None
        src_tokens = src.transpose(0, 1)  # batch x time
        src_lengths = (src_tokens.ne(self.src_eos) & src_tokens.ne(self.src_pad)).long().sum(dim=1)
        blacklist = src_tokens.new_zeros(bsz, beam_size).eq(-1)  # forward and backward-compatible False mask
        if prefix_tokens is not None:
            prefix_tokens = prefix_tokens.to(src.device)

            if bsz == 1:
                prefix_tokens = prefix_tokens.repeat(beam_size, 1)

                for b in range(bsz  * beam_size):
                    for l in range(min(max_len + 2, prefix_tokens.size(1))):
                        tokens[b, l].fill_(prefix_tokens[b, l])

            # In this case, the scores of the prefix positions should be 0

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
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
                return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores):
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
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
            assert not tokens_clone.eq(self.tgt_eos).any()
            tokens_clone[:, step] = self.tgt_eos
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
                unfin_idx = idx // beam_size
                sent = unfin_idx + cum_unfin[unfin_idx]

                sents_seen.add((sent, unfin_idx))

                # if self.match_source_len and step > src_lengths[unfin_idx]:
                #     score = -math.inf

                def get_hypo():

                    if attn_clone is not None:
                        # remove padding tokens from attn scores
                        hypo_attn = attn_clone[i]
                    else:
                        hypo_attn = None

                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'attention': hypo_attn,  # src_len x tgt_len
                        'alignment': None,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())

            newly_finished = []
            for sent, unfin_idx in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfin_idx):
                    finished[sent] = True
                    newly_finished.append(unfin_idx)
            return newly_finished

        reorder_state = None
        batch_idxs = None

        # initialize the decoder state, including:
        # - expanding the context over the batch dimension len_src x (B*beam) x H
        # - expanding the mask over the batch dimension    (B*beam) x len_src

        decoder_states = dict()
        sub_decoder_states = dict()  # for sub-model
        for i in range(self.n_models):
            # if self.opt.pretrained_classifier:
            #     pretrained_layer_states = self.pretrained_clfs[i].encode(batches[i])
            # else:
            #     pretrained_layer_states = None
            pretrained_clf = self.pretrained_clfs[i] if self.opt.pretrained_classifier else None
            decoder_states[i] = self.models[i].create_decoder_state(batches[i], beam_size, type=2,
                                                                    buffering=self.buffering,
                                                                    pretrained_classifier=pretrained_clf)
        if self.opt.sub_model:
            for i in range(self.n_sub_models):
                sub_decoder_states[i] = self.sub_models[i].create_decoder_state(sub_batches[i], beam_size, type=2,
                                                                                buffering=self.buffering)

        if self.dynamic_max_len:
            src_len = src.size(0)
            max_len = math.ceil(int(src_len) * self.dynamic_max_len_scale)

        # Start decoding
        if prefix_tokens is not None:
            if bsz == 1:
                # for this case we run the whole prefix as a preparation step, decoding starts from the last of the prefix
                step  = prefix_tokens.size(1) - 1
            else:
                # in this case we run decoding as usual but filter the output words for prefix
                step = 0
        else:
            step = 0

        # step = 0 if (prefix_tokens is None and bsz == 1) else prefix_tokens.size(1) - 1
        # for step in range(max_len + 1):  # one extra step for EOS marker
        while step < (max_len + 1):
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                for i, model in enumerate(self.models):
                    decoder_states[i]._reorder_incremental_state(reorder_state)
                for i, model in enumerate(self.sub_models):
                    sub_decoder_states[i]._reorder_incremental_state(reorder_state)

            decode_input = tokens[:, :step + 1]

            lprobs, avg_attn_scores = self._decode(decode_input, decoder_states,
                                                   sub_decoder_states=sub_decoder_states)

            avg_attn_scores = None

            lprobs = lprobs.contiguous()

            if self.use_filter:
                # the marked words are 1, so fill the reverse to inf
                lprobs.masked_fill_(~self.filter.unsqueeze(0), -math.inf)
            lprobs[:, self.tgt_pad] = -math.inf  # never select pad

            # handle min and max length constraints

            if step >= max_len:
                lprobs[:, :self.tgt_eos] = -math.inf
                lprobs[:, self.tgt_eos + 1:] = -math.inf
            elif step < self.min_len:
                lprobs[:, self.tgt_eos] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            # here prefix tokens is a list of word-ids
            if prefix_tokens is not None and bsz > 1:
                if step < prefix_tokens.size(1) and step < max_len:
                    prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
                    prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
                    prefix_mask = prefix_toks.ne(self.tgt_pad)
                    # originally infinity here, this number can return nan so thats quite dangerous
                    # put a large negative number here is better
                    lprobs[prefix_mask] = torch.tensor(-21111993).to(lprobs)

                    lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
                        -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
                    )
                    # lprobs[prefix_mask].scatter_()

                    # if prefix includes eos, then we should make sure tokens and
                    # scores are the same across all beams
                    eos_mask = prefix_toks.eq(self.tgt_eos)
                    if eos_mask.any():
                        # validate that the first beam matches the prefix
                        first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[:, 0, 1:step + 1]
                        eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
                        target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
                        assert (first_beam == target_prefix).all()

                        def replicate_first_beam(tensor, mask):
                            tensor = tensor.view(-1, beam_size, tensor.size(-1))
                            tensor[mask] = tensor[mask][:, :1, :]
                            return tensor.view(-1, tensor.size(-1))

                        # copy tokens, scores and lprobs from the first beam to all beams
                        tokens = replicate_first_beam(tokens, eos_mask_batch_dim)
                        scores = replicate_first_beam(scores, eos_mask_batch_dim)
                        lprobs = replicate_first_beam(lprobs, eos_mask_batch_dim)
                    else:
                        # force tgt_eos to not appear
                        lprobs[:, self.tgt_eos] = -math.inf

            if self.no_repeat_ngram_size > 0:
                # for each beam and batch sentence, generate a list of previous ngrams
                gen_ngrams = [{} for bbsz_idx in range(bsz * beam_size)]
                for bbsz_idx in range(bsz * beam_size):
                    gen_tokens = tokens[bbsz_idx].tolist()
                    for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                        gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
                            gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

            # Record attention scores
            if avg_attn_scores is not None:
                if attn is None:
                    attn = scores.new(bsz * beam_size, src_tokens.size(1), max_len + 2)
                    attn_buf = attn.clone()
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            scores_buf = scores_buf.type_as(lprobs)
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)

            if self.no_repeat_ngram_size > 0:
                def calculate_banned_tokens(bbsz_idx):
                    # before decoding the next token, prevent decoding of ngrams that have already appeared
                    ngram_index = tuple(tokens[bbsz_idx, step + 2 - self.no_repeat_ngram_size:step + 1].tolist())
                    return gen_ngrams[bbsz_idx].get(ngram_index, [])

                if step + 2 - self.no_repeat_ngram_size >= 0:
                    # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
                    banned_tokens = [calculate_banned_tokens(bbsz_idx) for bbsz_idx in range(bsz * beam_size)]
                else:
                    banned_tokens = [[] for bbsz_idx in range(bsz * beam_size)]

                for bbsz_idx in range(bsz * beam_size):
                    lprobs[bbsz_idx, banned_tokens[bbsz_idx]] = -math.inf

            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos (except for blacklisted ones)
            eos_mask = cand_indices.eq(self.tgt_eos)
            eos_mask[:, :beam_size][blacklist] = 0

            # only consider eos when it's among the top beam_size indices
            torch.masked_select(
                cand_bbsz_idx[:, :beam_size],
                mask=eos_mask[:, :beam_size],
                out=eos_bbsz_idx.resize_(0),
            )

            finalized_sents = set()
            if eos_bbsz_idx.numel() > 0:
                torch.masked_select(
                    cand_scores[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_scores.resize_(0),
                )
                finalized_sents = finalize_hypos(step, eos_bbsz_idx, eos_scores)
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            # assert step < max_len
            if len(finalized_sents) > 0:

                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = cand_indices.new_ones(bsz)
                batch_mask[cand_indices.new(finalized_sents)] = 0
                batch_idxs = batch_mask.nonzero(as_tuple=False).squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                if prefix_tokens is not None and bsz > 1:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                blacklist = blacklist[batch_idxs]

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

            # Set active_mask so that values > cand_size indicate eos or
            # blacklisted hypos and values < cand_size indicate candidate
            # active hypos. After this, the min values per row are the top
            # candidate active hypos.

            active_mask = buffer('active_mask')
            eos_mask[:, :beam_size] |= blacklist
            torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_mask.size(1)],
                out=active_mask.resize_(0),
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, new_blacklist = buffer('active_hypos'), buffer('new_blacklist')
            torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,
                out=(new_blacklist.resize_(0), active_hypos.resize_(0))
            )

            # update blacklist to ignore any finalized hypos
            blacklist = new_blacklist.ge(cand_size)[:, :beam_size]
            assert (~blacklist).any(dim=1).all()

            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx.resize_(0),
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

            step = step + 1

        # sort by score descending
        for sent in range(len(finalized)):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)

        return finalized, gold_scores, gold_words, allgold_scores

    def _decode(self, tokens, decoder_states, sub_decoder_states=None):

        # require batch first for everything
        outs = dict()
        attns = dict()

        for i in range(self.n_models):
            # decoder output contains the log-prob distribution of the next step
            decoder_output = self.models[i].step(tokens, decoder_states[i])

            outs[i] = decoder_output['log_prob']
            attns[i] = decoder_output['coverage']

        for j in range(self.n_sub_models):
            sub_decoder_output = self.sub_models[j].step(tokens, sub_decoder_states[j])
            outs[self.n_models + j] = sub_decoder_output['log_prob']

        out = self._combine_outputs(outs, weight=self.ensemble_weight)
        # attn = self._combine_attention(attns)

        if self.vocab_size > out.size(-1):
            self.vocab_size = out.size(-1)  # what the hell ?
        # attn = attn[:, -1, :] # I dont know what this line does
        attn = None  # attn is never used in decoding probably

        return out, attn

    def build_prefix(self, prefixes, bsz=None):
        """
        :param bsz:
        :param prefixes: List of strings
        :return:
        """
        if self.external_tokenizer is None:
            prefix_data = [self.tgt_dict.convertToIdx(sent.split(),
                                                      onmt.constants.UNK_WORD)
                            for sent in prefixes]
        else:
            # move the last element which is <eos>
            _prefix_data = [torch.LongTensor(self.external_tokenizer(sent)['input_ids'][:-1])
                            for sent in prefixes]

            prefix_data = _prefix_data

            for prefix_tensor in prefix_data:
                prefix_tensor[0] = self.bos_id

                # _listed_tensor = prefix_tensor.tolist()
                # if _listed_tensor[0] == self.tgt_bos:
                #     _listed_tensor = _listed_tensor[1:]
                # if _listed_tensor[0] == self.tgt_eos:
                #     _listed_tensor = _listed_tensor[:-1]
                # prefix_data.append(torch.LongTensor(_listed_tensor))

        # clone the same prefix for multiple sentences
        if len(prefix_data) == 1 and bsz > 1:
            prefix_data = prefix_data * bsz

        # collate into the same tensor with padding
        lengths = [x.size(0) for x in prefix_data]
        max_length = max(lengths)

        tensor = prefix_data[0].new(len(prefix_data), max_length).fill_(self.tgt_pad)

        for i in range(len(prefix_data)):
            data_length = prefix_data[i].size(0)
            offset = 0
            tensor[i].narrow(0, offset, data_length).copy_(prefix_data[i])

        return tensor

    # override the "build_data" from parent Translator
    def build_data(self, src_sents, tgt_sents, type='mt', past_sents=None):
        # This needs to be the same as preprocess.py.
        data_type = 'text'

        if type == 'mt':

            if self.external_tokenizer is None:
                # TODO: add external tokenizer
                if self.start_with_bos:
                    src_data = [self.src_dict.convertToIdx(b,
                                                           onmt.constants.UNK_WORD,
                                                           onmt.constants.BOS_WORD)
                                for b in src_sents]
                else:
                    src_data = [self.src_dict.convertToIdx(b,
                                                           onmt.constants.UNK_WORD)
                                for b in src_sents]

                if past_sents is not None:
                    if self.start_with_bos:
                        past_src_data = [self.src_dict.convertToIdx(b,
                                                                    onmt.constants.UNK_WORD,
                                                                    onmt.constants.BOS_WORD)
                                         for b in past_sents]
                    else:
                        past_src_data = [self.src_dict.convertToIdx(b,
                                                                    onmt.constants.UNK_WORD)
                                         for b in past_sents]
                else:
                    past_src_data = None
            else:
                src_data = [torch.LongTensor(self.external_tokenizer(" ".join(b))['input_ids'])
                            for b in src_sents]

                if past_sents is not None:
                    past_src_data = [torch.LongTensor(self.external_tokenizer(" ".join(b))['input_ids'])
                            for b in past_src_data]
                else:
                    past_src_data = None

        elif type == 'asr':
            # no need to deal with this
            src_data = src_sents
            past_src_data = past_sents
            data_type = 'audio'
        elif type == 'asr_wav':
            from onmt.data.wav_dataset import WavDataset
            src_data = src_sents
            past_src_data = past_sents
            data_type = 'wav'
        else:
            raise NotImplementedError

        tgt_bos_word = self.opt.bos_token
        if self.opt.no_bos_gold:
            tgt_bos_word = None
        tgt_data = None
        if tgt_sents:
            if self.tgt_external_tokenizer is not None:
                tgt_data = [torch.LongTensor(self.tgt_external_tokenizer(" ".join(b))['input_ids'])
                            for b in tgt_sents]
            else:
                tgt_data = [self.tgt_dict.convertToIdx(b,
                                                       onmt.constants.UNK_WORD,
                                                       tgt_bos_word,
                                                       onmt.constants.EOS_WORD) for b in tgt_sents]

        if self.src_lang in self.lang_dict:
            src_lang_data = [torch.Tensor([self.lang_dict[self.src_lang]])]
        else:
            src_lang_data = [torch.Tensor([0])]

        if self.tgt_lang in self.lang_dict:
            tgt_lang_data = [torch.Tensor([self.lang_dict[self.tgt_lang]])]
        else:
            tgt_lang_data = [torch.Tensor([0])]

        try:
            src_atb = self.opt.src_atb
            if src_atb in self.atb_dict:
                src_atb_data = [torch.Tensor([self.atb_dict[src_atb]])]
            else:
                src_atb_data = None
        except AttributeError:
            src_atb_data = None

        try:
            tgt_atb = self.opt.tgt_atb

            if tgt_atb in self.atb_dict:
                tgt_atb_data = [torch.Tensor([self.atb_dict[tgt_atb]])]
            else:
                tgt_atb_data = None
        except AttributeError:
            tgt_atb_data = None


        return onmt.Dataset(src_data, tgt_data,
                            src_langs=src_lang_data, tgt_langs=tgt_lang_data,
                            src_atbs=src_atb_data, tgt_atbs=tgt_atb_data,
                            batch_size_words=sys.maxsize,
                            batch_size_frames=sys.maxsize,
                            cut_off_size=sys.maxsize,
                            smallest_batch_size=sys.maxsize,
                            max_src_len=sys.maxsize,
                            data_type=data_type,
                            batch_size_sents=sys.maxsize,
                            src_align_right=self.opt.src_align_right,
                            past_src_data=past_src_data)

    def translate(self, src_data, tgt_data, past_src_data=None, sub_src_data=None, type='mt', prefix=None):

        if past_src_data is None or len(past_src_data) == 0:
            past_src_data = None

        #  (1) convert words to indexes
        if isinstance(src_data[0], list) and type in ['asr', 'asr_wav']:
            batches = list()
            for i, src_data_ in enumerate(src_data):
                if past_src_data is not None:
                    past_src_data_ = past_src_data[i]
                else:
                    past_src_data_ = None
                dataset = self.build_data(src_data_, tgt_data, type=type, past_sents=past_src_data_)
                batch = dataset.get_batch(0)
                batches.append(batch)
        else:
            dataset = self.build_data(src_data, tgt_data, type=type, past_sents=past_src_data)
            batch = dataset.get_batch(0)  # this dataset has only one mini-batch
            batches = [batch] * self.n_models
            src_data = [src_data] * self.n_models

        if sub_src_data is not None and len(sub_src_data) > 0:
            sub_dataset = self.build_data(sub_src_data, tgt_data, type='mt')
            sub_batch = sub_dataset.get_batch(0)
            sub_batches = [sub_batch] * self.n_sub_models
            sub_src_data = [sub_src_data] * self.n_sub_models
        else:
            sub_batches, sub_src_data = None, None

        batch_size = batches[0].size
        if self.cuda:
            for i, _ in enumerate(batches):
                batches[i].cuda(fp16=self.fp16)
            if sub_batches:
                for i, _ in enumerate(sub_batches):
                    sub_batches[i].cuda(fp16=self.fp16)

        if prefix is not None:
            prefix_tensor = self.build_prefix(prefix, bsz=batch_size)
        else:
            prefix_tensor = None

        #  (2) translate
        #  each model in the ensemble uses one batch in batches
        finalized, gold_score, gold_words, allgold_words = self.translate_batch(batches, sub_batches=sub_batches,
                                                                                prefix_tokens=prefix_tensor)
        pred_length = []

        #  (3) convert indexes to words
        pred_batch = []
        pred_ids = []
        src_data = src_data[0]
        for b in range(batch_size):

            # probably when the src is empty so beam search stops immediately
            if len(finalized[b]) == 0:
                # assert len(src_data[b]) == 0, "The target search result is empty, assuming that the source is empty."
                pred_batch.append(
                    [self.build_target_tokens([], src_data[b], None)
                     for n in range(self.opt.n_best)]
                )
                pred_ids.append([[] for n in range(self.opt.n_best)])
            else:
                pred_batch.append(
                    [self.build_target_tokens(finalized[b][n]['tokens'], src_data[b], None)
                     for n in range(self.opt.n_best)]
                )
                pred_ids.append([finalized[b][n]['tokens'] for n in range(self.opt.n_best)])
        pred_score = []
        for b in range(batch_size):
            if len(finalized[b]) == 0:
                pred_score.append(
                    [torch.FloatTensor([0])
                     for n in range(self.opt.n_best)]
                )
            else:
                pred_score.append(
                    [torch.FloatTensor([finalized[b][n]['score']])
                     for n in range(self.opt.n_best)]
                )

        return pred_batch, pred_ids, pred_score, pred_length, gold_score, gold_words, allgold_words
