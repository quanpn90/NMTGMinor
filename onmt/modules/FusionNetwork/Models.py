import numpy as np
import torch, math
import torch.nn as nn
from onmt.modules.BaseModel import DecoderState
from onmt.modules.Transformer.Models import TransformerDecodingState
from collections import defaultdict
import torch.nn.functional as F


class FusionNetwork(nn.Module):
    """Main model in 'Attention is all you need' """

    def __init__(self, tm_model, lm_model):
        super(FusionNetwork, self).__init__()
        self.tm_model = tm_model
        self.lm_model = lm_model

        # freezing the parameters for the language model
        for param in self.lm_model.parameters():
            param.requires_grad = False

    def forward(self, batch):
        """
        Inputs Shapes:
            src: len_src x batch_size
            tgt: len_tgt x batch_size

        Outputs Shapes:
            out:      batch_size*len_tgt x model_size


        """

        nmt_output_dict = self.tm_model(batch)

        # no gradient for the LM side
        with torch.no_grad():
            lm_output_dict = self.lm_model(batch)

        output_dict = defaultdict(lambda: None)

        output_dict['tm'] = nmt_output_dict
        output_dict['lm'] = lm_output_dict

        return output_dict

    # an utility function to fuse two states
    # return log prob
    def fuse_states(self, tm_state, lm_state):

        # PRENORM algorithm
        # (1) generate the log P_lm
        with torch.no_grad():
            log_lm = self.lm_model.generator[0](lm_state, log_softmax=True)

        # (2) generate the logits for tm
        tm_logits = self.tm_model.generator[0](tm_state, log_softmax=False)

        # (3) add the bias of lm to the logits
        dists = F.log_softmax(tm_logits + log_lm, dim=-1)

        # ## POSTNORM
        # # (1) generate the P_lm
        # with torch.no_grad():
        #     lm_logits = self.lm_model.generator[0](lm_state, log_softmax=False)
        #
        # # (2) generate the logits for tm
        # tm_logits = self.tm_model.generator[0](tm_state, log_softmax=False)
        #
        # dists = F.log_softmax(F.softmax(tm_logits, dim=-1) * F.softmax(lm_logits, dim=-1), dim=-1)

        return dists

    def renew_buffer(self, new_len):
        self.tm_model.decoder.renew_buffer(new_len)
        self.lm_model.decoder.renew_buffer(new_len)

    def decode(self, batch):
        """
        :param batch: (onmt.Dataset.Batch) an object containing tensors needed for training
        :return: gold_scores (torch.Tensor) log probs for each sentence
                 gold_words  (Int) the total number of non-padded tokens
                 allgold_scores (list of Tensors) log probs for each word in the sentence
        """

        src = batch.get('source')
        tgt_input = batch.get('target_input')
        tgt_output = batch.get('target_output')

        # transpose to have batch first
        src = src.transpose(0, 1)
        tgt_input = tgt_input.transpose(0, 1)
        batch_size = tgt_input.size(0)

        # (1) we decode using language model
        context = self.tm_model.encoder(src)['context']

        if (hasattr(self,
            'autoencoder') and self.autoencoder and self.autoencoder.representation == "EncoderHiddenState"):
            context = self.autoencoder.autocode(context)

        decoder_output = self.tm_model.decoder(tgt_input, context, src)['hidden']

        output = decoder_output

        if (hasattr(self, 'autoencoder')
                     and self.autoencoder and self.autoencoder.representation == "DecoderHiddenState"):
            output = self.autoencoder.autocode(output)

        gold_scores = context.new(batch_size).zero_()
        gold_words = 0
        allgold_scores = list()

        # (2) decode using the language model
        lm_decoder_output = self.lm_model.decoder(tgt_input)['hidden']

        for dec_t, lm_t, tgt_t in zip(decoder_output, lm_decoder_output, tgt_output):

            # generate the current step distribution from both states
            gen_t = self.fuse_states(dec_t, lm_t)
            tgt_t = tgt_t.unsqueeze(1)
            scores = gen_t.gather(1, tgt_t)
            scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
            gold_scores += scores.squeeze(1).type_as(gold_scores)
            gold_words += tgt_t.ne(onmt.Constants.PAD).sum().item()
            allgold_scores.append(scores.squeeze(1).type_as(gold_scores))

        return gold_words, gold_scores, allgold_scores

    def step(self, input_t, decoder_state):
        """
        Decoding function:
        generate new decoder output based on the current input and current decoder state
        the decoder state is updated in the process
        :param input_t: the input word index at time t
        :param decoder_state: object FusionDecoderState containing the buffers required for decoding
        :return: a dictionary containing: log-prob output and the attention coverage
        """

        # (1) decode using the translation model
        tm_hidden, coverage = self.tm_model.decoder.step(input_t, decoder_state.tm_state)

        # (2) decode using the translation model
        lm_hidden, ________ = self.lm_model.decoder.step(input_t, decoder_state.lm_state)

        log_prob = self.fuse_states(tm_hidden, lm_hidden)
        # log_prob = self.tm_model.generator[0](tm_hidden)

        last_coverage = coverage[:, -1, :].squeeze(1)

        output_dict = defaultdict(lambda: None)

        output_dict['log_prob'] = log_prob
        output_dict['coverage'] = last_coverage

        return output_dict

    def create_decoder_state(self, batch, beam_size=1):
        """
        Generate a new decoder state based on the batch input
        :param batch: Batch object (may not contain target during decoding)
        :param beam_size: Size of beam used in beam search
        :return:
        """
        tm_decoder_state = self.tm_model.create_decoder_state(batch, beam_size=beam_size)

        lm_decoder_state = self.lm_model.create_decoder_state(batch, beam_size=beam_size)

        decoder_state = FusionDecodingState(tm_decoder_state, lm_decoder_state)

        return decoder_state


class FusionDecodingState(DecoderState):

    def __init__(self, tm_state, lm_state):

        self.tm_state = tm_state
        self.lm_state = lm_state

        self.original_src = tm_state.original_src
        self.beam_size = tm_state.beam_size

    def update_beam(self, beam, b, remaining_sents, idx):

        self.tm_state.update_beam(beam, b, remaining_sents, idx)
        self.lm_state.update_beam(beam, b, remaining_sents, idx)

    # in this section, the sentences that are still active are
    # compacted so that the decoder is not run on completed sentences
    def prune_complete_beam(self, active_idx, remaining_sents):

        self.tm_state.prune_complete_beam(active_idx, remaining_sents)
        self.lm_state.prune_complete_beam(active_idx, remaining_sents)


