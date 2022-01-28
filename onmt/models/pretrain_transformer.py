import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.checkpoint import checkpoint

import onmt
from onmt.modules.base_seq2seq import NMTModel, Reconstructor, DecoderState
from onmt.models.transformers import TransformerDecodingState

torch_version = float(torch.__version__[:3])


class PretrainTransformer(NMTModel):
    """Main model in 'Attention is all you need' """

    def __init__(self, encoder, decoder, generator=None, rec_decoder=None, rec_generator=None,
                 mirror=False, ctc=False):
        super().__init__(encoder, decoder, generator, rec_decoder, rec_generator, ctc=ctc)
        if hasattr(decoder, 'dec_pretrained_model') and decoder.dec_pretrained_model:
            self.model_size = self.decoder.config.bert_hidden_size
            self.tgt_vocab_size = self.decoder.config.vocab_size
            self.switchout = 0
        else:
            self.model_size = self.decoder.model_size
            self.tgt_vocab_size = self.decoder.word_lut.weight.size(0)
            self.switchout = self.decoder.switchout

        if self.encoder.input_type == 'text':
            if hasattr(encoder, 'enc_pretrained_model') and encoder.enc_pretrained_model:
                self.src_vocab_size = self.encoder.config.vocab_size
            else:
                self.src_vocab_size = self.encoder.word_lut.weight.size(0)
        else:
            self.src_vocab_size = 0

        if mirror:
            self.mirror_decoder = copy.deepcopy(self.decoder)
            self.mirror_g = nn.Linear(decoder.model_size, decoder.model_size)
            self.mirror_generator = copy.deepcopy(self.generator)
            self.mirror_generator[0].linear.weight = self.decoder.word_lut.weight

        if self.reconstruct:
            self.rec_linear = nn.Linear(decoder.model_size, decoder.model_size)

        if self.ctc:
            self.ctc_linear = nn.Linear(encoder.model_size, self.tgt_vocab_size)

    def reset_states(self):
        return

    def forward(self, batch, target_mask=None, streaming=False, zero_encoder=False,
                mirror=False, streaming_state=None, nce=False, **kwargs):
        """
        :param nce: use noise contrastive estimation
        :param streaming_state:
        :param streaming:
        :param mirror: if using mirror network for future anticipation
        :param batch: data object sent from the dataset
        :param target_mask:
        :param zero_encoder: zero out the encoder output (if necessary)
        :return:
        """
        if self.switchout > 0 and self.training:
            batch.switchout(self.switchout, self.src_vocab_size, self.tgt_vocab_size)

        src = batch.get('source')
        tgt = batch.get('target_input')
        src_pos = batch.get('source_pos')
        tgt_pos = batch.get('target_pos')
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')
        src_lengths = batch.src_lengths
        tgt_lengths = batch.tgt_lengths

        org_src = src
        org_tgt = tgt
        src = src.transpose(0, 1)  # transpose to have batch first
        tgt = tgt.transpose(0, 1)

        src_attention_mask = src.ne(onmt.constants.SRC_PAD).long()  # [b, src_len]
        if hasattr(self.encoder, 'enc_pretrained_model') and self.encoder.enc_pretrained_model in ["bert", "roberta"]:
            segments_tensor = src.ne(onmt.constants.SRC_PAD).long()
            enc_outputs = self.encoder(src, src_attention_mask, segments_tensor)  # the encoder is a pretrained model
            context = enc_outputs[0]
            encoder_output = defaultdict(lambda: None)
            encoder_output['context'] = context
            encoder_output['src_attention_mask'] = src_attention_mask
            encoder_output['streaming_state'] = None
        if hasattr(self.encoder, 'enc_pretrained_model') and \
                self.encoder.enc_pretrained_model in ["mbart", "mbart50", "m2m", "m2m100"]:
            # src_attention_mask = src.ne(onmt.constants.SRC_PAD).long()
            src_attention_mask = batch.get("src_selfattn_mask")
            enc_outputs = self.encoder(src, src_attention_mask)  # the encoder is a pretrained model
            context = enc_outputs[0]
            context = context  # .transpose(0, 1).contiguous()
            encoder_output = defaultdict(lambda: None)
            encoder_output['context'] = context
            encoder_output['src_attention_mask'] = src_attention_mask
            encoder_output['streaming_state'] = None

        else:
            encoder_output = self.encoder(src, input_pos=src_pos, input_lang=src_lang, streaming=streaming,
                                          src_lengths=src_lengths, streaming_state=streaming_state)

            encoder_output = defaultdict(lambda: None, encoder_output)
            context = encoder_output['context']
            context = context.transpose(0, 1)  # to make it consistent with bert batch first
            # the state is changed
            streaming_state = encoder_output['streaming_state']

        # DECODER PART

        if hasattr(self.decoder, 'dec_pretrained_model') and self.decoder.dec_pretrained_model in ["bert", "roberta"]:
            # src: [b, src_l]  context: [b, src_l, de_model]
            tgt_token_type = tgt.ne(onmt.constants.TGT_PAD).long()  # [bsz, len]
            tgt_attention_mask = tgt.ne(onmt.constants.TGT_PAD).long()  # [bsz, len]
            decoder_output = self.decoder(input_ids=tgt,
                                          attention_mask=tgt_attention_mask,
                                          token_type_ids=tgt_token_type,
                                          encoder_hidden_states=context,
                                          encoder_attention_mask=src_attention_mask,
                                          )

            decoder_output = decoder_output[0]
            output = decoder_output.transpose(0, 1)  # [bsz, tgt_len, d] => [tgt_len, bsz, d]
            output_dict = defaultdict(lambda: None)
            context = context.transpose(0, 1)  # to [src_l, b, de_model]
        elif hasattr(self.decoder, 'dec_pretrained_model') and self.decoder.dec_pretrained_model in \
                ["mbart", "mbart50", "m2m", "m2m100"]:
            # print("HELLO DECODER")
            # src: [b, src_l]  context: [b, src_l, de_model]
            # src_attention_mask = src.eq(onmt.constants.SRC_PAD).long()
            # src_attention_mask = (1 - src_attention_mask.long())
            tgt_attention_mask = tgt.ne(onmt.constants.TGT_PAD).long()  # [bsz, len]
            decoder_output = self.decoder(input_ids=tgt,
                                          attention_mask=tgt_attention_mask,
                                          encoder_hidden_states=context,
                                          encoder_attention_mask=src_attention_mask,
                                          )

            decoder_output = decoder_output[0]
            # output = decoder_output
            output = decoder_output  # .transpose(0, 1)  # [bsz, tgt_len, d] => [tgt_len, bsz, d]
            output_dict = defaultdict(lambda: None)
            # context = context.transpose(0, 1)  # to [src_l, b, de_model]
        else:
            context = context.transpose(0, 1)  # to  [src_l, b, de_model] src: [b, l]
            decoder_output = self.decoder(tgt, context, src,
                                          src_lang=src_lang, tgt_lang=tgt_lang,
                                          input_pos=tgt_pos, streaming=streaming,
                                          src_lengths=src_lengths, tgt_lengths=tgt_lengths,
                                          streaming_state=streaming_state)

            # update the streaming state again
            decoder_output = defaultdict(lambda: None, decoder_output)
            streaming_state = decoder_output['streaming_state']
            output = decoder_output['hidden']  # [tgt_len, bsz, d]

            # build the output dict based on decoder output
            output_dict = defaultdict(lambda: None, decoder_output)

        output_dict['hidden'] = output  # [tgt_len, bsz, d]
        output_dict['context'] = context  # [b, l, de_model]
        output_dict['src_mask'] = encoder_output['src_attention_mask']  # [b, l, de_model]
        output_dict['src'] = src
        output_dict['target_mask'] = target_mask
        output_dict['streaming_state'] = streaming_state
        output_dict['target'] = batch.get('target_output')

        # final layer: computing softmax
        if self.training and nce:
            output_dict = self.generator[0](output_dict)
        else:
            logprobs = self.generator[0](output_dict)['logits']
            output_dict['logprobs'] = logprobs

        # Mirror network: reverse the target sequence and perform backward language model
        if mirror:
            # tgt_reverse = torch.flip(batch.get('target_input'), (0, ))
            tgt_pos = torch.flip(batch.get('target_pos'), (0,))
            tgt_reverse = torch.flip(batch.get('target'), (0,))
            tgt_reverse_input = tgt_reverse[:-1]
            tgt_reverse_output = tgt_reverse[1:]

            tgt_reverse_input = tgt_reverse_input.transpose(0, 1)
            # perform an additional backward pass
            reverse_decoder_output = self.mirror_decoder(tgt_reverse_input, context, src, src_lang=src_lang,
                                                         tgt_lang=tgt_lang, input_pos=tgt_pos)

            reverse_decoder_output['src'] = src
            reverse_decoder_output['context'] = context
            reverse_decoder_output['target_mask'] = target_mask

            reverse_logprobs = self.mirror_generator[0](reverse_decoder_output)['logits']

            output_dict['reverse_target'] = tgt_reverse_output
            output_dict['reverse_hidden'] = reverse_decoder_output['hidden']
            output_dict['reverse_logprobs'] = reverse_logprobs
            output_dict['target_input'] = batch.get('target_input')
            output_dict['target_lengths'] = batch.tgt_lengths

            # learn weights for mapping (g in the paper)
            output_dict['hidden'] = self.mirror_g(output_dict['hidden'])

        # Reconstruction network
        if self.reconstruct:
            bos = org_tgt[0].unsqueeze(0)  # 1 x B
            src_input = torch.cat([bos, org_src[:-1]], dim=0)  # T x B
            src_output = org_src

            src_input = src_input.transpose(0, 1)
            rec_context = self.rec_linear(output_dict['hidden'])  # T x B x H
            rec_decoder_output = self.rec_decoder(src_input, rec_context, tgt, tgt_lang=src_lang, input_pos=src_pos)
            rec_output = rec_decoder_output['hidden']
            rec_logprobs = self.rec_generator[0](rec_decoder_output)['logits']

            output_dict['rec_logprobs'] = rec_logprobs
            output_dict['rec_hidden'] = rec_output
            output_dict['reconstruct'] = True
            output_dict['rec_target'] = src_output
        else:
            output_dict['reconstruct'] = False

        # compute the logits for each encoder step
        if self.ctc:
            output_dict['encoder_logits'] = self.ctc_linear(output_dict['context'])

        return output_dict

    def decode(self, batch):
        """
        :param batch: (onmt.Dataset.Batch) an object containing tensors needed for training
        :return: gold_scores (torch.Tensor) log probs for each sentence
                 gold_words  (Int) the total number of non-padded tokens
                 allgold_scores (list of Tensors) log probs for each word in the sentence
        """

        src = batch.get('source')
        src_pos = batch.get('source_pos')
        tgt_input = batch.get('target_input')
        tgt_output = batch.get('target_output')
        tgt_pos = batch.get('target_pos')
        # tgt_atb = batch.get('target_atb')  # a dictionary of attributes
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')

        # transpose to have batch first
        src = src.transpose(0, 1)
        tgt_input = tgt_input.transpose(0, 1)
        batch_size = tgt_input.size(0)

        context = self.encoder(src, input_pos=src_pos, input_lang=src_lang)['context']

        if hasattr(self, 'autoencoder') and self.autoencoder \
                and self.autoencoder.representation == "EncoderHiddenState":
            context = self.autoencoder.autocode(context)

        gold_scores = context.new(batch_size).zero_()
        gold_words = 0
        allgold_scores = list()
        decoder_output = self.decoder(tgt_input, context, src, tgt_lang=tgt_lang, src_lang=src_lang,
                                      input_pos=tgt_pos)['hidden']

        output = decoder_output

        if hasattr(self, 'autoencoder') and self.autoencoder and \
                self.autoencoder.representation == "DecoderHiddenState":
            output = self.autoencoder.autocode(output)

        for dec_t, tgt_t in zip(output, tgt_output):

            dec_out = defaultdict(lambda: None)
            dec_out['hidden'] = dec_t.unsqueeze(0)
            dec_out['src'] = src
            dec_out['context'] = context

            if isinstance(self.generator, nn.ModuleList):
                gen_t = self.generator[0](dec_out)['logits']
            else:
                gen_t = self.generator(dec_out)['logits']
            gen_t = F.log_softmax(gen_t, dim=-1, dtype=torch.float32)
            gen_t = gen_t.squeeze(0)
            tgt_t = tgt_t.unsqueeze(1)
            scores = gen_t.gather(1, tgt_t)
            scores.masked_fill_(tgt_t.eq(onmt.constants.TGT_PAD), 0)
            gold_scores += scores.squeeze(1).type_as(gold_scores)
            gold_words += tgt_t.ne(onmt.constants.TGT_PAD).sum().item()
            allgold_scores.append(scores.squeeze(1).type_as(gold_scores))

        return gold_words, gold_scores, allgold_scores

    def renew_buffer(self, new_len):
        self.decoder.renew_buffer(new_len)

    def step(self, input_t, decoder_state, streaming=False):
        """
        Decoding function:
        generate new decoder output based on the current input and current decoder state
        the decoder state is updated in the process
        :param streaming:
        :param input_t: the input word index from time 0 to time t
        :param decoder_state: object DecoderState containing the buffers required for decoding
        :return: a dictionary containing: log-prob output and the attention coverage
        """

        output_dict = self.decoder.step(input_t, decoder_state, streaming=streaming)
        output_dict['src'] = decoder_state.src.transpose(0, 1)

        # squeeze to remove the time step dimension
        log_prob = self.generator[0](output_dict)['logits'].squeeze(0)
        log_prob = F.log_softmax(log_prob, dim=-1, dtype=torch.float32)  # [beam*b, 1, vocab_size]
        output_dict['log_prob'] = log_prob.squeeze(1)

        # Currently attention score is not returned
        # coverage = output_dict['coverage']
        # last_coverage = coverage[:, -1, :].squeeze(1)
        # output_dict['coverage'] = last_coverage

        return output_dict

    def create_decoder_state(self, batch, beam_size=1, type=1, buffering=True, **kwargs):
        """
        Generate a new decoder state based on the batch input
        :param buffering:
        :param streaming:
        :param type:
        :param batch: Batch object (may not contain target during decoding)
        :param beam_size: Size of beam used in beam search
        :return:
        """
        src = batch.get('source')
        src_pos = batch.get('source_pos')
        tgt_atb = batch.get('target_atb')
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')

        src_transposed = src.transpose(0, 1)  # [batch_size, src_len]

        if not self.encoder.enc_pretrained_model:
            encoder_output = self.encoder(src_transposed, input_pos=src_pos, input_lang=src_lang)
        elif self.encoder.enc_pretrained_model in ['bert', 'roberta']:
            segments_tensor = src_transposed.ne(onmt.constants.SRC_PAD).long()  # [batch_size, src_len]
            src_attention_mask = segments_tensor

            enc_outputs = self.encoder(src_transposed, src_attention_mask, segments_tensor)

            context = enc_outputs[0]  # [batch_size , len, hidden]
            context = context.transpose(0, 1)  # [len, batch_size, hidden]
            encoder_output = defaultdict(lambda: None)
            encoder_output["context"] = context
        elif self.encoder.enc_pretrained_model in ['mbart', 'mbart50']:
            src_attention_mask = batch.get("src_selfattn_mask")
            enc_outputs = self.encoder(src_transposed, src_attention_mask)
            context = enc_outputs[0]
            encoder_output = defaultdict(lambda: None)
            encoder_output["context"] = context
        else:
            print("Warning: unknown enc_pretrained_model")
            raise NotImplementedError

        dec_pretrained_model = self.decoder.dec_pretrained_model
        if not dec_pretrained_model:
            mask_src = None
        elif dec_pretrained_model in["bert", "roberta"]:
            mask_src = src_transposed.ne(onmt.constants.SRC_PAD).unsqueeze(1)  # batch_size  x 1 x len_src for broadcasting
        elif dec_pretrained_model in["mbart", "mbart50"]:
            mask_src = src_attention_mask  # batch_size  x 1 x len_src for broadcasting
        else:
            print("Warning: unknown dec_pretrained_model")
            raise NotImplementedError

        decoder_state = TransformerDecodingState(src, tgt_lang, encoder_output['context'], src_lang,
                                                 beam_size=beam_size, model_size=self.model_size,
                                                 type=type, buffering=buffering, src_mask=mask_src,
                                                 dec_pretrained_model=self.decoder.dec_pretrained_model)

        return decoder_state

    def init_stream(self):

        pass

    def set_memory_size(self, src_memory_size, tgt_memory_size):

        pass

    def tie_weights(self):
        assert self.generator is not None, "The generator needs to be created before sharing weights"
        if hasattr(self.decoder, 'dec_pretrained_model') and self.decoder.dec_pretrained_model in ["bert", "roberta"]:
            self.generator[0].linear.weight = self.decoder.embeddings.word_embeddings.weight
        if hasattr(self.decoder, 'dec_pretrained_model') and self.decoder.dec_pretrained_model \
                in ["mbart", "mbart50", "m2m", "m2m100"]:
            self.generator[0].linear.weight = self.decoder.embed_tokens.weight
        else:
            self.generator[0].linear.weight = self.decoder.word_lut.weight

