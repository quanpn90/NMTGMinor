import torch
import torch.nn as nn
import torch.nn.functional as F
from onmt.models.transformers import Transformer, TransformerDecodingState
from typing import List, Optional, Union
from collections import defaultdict
import onmt


# # the input should have size [b x t x d]
#
# #
# #

#
# # maybe just need d / F.normalize(d, p=2, dim=2)
#
# def norm_vec_sentence_level(d, xp):
#     # d         : (max_len, batchsize, emb_dim)
#     # trans_d   : (batchsize, max_len, emb_dim)
#     trans_d = xp.transpose(d, (1, 0, 2))
#     norm_term = xp.linalg.norm(trans_d, axis=(1, 2), keepdims=True) + 1e-12
#     trans_d = trans_d / norm_term
#     d_sent_norm = xp.transpose(trans_d, (1, 0, 2))
#     return d_sent_norm


# defining a Wav2vec2 encoder wrapping the HuggingFace model here
class FairseqWav2VecExtractor(nn.Module):

    def __init__(self, model_path="wav2vec_vox_new.pt"):
        self.model_path = model_path
        import fairseq
        from fairseq.checkpoint_utils import load_model_ensemble_and_task, load_checkpoint_to_cpu
        from .fairseq_wav2vec2.wav2vec2 import Wav2Vec2Model

        super().__init__()
        state = load_checkpoint_to_cpu(model_path)

        self.cfg = state['cfg']['model']
        self.wav2vec_encoder = Wav2Vec2Model(cfg=self.cfg)
        self.wav2vec_encoder.load_state_dict(state['model'])
        self.wav2vec_encoder.remove_pretraining_modules()

    def forward(self, batch, **kwargs):
        """
        :param batch_first_output: [bsz, seq_len, hidden_size] as output size, else transpose(0, 1)
        :param input: torch.Tensor [batch_size, sequence_length, 2]
        :param kwargs:
        :return:
        """
        input = batch.get('source').transpose(0, 1)  # T x B x H -> B x T x H

        # 0 for tokens that are not masked, 1 for tokens that are masked
        long_mask = input.narrow(2, 0, 1).squeeze(2).eq(0).long()
        input = input.narrow(2, 1, input.size(2) - 1).squeeze(-1)

        attn_mask = long_mask
        # wav2vec_output = self.wav2vec_encoder.extract_features(input, attn_mask, mask=self.training)
        features, padding_mask = self.wav2vec_encoder.extract_conv_features(input, attn_mask)

        return features, padding_mask


class FairseqWav2Vec(nn.Module):

    def __init__(self, opt, model_path="wav2vec_vox_new.pt"):

        super().__init__()
        # do we need opt for this?
        self.opt = opt
        self.model_path = model_path
        import fairseq
        from fairseq.checkpoint_utils import load_model_ensemble_and_task, load_checkpoint_to_cpu
        # from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
        from .fairseq_wav2vec2.wav2vec2 import Wav2Vec2Model
        state = load_checkpoint_to_cpu(model_path)
        self.cfg = state['cfg']['model']

        # don't override the options for wav2vec yet (some of them can create NaN)
        self.cfg.dropout = self.opt.enc_pretrain_emb_dropout
        # self.cfg.activation_dropout = self.opt.ffn_dropout
        self.cfg.attention_dropout = self.opt.enc_pretrain_hidden_dropout
        self.cfg.encoder_layerdrop = self.opt.death_rate
        # self.cfg.dropout_features = self.opt.emb_dropout
        # self.cfg.mask_channel_before = True
        self.cfg.mask_channel_prob = 0.2 if self.opt.wav2vec_spec_augment else 0.0
        self.cfg.mask_channel_length = 64
        self.cfg.mask_prob = 0.0

        self.wav2vec_encoder = Wav2Vec2Model(cfg=self.cfg, favor=opt.favor_attention, weight_drop=opt.weight_drop)
        self.favor = opt.favor_attention
        if self.favor:
            from onmt.modules.performer import ProjectionUpdater
            self.proj_updater = ProjectionUpdater(self.wav2vec_encoder.encoder,
                                                  feature_redraw_interval=1000)
            self.auto_check_redraw = True

        # load wav2vec weights
        wav2vec_weights = state['model']
        existed_weights = self.wav2vec_encoder.state_dict()

        # if we add new weights/buffers to new model then put them into the state_dict
        keys = existed_weights.keys()
        for key in keys:
            if key not in wav2vec_weights:
                wav2vec_weights[key] = existed_weights[key]

        self.wav2vec_encoder.load_state_dict(state['model'])
        self.wav2vec_encoder.remove_pretraining_modules()  # remove the quantization modules

        cfg = self.wav2vec_encoder.cfg
        assert self.opt.model_size == cfg.encoder_embed_dim
        self.input_type = self.opt.encoder_type
        self.model_size = cfg.encoder_embed_dim
        self.wav2vec_encoder.feature_grad_mult = 0.0
        self.time = None

        # freezing the parameters of the Convolutional feature extractors
        for param in self.wav2vec_encoder.feature_extractor.parameters():
            param.requires_grad = False

        if opt.wav2vec_adapter > 0:
            print("[INFO] Adding adapters for Wav2vec model with %d languages" % opt.n_languages)
            self.wav2vec_encoder.encoder.add_adapters(opt.n_languages, adapter_location=opt.wav2vec_adapter)

        if opt.multilingual_factorized_weights:
            print("[INFO] Factorizing Wav2vec model into %d languages" % opt.n_languages)
            self.wav2vec_encoder.encoder.add_factorize(opt.n_languages, rank=opt.mfw_rank,
                                                       multiplicative=opt.mfw_multiplicative,
                                                       fast=opt.fast_factorize)

        if opt.freeze_encoder_ffn:
            self.wav2vec_encoder.encoder.freeze_or_unfreeze_ffn_params()

    def fix_projection_matrices_(self):
        if self.favor:
            self.proj_updater.fix_projections_()

    def convert_fast_attention(self):
        self.wav2vec_encoder.convert_fast_attention()

    def test_run(self, input, mask):

        # input should have size [B x T x H]
        # H == 1: audio samples
        # H > 1: precomputed samples

        if input.size(-1) == 1:
            precomputed_tdnn = False
            input = input.squeeze(-1)
        else:
            precomputed_tdnn = True

        wav2vec_output = self.wav2vec_encoder.extract_features(input, mask,
                                                               mask=False,
                                                               precomputed_tdnn=precomputed_tdnn,
                                                               lang=None, mixture=None)

        context = wav2vec_output['x']
        return context

    def forward(self, input, batch_first_output=False, adv_ptb_grad=False, input_ptb=None,
                lang=None, mixture=None, **kwargs):
        """
        :param mixture:
        :param lang:
        :param input_ptb: perturbation added to the input itself
        :param adv_ptb_grad: adversarial perturbation step which we need the gradients w.r.t the input (wavs)
        :param batch_first_output: [bsz, seq_len, hidden_size] as output size, else transpose(0, 1)
        :param input: torch.Tensor [batch_size, sequence_length, 2]
        :param kwargs:
        :return:
        """

        # 0 for tokens that are not masked, 1 for tokens that are masked
        with torch.no_grad():
            long_mask = input.narrow(2, 0, 1).squeeze(2).eq(0).long()
            input = input.narrow(2, 1, input.size(2) - 1)

        if adv_ptb_grad:
            input.requires_grad = True

        if input_ptb is not None:
            assert not adv_ptb_grad
            with torch.no_grad():
                # normalize and add to input / maybe scale over input length?
                # do this under fp32
                with torch.cuda.amp.autocast(enabled=False):
                    epsilon = 1.0
                    input_ptb = input_ptb.float()
                    input_ptb = input_ptb / F.normalize(input_ptb, p=2.0, dim=2)
                    input = input.float() + input_ptb * epsilon

        if input.size(-1) == 1:
            precomputed_tdnn = False
            input = input.squeeze(-1)
        else:
            precomputed_tdnn = True

        attn_mask = long_mask
        if self.favor:  # favor+ attention
            if self.auto_check_redraw:
                # print("Redraw projection ....")
                self.proj_updater.redraw_projections()

        # don't mask when precomputed tdnn is used, because spec augmentation is used in the dataset
        # wav2vec_output = self.wav2vec_encoder.extract_features(input, attn_mask,
        #                                                        mask=self.training,
        #                                                        precomputed_tdnn=precomputed_tdnn,
        #                                                        lang=lang, mixture=mixture)
        wav2vec_output = self.wav2vec_encoder(input, attn_mask,
                                              mask=self.training, features_only=True, layer=None,
                                              precomputed_tdnn=precomputed_tdnn,
                                              lang=lang, mixture=mixture)

        # TODO: move batch_first_output up to avoid confusion
        if not batch_first_output:
            context = wav2vec_output['x'].transpose(0, 1).contiguous()
            batch_size, time = context.size(1), context.size(0)
        else:
            context = wav2vec_output['x']
            time, batch_size = context.size(1), context.size(0)

        dec_attn_mask = wav2vec_output['padding_mask']
        if dec_attn_mask is None:
            dec_attn_mask = context.new_zeros(batch_size, time).byte()
        else:
            dec_attn_mask = dec_attn_mask.byte()

        # how to get the correct attention mask?
        output_dict = defaultdict(lambda: None, {'source': input, 'context': context, 'src_mask': dec_attn_mask,
                                                 'src': dec_attn_mask, 'pos_emb': None})

        return output_dict


class Wav2vecTransformer(Transformer):
    """Main model in 'Attention is all you need' """

    def __init__(self, encoder, decoder, generator=None,
                 mirror=False, ctc=False, **kwargs):
        super().__init__(encoder, decoder, generator, None, None, ctc=ctc)
        self.model_size = self.decoder.model_size
        self.switchout = self.decoder.switchout

        if mirror:
            self.mirror_decoder = copy.deepcopy(self.decoder)
            self.mirror_g = nn.Linear(decoder.model_size, decoder.model_size)
            self.mirror_generator = copy.deepcopy(self.generator)
            self.mirror_generator[0].linear.weight = self.decoder.word_lut.weight

        if self.ctc:
            self.ctc_linear = nn.Linear(encoder.model_size, self.tgt_vocab_size)

    def reset_states(self):
        return

    def forward(self, batch, adv_ptb_grad=False, input_ptb=None, factorize=False,
                mirror=False, target_mask=None, **kwargs):
        """
        :param factorize:
        :param mirror:
        :param adv_ptb_grad: If we need to tell the model to set input.requires_grad=True (1st step)
        :param input_ptb: 2nd step of adversarial: add the perturbation to input
        :param batch: data object sent from the dataset
        :return:
        """
        if self.switchout > 0 and self.training:
            batch.switchout(self.switchout, self.src_vocab_size, self.tgt_vocab_size)

        src = batch.get('source')
        tgt = batch.get('target_input')
        tgt_pos = batch.get('target_pos')
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')
        src_lengths = batch.src_lengths
        tgt_lengths = batch.tgt_lengths

        org_src = src
        org_tgt = tgt
        src = src.transpose(0, 1)  # transpose to have batch first
        tgt = tgt.transpose(0, 1)

        encoder_output = self.encoder(src, adv_ptb_grad=adv_ptb_grad, input_ptb=input_ptb)

        encoder_output = defaultdict(lambda: None, encoder_output)
        context = encoder_output['context']
        src = encoder_output['src']

        # pass the mask ('src') from the encoder output the decoder as the attention mask
        decoder_output = self.decoder(tgt, context, src,
                                      src_lang=src_lang, tgt_lang=tgt_lang, input_pos=tgt_pos,
                                      src_lengths=src_lengths, tgt_lengths=tgt_lengths,
                                      factorize=factorize)

        decoder_output = defaultdict(lambda: None, decoder_output)
        output = decoder_output['hidden']

        # build the output dict based on decoder output
        output_dict = defaultdict(lambda: None, decoder_output)
        output_dict['hidden'] = output
        output_dict['context'] = context
        output_dict['src_mask'] = encoder_output['src']
        output_dict['src'] = src
        output_dict['target_mask'] = target_mask
        output_dict['target'] = batch.get('target_output')
        output_dict['source'] = encoder_output['source']

        # final layer: computing softmax
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

        output_dict['reconstruct'] = False

        # compute the logits for each encoder step
        if self.ctc:
            # raise NotImplementedError
            output_dict['encoder_logits'] = self.ctc_linear(output_dict['context'])

        return output_dict

    # load pretrained wav2vec weights
    def load_encoder_weights(self, checkpoint):
        self.encoder.wav2vec_encoder.load_state_dict(checkpoint['model'])

    def create_decoder_state(self, batch, beam_size=1, type=1, buffering=True,
                             pretrained_layer_states=None, **kwargs):
        """
        Generate a new decoder state based on the batch input
        :param pretrained_layer_states:
        :param buffering:
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

        src_transposed = src.transpose(0, 1)  # transpose -> batch first
        encoder_output = self.encoder(src_transposed)

        src = encoder_output['src'].transpose(0, 1)
        src_mask = encoder_output['src']

        print("[INFO] create Transformer decoding state with buffering", buffering)
        decoder_state = TransformerDecodingState(src, tgt_lang, encoder_output['context'], src_lang,
                                                 beam_size=beam_size, model_size=self.model_size,
                                                 type=type, buffering=buffering, src_mask=src_mask)

        return decoder_state

    def step(self, input_t, decoder_state, streaming=False):
        """
        Decoding function:
        generate new decoder output based on the current input and current decoder state
        the decoder state is updated in the process
        :param streaming:
        :param input_t: the input word index at time t
        :param decoder_state: object DecoderState containing the buffers required for decoding
        :return: a dictionary containing: log-prob output and the attention coverage
        """

        output_dict = self.decoder.step(input_t, decoder_state, streaming=streaming)
        output_dict['src'] = decoder_state.src.transpose(0, 1)

        log_prob = self.generator[0](output_dict)['logits'].squeeze(0)
        log_prob = torch.nn.functional.log_softmax(log_prob, dim=-1, dtype=torch.float32)

        coverage = output_dict['coverage']
        last_coverage = coverage[:, -1, :].squeeze(1)

        output_dict['log_prob'] = log_prob
        output_dict['coverage'] = last_coverage

        return output_dict


class Wav2vecBERT(Wav2vecTransformer):

    def __init__(self, encoder, decoder, generator=None,
                 mirror=False, ctc=False, encoder_type='wav2vec2',
                 decoder_type='bart',
                 sub_encoder=None, mutual_modality_training=False, **kwargs):
        super().__init__(encoder, decoder, generator, mirror=mirror, ctc=ctc)

        self.src_vocab_size = 0
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type

        if hasattr(decoder, 'dec_pretrained_model') and decoder.dec_pretrained_model:
            self.model_size = self.decoder.config.bert_hidden_size
            self.tgt_vocab_size = self.decoder.config.vocab_size
            self.switchout = 0
        else:
            self.model_size = self.decoder.model_size
            self.tgt_vocab_size = self.decoder.word_lut.weight.size(0)
            self.switchout = self.decoder.switchout

        if mirror:
            self.mirror_decoder = copy.deepcopy(self.decoder)
            self.mirror_g = nn.Linear(decoder.model_size, decoder.model_size)
            self.mirror_generator = copy.deepcopy(self.generator)
            self.mirror_generator[0].linear.weight = self.decoder.word_lut.weight

        if self.ctc:
            self.ctc_linear = nn.Linear(encoder.model_size, self.tgt_vocab_size)

    def forward(self, batch, zero_encoder=False, factorize=False, target_mask=None, mirror=False, **kwargs):
        """
        :param batch: data object sent from the dataset
        :return:
        """
        if self.switchout > 0 and self.training:
            batch.switchout(self.switchout, self.src_vocab_size, self.tgt_vocab_size)

        src = batch.get('source')
        tgt = batch.get('target_input')
        tgt_pos = batch.get('target_pos')
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')
        src_lengths = batch.src_lengths
        tgt_lengths = batch.tgt_lengths

        org_src = src
        org_tgt = tgt
        src = src.transpose(0, 1)  # transpose to have batch first
        tgt = tgt.transpose(0, 1)

        batch_first_output = False
        if hasattr(self.decoder, 'dec_pretrained_model') and self.decoder.dec_pretrained_model in ["bart"]:
            batch_first_output = True

        # during training mixture is always None
        encoder_output = self.encoder(src, batch_first_output=batch_first_output, lang=src_lang, mixture=None)

        encoder_output = defaultdict(lambda: None, encoder_output)

        context = encoder_output['context']
        src_attention_mask = encoder_output['src']
        if hasattr(self.decoder, 'dec_pretrained_model') and self.decoder.dec_pretrained_model in ["bert", "roberta"]:
            # src: [b, src_l]  context: [b, src_l, de_model]
            tgt_token_type = tgt.ne(onmt.constants.TGT_PAD).long()  # [bsz, len]
            tgt_attention_mask = tgt.new(*tgt.size()).fill_(1)  # [bsz, len]
            decoder_output = self.decoder(input_ids=tgt,
                                          attention_mask=tgt_attention_mask,
                                          token_type_ids=tgt_token_type,
                                          encoder_hidden_states=context,
                                          encoder_attention_mask=src_attention_mask,
                                          no_offset=True)

            decoder_output = decoder_output[0]
            output = decoder_output.transpose(0, 1)  # [bsz, tgt_len, d] => [tgt_len, bsz, d]
            output_dict = defaultdict(lambda: None)
            context = context.transpose(0, 1)  # to [src_l, b, de_model]
        elif hasattr(self.decoder, 'dec_pretrained_model') and self.decoder.dec_pretrained_model in ["bart"]:
            tgt_token_type = tgt.ne(onmt.constants.TGT_PAD).long()  # [bsz, len]
            tgt_attention_mask = tgt.new(*tgt.size()).fill_(1)  # [bsz, len]

            # the wav2vec returned mask is 1 for masked and 0 for un-masked, which is opposite to huggingface
            src_attention_mask = 1 - (src_attention_mask.long())

            decoder_output = self.decoder(input_ids=tgt,
                                          attention_mask=tgt_attention_mask,
                                          encoder_hidden_states=context,
                                          encoder_attention_mask=src_attention_mask)
            decoder_output = decoder_output[0]
            output = decoder_output.transpose(0, 1)  # [bsz, tgt_len, d] => [tgt_len, bsz, d]
            context = context.transpose(0, 1)
            output_dict = defaultdict(lambda: None)
        elif hasattr(self.decoder, 'dec_pretrained_model') and self.decoder.dec_pretrained_model in ["mbart", "mbart50"]:
            src_attention_mask = src_attention_mask   # new version
            # tgt_attention_mask = tgt.ne(onmt.constants.TGT_PAD).long()  # [bsz, len]
            tgt_attention_mask = tgt.new(*tgt.size()).fill_(1)

            decoder_output = self.decoder(input_ids=tgt,
                                          attention_mask=tgt_attention_mask,
                                          encoder_hidden_states=context,
                                          encoder_attention_mask=src_attention_mask)
            decoder_output = decoder_output[0]
            output = decoder_output
            output_dict = defaultdict(lambda: None)

        else:
            # pass the mask ('src') from the encoder output the decoder as the attention mask
            decoder_output = self.decoder(tgt, context, src,
                                          src_lang=src_lang, tgt_lang=tgt_lang, input_pos=tgt_pos,
                                          src_lengths=src_lengths, tgt_lengths=tgt_lengths,
                                          factorize=factorize)

            decoder_output = defaultdict(lambda: None, decoder_output)
            output = decoder_output['hidden']

        output_dict['hidden'] = output
        output_dict['context'] = context
        output_dict['src_mask'] = encoder_output['src']
        output_dict['src'] = src
        output_dict['target_mask'] = target_mask
        output_dict['target'] = batch.get('target_output')

        # final layer: computing softmax
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

        output_dict['reconstruct'] = False

        # compute the logits for each encoder step
        if self.ctc:
            # raise NotImplementedError
            output_dict['encoder_logits'] = self.ctc_linear(output_dict['context'])

        return output_dict

    def create_decoder_state(self, batch, beam_size=1, type=1, buffering=True, mixture=None, **kwargs):
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
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')

        if mixture is not None:
            raise NotImplementedError
        encoder_output = self.encoder(src.transpose(0, 1), batch_first_output=False, lang=src_lang, mixture=mixture)
        src_attention_mask = encoder_output['src']

        dec_pretrained_model = self.decoder.dec_pretrained_model
        if not dec_pretrained_model:
            mask_src = None
        elif dec_pretrained_model in ["bert", "roberta"]:
            mask_src = src_attention_mask.unsqueeze(1)  # batch_size  x 1 x len_src for broadcasting

        elif dec_pretrained_model in ["bart"]:
            mask_src = 1 - (src_attention_mask.long())
        elif dec_pretrained_model in ["mbart", "mbart50"]:
            mask_src = src_attention_mask
        else:
            print("Warning: unknown dec_pretrained_model")
            raise NotImplementedError

        decoder_state = TransformerDecodingState(src, tgt_lang, encoder_output['context'], src_lang,
                                                 beam_size=beam_size, model_size=self.model_size,
                                                 type=type, buffering=buffering, src_mask=mask_src,
                                                 dec_pretrained_model=self.decoder.dec_pretrained_model)

        return decoder_state

    def tie_weights(self):
        assert self.generator is not None, "The generator needs to be created before sharing weights"
        if hasattr(self.decoder, 'dec_pretrained_model') and self.decoder.dec_pretrained_model in ["bert", "roberta"]:
            self.generator[0].linear.weight = self.decoder.embeddings.word_embeddings.weight
        elif hasattr(self.decoder, 'dec_pretrained_model') and self.decoder.dec_pretrained_model in ["mbart", "mbart50"]:
            self.generator[0].linear.weight = self.decoder.embed_tokens.weight
        else:
            self.generator[0].linear.weight = self.decoder.word_lut.weight

    def decode(self, batch):

        raise NotImplementedError
        # """
        # :param batch: (onmt.Dataset.Batch) an object containing tensors needed for training
        # :return: gold_scores (torch.Tensor) log probs for each sentence
        #          gold_words  (Int) the total number of non-padded tokens
        #          allgold_scores (list of Tensors) log probs for each word in the sentence
        # """
        #
        # src = batch.get('source')
        # src_pos = batch.get('source_pos')
        # tgt_input = batch.get('target_input')
        # tgt_output = batch.get('target_output')
        # tgt_pos = batch.get('target_pos')
        # # tgt_atb = batch.get('target_atb')  # a dictionary of attributes
        # src_lang = batch.get('source_lang')
        # tgt_lang = batch.get('target_lang')
        #
        # # transpose to have batch first
        # src = src.transpose(0, 1)
        # tgt_input = tgt_input.transpose(0, 1)
        # batch_size = tgt_input.size(0)
        #
        # context = self.encoder(src, input_pos=src_pos, input_lang=src_lang)['context']
        #
        # if hasattr(self, 'autoencoder') and self.autoencoder \
        #         and self.autoencoder.representation == "EncoderHiddenState":
        #     context = self.autoencoder.autocode(context)
        #
        # gold_scores = context.new(batch_size).zero_()
        # gold_words = 0
        # allgold_scores = list()
        # decoder_output = self.decoder(tgt_input, context, src, tgt_lang=tgt_lang, src_lang=src_lang,
        #                               input_pos=tgt_pos)['hidden']
        #
        # output = decoder_output
        #
        # if hasattr(self, 'autoencoder') and self.autoencoder and \
        #         self.autoencoder.representation == "DecoderHiddenState":
        #     output = self.autoencoder.autocode(output)
        #
        # for dec_t, tgt_t in zip(output, tgt_output):
        #
        #     dec_out = defaultdict(lambda: None)
        #     dec_out['hidden'] = dec_t.unsqueeze(0)
        #     dec_out['src'] = src
        #     dec_out['context'] = context
        #
        #     if isinstance(self.generator, nn.ModuleList):
        #         gen_t = self.generator[0](dec_out)['logits']
        #     else:
        #         gen_t = self.generator(dec_out)['logits']
        #     gen_t = F.log_softmax(gen_t, dim=-1, dtype=torch.float32)
        #     gen_t = gen_t.squeeze(0)
        #     tgt_t = tgt_t.unsqueeze(1)
        #     scores = gen_t.gather(1, tgt_t)
        #     scores.masked_fill_(tgt_t.eq(onmt.constants.TGT_PAD), 0)
        #     gold_scores += scores.squeeze(1).type_as(gold_scores)
        #     gold_words += tgt_t.ne(onmt.constants.TGT_PAD).sum().item()
        #     allgold_scores.append(scores.squeeze(1).type_as(gold_scores))
        #
        # return gold_words, gold_scores, allgold_scores