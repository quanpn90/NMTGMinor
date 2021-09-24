import torch
import torch.nn as nn
from onmt.models.transformers import Transformer, TransformerDecodingState
from typing import List, Optional, Union
from collections import defaultdict


# defining a Wav2vec2 encoder wrapping the HuggingFace model here


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

        self.cfg.dropout = self.opt.residual_dropout
        self.cfg.activation_dropout = self.opt.ffn_dropout
        self.cfg.attention_dropout = self.opt.attn_dropout
        self.cfg.encoder_layerdrop = self.opt.death_rate / 2
        self.cfg.dropout_features = self.opt.emb_dropout

        self.wav2vec_encoder = Wav2Vec2Model(cfg=self.cfg)
        self.wav2vec_encoder.load_state_dict(state['model'])
        self.wav2vec_encoder.remove_pretraining_modules()

        # model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
        # self.wav2vec_encoder = model[0]

        cfg = self.wav2vec_encoder.cfg
        assert self.opt.model_size == cfg.encoder_embed_dim
        self.input_type = self.opt.encoder_type
        self.model_size = cfg.encoder_embed_dim
        self.wav2vec_encoder.feature_grad_mult = 0.0
        self.time = None

        for param in self.wav2vec_encoder.feature_extractor.parameters():
        # for param in self.wav2vec_encoder.parameters():
            param.requires_grad = False
        # for i, layer in enumerate(self.wav2vec_encoder.encoder.layers):
        #     if i >= 0.5 *  self.wav2vec_encoder.config.num_hidden_layers:
        #         for param in layer.parameters():
        #             param.requires_grad = True

    def forward(self, input, **kwargs):
        """
        :param input: torch.Tensor [batch_size, sequence_length, 2]
        :param kwargs:
        :return:
        """

        # 0 for tokens that are not masked, 1 for tokens that are masked
        long_mask = input.narrow(2, 0, 1).squeeze(2).eq(0).long()
        input = input.narrow(2, 1, input.size(2) - 1).squeeze(-1)

        attn_mask = long_mask
        wav2vec_output = self.wav2vec_encoder.extract_features(input, attn_mask)

        context = wav2vec_output['x'].transpose(0, 1).contiguous()
        batch_size, time = context.size(1), context.size(0)

        dec_attn_mask = wav2vec_output['padding_mask']
        if dec_attn_mask is None:
            dec_attn_mask = context.new_zeros(batch_size, time).byte()
        else:
            dec_attn_mask = (dec_attn_mask).byte()

        # how to get the correct attention mask?
        output_dict = defaultdict(lambda: None, {'context': context, 'src_mask': dec_attn_mask,
                                                 'src': dec_attn_mask, 'pos_emb': None})

        return output_dict


class Wav2vecTransformer(Transformer):
    """Main model in 'Attention is all you need' """

    def __init__(self, encoder, decoder, generator=None,
                 mirror=False, ctc=False, **kwargs):
        super().__init__(encoder, decoder, generator, None, None, ctc=ctc)
        self.model_size = self.decoder.model_size
        self.switchout = self.decoder.switchout
        self.tgt_vocab_size = self.decoder.word_lut.weight.size(0)

        if self.encoder.input_type == 'text':
            self.src_vocab_size = self.encoder.word_lut.weight.size(0)
        else:
            self.src_vocab_size = 0

        if mirror:
            self.mirror_decoder = copy.deepcopy(self.decoder)
            self.mirror_g = nn.Linear(decoder.model_size, decoder.model_size)
            self.mirror_generator = copy.deepcopy(self.generator)
            self.mirror_generator[0].linear.weight = self.decoder.word_lut.weight

        if self.ctc:
            self.ctc_linear = nn.Linear(encoder.model_size, self.tgt_vocab_size)

    def reset_states(self):
        return

    def forward(self, batch, zero_encoder=False, factorize=False, target_mask=None, mirror=False, **kwargs):
        """
        :param batch: data object sent from the dataset
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

        encoder_output = self.encoder(src)
        # src = src.new(src.size(0), 100).zero_()
        # context = src.new_zeros(100, src.size(0), 1024)

        encoder_output = defaultdict(lambda: None, encoder_output)
        # encoder_output['context'] = context
        # encoder_output['src'] = src
        # context = encoder_output['context']
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

    def load_encoder_weights(self, pretrained_model):
        super().load_encoder_weights(pretrained_model)

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

        print("[INFO] create Transformer decoding state with buffering", buffering)
        decoder_state = TransformerDecodingState(src, tgt_lang, encoder_output['context'], src_lang,
                                                 beam_size=beam_size, model_size=self.model_size,
                                                 type=type, buffering=buffering)

        return decoder_state