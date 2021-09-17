import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from onmt.models.transformers import Transformer
from typing import List, Optional, Union
from collections import defaultdict

# defining a Wav2vec2 encoder wrapping the HuggingFace model here


class HuggingFaceWav2Vec(nn.Module):

    def __init__(self, opt, model_path="facebook/wav2vec2-large-lv60"):

        super().__init__()
        # do we need opt for this?
        self.model_path = model_path
        self.wav2vec_encoder = Wav2Vec2Model.from_pretrained(model_path, gradient_checkpointing=False)
        # print(self.wav2vec_encoder)
        self.wav2vec_encoder.feature_extractor._freeze_parameters()
        self.opt = opt
        assert self.opt.encoder_type == 'wav2vec2', "expecting wav2vec2 but get %s" % self.opt.encoder_type
        # assert self.opt.model_size == 1024
        self.input_type = self.opt.encoder_type
        self.model_size = self.wav2vec_encoder.config.hidden_size
        self.wav2vec_encoder.config.mask_time_prob = 0.0

        assert self.model_size == self.opt.model_size

        for param in self.wav2vec_encoder.parameters():
            param.requires_grad = False
        # for i, layer in enumerate(self.wav2vec_encoder.encoder.layers):
        #     if i >= 0.5 *  self.wav2vec_encoder.config.num_hidden_layers:
        #         for param in layer.parameters():
        #             param.requires_grad = True


    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(self.wav2vec_encoder.config.conv_kernel, self.wav2vec_encoder.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor):
        output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    def forward(self, input, **kwargs):
        """
        :param input: torch.Tensor [batch_size, sequence_length, 2]
        :param kwargs:
        :return:
        """

        with torch.no_grad():
            # 1 for tokens that are not masked, 0 for tokens that are masked
            long_mask = input.narrow(2, 0, 1).squeeze(2).eq(1).long()
            input = input.narrow(2, 1, input.size(2) - 1).squeeze(-1)

            attn_mask = long_mask
            wav2vec_attn_mask = attn_mask if 'base' not in self.model_path else None
            wav2vec_encoder_states = self.wav2vec_encoder(input, wav2vec_attn_mask).last_hidden_state
            context = wav2vec_encoder_states.transpose(0, 1).contiguous()

            dec_attn_mask = self._get_feature_vector_attention_mask(context.size(0), attn_mask).bool()
            # 1 for tokens that are masked, 0 for non-masked tokens
            dec_attn_mask = ~dec_attn_mask

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

        # if self.reconstruct:
        #     self.rec_linear = nn.Linear(decoder.model_size, decoder.model_size)
        #
        #
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

        encoder_output = defaultdict(lambda: None, encoder_output)
        context = encoder_output['context']

        # zero out the encoder part for pre-training
        # # if zero_encoder:
        # #     context.zero_()
        # print(context)
        # context.zero_()

        # pass the mask ('src') from the encoder output the decoder as the attention mask
        decoder_output = self.decoder(tgt, context, encoder_output['src'],
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