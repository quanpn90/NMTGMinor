# Transformer with discourse information
from collections import defaultdict
import onmt
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..speech_recognizer.relative_transformer_layers import \
    RelativeTransformerEncoderLayer, RelativeTransformerDecoderLayer
from onmt.models.transformers import Transformer, TransformerDecodingState
from onmt.modules.pre_post_processing import PrePostProcessing

from .gate_layer import RelativeGateEncoderLayer


class DiscourseTransformerEncoder(nn.Module):

    def __init__(self, opt, encoder=None):
        self.opt = opt
        super(DiscourseTransformerEncoder, self).__init__()
        # a shared encoder for all present, past and future
        self.encoder = encoder

        self.past_layer = RelativeTransformerEncoderLayer(self.opt)
        self.input_type = encoder.input_type
        self.time = None  # backward compatible

        self.gate_layer = RelativeGateEncoderLayer(self.opt)

        self.postprocess_layer = PrePostProcessing(opt.model_size, 0.0, sequence='n')

    def forward(self, input, past_input=None, input_lang=None, factorize=False):

        assert past_input is not None

        # the same encoder is used to encode the previous and current segment
        past_encoder_output = self.encoder(past_input, input_lang=input_lang, factorize=factorize)
        #
        past_context = past_encoder_output['context']
        past_pos_emb = past_encoder_output['pos_emb']

        encoder_output = self.encoder(input, input_lang=input_lang, factorize=factorize)

        # past_mask_src = past_input.narrow(2, 0, 1).squeeze(2).transpose(0, 1).eq(onmt.constants.PAD).unsqueeze(0)
        # past_context = self.past_layer(past_context, past_pos_emb, past_mask_src,
        #                                src_lang=input_lang, factorize=factorize)

        current_context = encoder_output['context']
        current_pos_emb = encoder_output['pos_emb']

        mask_src = input.narrow(2, 0, 1).squeeze(2).transpose(0, 1).eq(onmt.constants.PAD).unsqueeze(0)
        past_mask = past_input.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD).unsqueeze(1)
        dec_attn_mask = input.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD).unsqueeze(1)
        context = self.gate_layer(current_context, past_context, current_pos_emb, mask_src, past_mask,
                                   src_lang=input_lang, factorize=factorize)
        # context = current_context

        # final layer norm
        context = self.postprocess_layer(context, factor=input_lang)

        output_dict = defaultdict(lambda: None, {'context': context, 'src_mask': dec_attn_mask,
                                                 'src': input, 'pos_emb': current_pos_emb})

        del past_encoder_output

        return output_dict


class DiscourseTransformer(Transformer):
    """Main model in 'Attention is all you need' """

    def forward(self, batch, target_mask=None, streaming=False, zero_encoder=False,
                mirror=False, streaming_state=None, nce=False, factorize=True, **kwargs):
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

        past_src = batch.get('past_source')

        org_src = src
        org_tgt = tgt
        src = src.transpose(0, 1)  # transpose to have batch first
        tgt = tgt.transpose(0, 1)
        past_src = past_src.transpose(0, 1)

        # Encoder has to receive different inputs
        encoder_output = self.encoder(src, past_input=past_src, input_lang=src_lang,
                                      factorize=factorize)

        encoder_output = defaultdict(lambda: None, encoder_output)
        context = encoder_output['context']

        # the state is changed
        streaming_state = encoder_output['streaming_state']

        # zero out the encoder part for pre-training
        if zero_encoder:
            context.zero_()

        decoder_output = self.decoder(tgt, context, src,
                                      src_lang=src_lang, tgt_lang=tgt_lang, input_pos=tgt_pos, streaming=streaming,
                                      src_lengths=src_lengths, tgt_lengths=tgt_lengths,
                                      streaming_state=streaming_state, factorize=factorize)

        # update the streaming state again
        decoder_output = defaultdict(lambda: None, decoder_output)
        streaming_state = decoder_output['streaming_state']
        output = decoder_output['hidden']

        # build the output dict based on decoder output
        output_dict = defaultdict(lambda: None, decoder_output)
        output_dict['hidden'] = output
        output_dict['context'] = context
        output_dict['src_mask'] = encoder_output['src_mask']
        output_dict['src'] = src
        output_dict['target_mask'] = target_mask
        output_dict['streaming_state'] = streaming_state
        output_dict['target'] = batch.get('target_output')
        # output_dict['lid_logits'] = decoder_output['lid_logits']

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

        # compute the logits for each encoder step
        if self.ctc:
            output_dict['encoder_logits'] = self.ctc_linear(output_dict['context'])

        del encoder_output

        return output_dict

    def load_encoder_weights(self, pretrained_model):

        # take the shared encoder section of the encoder
        encoder_ = self.encoder.encoder

        pretrained_model.encoder.language_embedding = None

        enc_language_embedding = encoder_.language_embedding
        encoder_.language_embedding = None
        encoder_state_dict = pretrained_model.encoder.state_dict()

        encoder_.load_state_dict(encoder_state_dict)
        encoder_.language_embedding = enc_language_embedding

    # TODO: override
    def create_decoder_state(self, batch, beam_size=1, type=1, buffering=True, factorize=True, **kwargs):
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
        tgt_atb = batch.get('target_atb')
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')
        past_src = batch.get('past_source')

        src_transposed = src.transpose(0, 1)
        # encoder_output = self.encoder(src_transposed, input_pos=src_pos, input_lang=src_lang)
        encoder_output = self.encoder(src_transposed, past_input=past_src.transpose(0, 1), input_lang=src_lang,
                                      factorize=factorize)

        # The decoding state is still the same?
        print("[INFO] create Transformer decoding state with buffering", buffering)
        decoder_state = TransformerDecodingState(src, tgt_lang, encoder_output['context'], src_lang,
                                                 beam_size=beam_size, model_size=self.model_size,
                                                 type=type, buffering=buffering)

        return decoder_state