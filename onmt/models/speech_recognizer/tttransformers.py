import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from onmt.models.transformers import Transformer, TransformerDecodingState
from typing import List, Optional, Union
from collections import defaultdict
import onmt
from onmt.modules.optimized.linear import Linear
from onmt.modules.layer_norm import LayerNorm
import math
from .wav2vec2 import Wav2vecBERT, Wav2vecTransformer

from pretrain_module.modeling_mbart import MBartCrossAttention


class PerceiverLayer(nn.Module):

    def __init__(self, max_length, model_size, num_heads, need_latent_array=True):

        super(PerceiverAdapterLayer, self).__init__()
        self.max_length = max_length
        self.model_size = model_size
        self.hidden_size = model_size * 4

        if need_latent_array:
            self.weights = nn.Embedding(max_length, model_size, padding_idx=-1)
        else:
            self.weights = None

        self.ln = LayerNorm(model_size)

        self.perceiver_attn = MBartCrossAttention(
            model_size,
            num_heads,
            dropout=0.0
        )

    def forward(self, input=None, input_length=None, context=None, context_mask=None):
        """
        Args:
            input: torch.Tensor [T_q x B x C]
            input_length: int - the output
            context: torch Tensor [T x B x C]
            context_mask: torch.Tensor [B x T]

        Returns:

        """
        len_k, bsz = context.size(0), context,size(1)

        if input is None:
            with torch.no_grad():
                emb_input = torch.arange(0, input_length, step=1, dtype=torch.int, device=context.device)
                emb_input = emb_input.sequeeze(1).repeat(1, bsz)
                emb_input = torch.clamp(emb_input, max=self.max_length - 1)

            # don't forget to clamp the embedding input
            hidden_states = self.weights(emb_input)
        else:
            hidden_states = input

        residual = hidden_states

        hidden_states = self.ln(hidden_states)

        cu_seqlens, max_len, cu_seqlens_kv, max_len_kv = None, None, None, None

        # todo: add flash attention here
        hidden_states, cross_attn_weights, _ = self.perceiver_attn(
            hidden_states=hidden_states,
            key_value_states=context,
            attention_mask=context_mask,
            output_attentions=False,
            incremental=False, incremental_cache=None,
            checkpointing=False,
            lang=None, atb=None,
            cu_seqlens=cu_seqlens, max_len=max_len,
            cu_seqlens_kv=cu_seqlens_kv, max_len_kv=max_len_kv
        )

        # if we don't use residual connection here, hidden states will only carry the information of the byte array
        # (the audio signals in our case)
        hidden_states = residual + hidden_states

        # do we need to return anything else?
        return hidden_states


class Perceiver(nn.Module):

    def __init__(self, max_length, model_size, num_heads, num_layers=2, n_repeat=1):

        super(PerceiverAdapter, self).__init__()

        self.num_layers = num_layers
        self.n_repeat = n_repeat

        self.perceiver_layer = PerceiverLayer(max_length, model_size, num_heads)

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=model_size,
                    ffn_embedding_dim=model_size * 4,
                    num_attention_heads=num_heads,
                    dropout=0.0,
                    weight_drop=0.0,
                    attention_dropout=0.0,
                    activation_dropout=args.activation_dropout,
                    activation_fn="relu",
                    layer_norm_first=True,
                    favor=False
                )
                for _ in range(num_layers)
            ]
        )

        #todo: add flash attention requirement

    def forward(self, input_length, context, context_mask):

        x = None

        # todo: add flashattn

        for j in range(self.n_repeat):

            # first, do perceiver cross-attention
            if j == 0:
                x = self.perceiver_layer(input=None, input_length=input_length,
                                         context=context, context_mask=context_mask)

            else:
                x = self.perceiver_layer(input=x, context=context, context_mask=context_mask)

            max_len, cu_seqlens, lang, atb = None, None, None, None

            # self-attention over here
            for layer in self.layers:
                x, z = layer(x, self_attn_padding_mask=context_mask, positions=None,
                             max_len=max_len, cu_seqlens=cu_seqlens,
                             lang=lang, atb=atb)

        return x

class DeltaWav2vecBERT(Wav2vecTransformer):

    def __init__(self, encoder, decoder, aux_encoder=None, adapter=None, generator=None,
                 mirror=False, ctc=False, encoder_type='wav2vec2',
                 decoder_type='bart',
                 sub_encoder=None, **kwargs):
        super().__init__(encoder, decoder, generator, mirror=mirror, ctc=ctc)

        self.src_vocab_size = 0
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.sub_encoder = sub_encoder # what is this for???
        self.aux_encoder = aux_encoder # the encoder that connects audio encoder and text decoder
        self.adapter = adapter

        if hasattr(aux_encoder, 'enc_pretrained_model') and aux_encoder.enc_pretrained_model:
            try:
                self.src_vocab_size = self.decoder.config.vocab_size
            except AttributeError:
                # if cannot get the src vocab size, we get from the generator
                self.model_size = self.decoder.model_size
                self.src_vocab_size = self.generator[0].linear.weight.size(0)

        if hasattr(decoder, 'dec_pretrained_model') and decoder.dec_pretrained_model:
            try:
                self.model_size = self.decoder.config.bert_hidden_size
                self.tgt_vocab_size = self.decoder.config.vocab_size
            except AttributeError:
                self.model_size = self.decoder.model_size
                self.tgt_vocab_size = self.generator[0].linear.weight.size(0)

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

        # CTC from audio encoder to aux targets
        self.src_ctc_linear = Linear(self.model_size, self.src_vocab_size)

        # join all embedding weights (assuming the aux encoder and decoder have the same vocabulary/embeddings)
        self.src_ctc_linear.weight = self.generator[0].linear.weight

        # CTC from encoder to targets
        if self.ctc:
            self.ctc_linear = Linear(self.model_size, self.tgt_vocab_size)
            self.ctc_linear.weight = self.generator[0].linear.weight


    def forward(self, batch, zero_encoder=False, factorize=False, target_mask=None, mirror=False,
                checkpointing_ffn=False, checkpointing_cross_attn=False, checkpointing_self_attn=False, **kwargs):
        """
        :param checkpointing_self_attn:
        :param checkpointing_cross_attn:
        :param checkpointing_ffn:
        :param batch:
        :param zero_encoder:
        :param factorize:
        :param target_mask:
        :param mirror:
        :param kwargs:
        :return:
        """
        if self.switchout > 0 and self.training:
            batch.switchout(self.switchout, self.src_vocab_size, self.tgt_vocab_size)

        src = batch.get('source')
        tgt = batch.get('target_input')
        aux_tgt = batch.get('target_aux_input')

        tgt_pos = batch.get('target_pos')
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')
        src_atb = batch.get('source_atbs')
        tgt_atb = batch.get('target_atbs')
        src_lengths = batch.src_lengths
        tgt_lengths = batch.tgt_lengths

        org_src = src
        org_tgt = tgt
        src = src.transpose(0, 1)  # transpose to have batch first
        tgt = tgt.transpose(0, 1)

        batch_first_output = False
        if hasattr(self.decoder, 'dec_pretrained_model') and self.decoder.dec_pretrained_model in ["bart"]:
            batch_first_output = True

        # print(src_lang, src_atb, tgt_lang, tgt_atb)

        # during training mixture is always None
        encoder_output = self.encoder(src, batch_first_output=batch_first_output,
                                      lang=src_lang, atb=src_atb,
                                      checkpointing_ffn=checkpointing_ffn,
                                      checkpointing_self_attn=checkpointing_self_attn)

        encoder_output = defaultdict(lambda: None, encoder_output)

        context = encoder_output['context']
        src_attention_mask = encoder_output['src']
        contrastive_loss = 0

        if hasattr(self.decoder, 'dec_pretrained_model') and self.decoder.dec_pretrained_model in ["bert", "roberta"]:
            # src: [b, src_l]  context: [b, src_l, de_model]
            tgt_token_type = tgt.ne(onmt.constants.TGT_PAD).long()  # [bsz, len]
            tgt_attention_mask = tgt.new(*tgt.size()).fill_(1)  # [bsz, len]
            decoder_output = self.decoder(input_ids=tgt,
                                          attention_mask=tgt_attention_mask,
                                          token_type_ids=tgt_token_type,
                                          encoder_hidden_states=context,
                                          encoder_attention_mask=src_attention_mask,
                                          no_offset=True,
                                          )

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
        elif hasattr(self.decoder, 'dec_pretrained_model') and self.decoder.dec_pretrained_model \
                in ["deltalm", "mbart", "mbart50"]:
            if self.sub_encoder is not None:
                src_text_input = batch.get('target')
                sub_context_mask = batch.get('tgt_selfattn_mask')

                with torch.no_grad():
                    sub_encoder_output = self.sub_encoder(input_ids=src_text_input,
                                                          attention_mask=sub_context_mask)
                    sub_context = sub_encoder_output[0]
                    # print(torch.isnan(sub_context).float().sum())

            else:
                sub_context = None
                sub_context_mask = None

            src_attention_mask = src_attention_mask  # new version
            # tgt_attention_mask = tgt.ne(onmt.constants.TGT_PAD).long()  # [bsz, len]
            # tgt_attention_mask = tgt.new(*tgt.size()).fill_(1)
            tgt_attention_mask = batch.get('target_input_selfattn_mask')

            decoder_outputs = self.decoder(input_ids=tgt,
                                           attention_mask=tgt_attention_mask,
                                           encoder_hidden_states=context,
                                           encoder_attention_mask=src_attention_mask,
                                           sub_encoder_hidden_states=sub_context,
                                           sub_encoder_attention_mask=sub_context_mask,
                                           lang=tgt_lang, atb=tgt_atb,
                                           checkpointing_ffn=checkpointing_ffn,
                                           checkpointing_cross_attn=checkpointing_cross_attn,
                                           checkpointing_self_attn=checkpointing_self_attn)
            decoder_output = decoder_outputs[0]
            contrastive_loss = decoder_outputs[-1]
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

        output_dict['wav2vec_context'] = encoder_output['wav2vec_context']
        output_dict['wav2vec_padding_mask'] = encoder_output['wav2vec_padding_mask']

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
            # run the ctcoutput via the wav2vec context (not context)
            output_dict['encoder_logits'] = self.ctc_linear(output_dict['wav2vec_context'])

        if self.sub_encoder is not None:
            # contrastive loss has size: t x b x h
            # stacked sum from multiple layers
            contrastive_loss = contrastive_loss.transpose(0, 1).contiguous()

            # the input is the target full without the final token so
            # remove the last time step from the mask
            mask = sub_context_mask[:, :-1].unsqueeze(-1)  # b x t x 1
            contrastive_loss.masked_fill_(mask, 0)  # masked values = zero

            output_dict['contrastive_loss'] = contrastive_loss.sum()

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
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')
        src_atb = batch.get('source_atbs')
        tgt_atb = batch.get('target_atbs')

        encoder_output = self.encoder(src.transpose(0, 1), batch_first_output=False,
                                      lang=src_lang, atb=src_atb)
        src_attention_mask = encoder_output['src']

        dec_pretrained_model = self.decoder.dec_pretrained_model
        if not dec_pretrained_model:
            mask_src = None
        elif dec_pretrained_model in ["bert", "roberta"]:
            mask_src = src_attention_mask.unsqueeze(1)  # batch_size  x 1 x len_src for broadcasting

        elif dec_pretrained_model in ["bart"]:
            mask_src = 1 - (src_attention_mask.long())
        elif dec_pretrained_model in ["deltalm", "mbart", "mbart50"]:
            mask_src = src_attention_mask
        else:
            print("Warning: unknown dec_pretrained_model")
            raise NotImplementedError

        decoder_state = TransformerDecodingState(src, tgt_lang, encoder_output['context'], src_lang,
                                                 beam_size=beam_size, model_size=self.model_size,
                                                 type=type, buffering=buffering, src_mask=mask_src,
                                                 dec_pretrained_model=self.decoder.dec_pretrained_model,
                                                 tgt_atb=tgt_atb)

        return decoder_state

    def tie_weights(self):
        assert self.generator is not None, "The generator needs to be created before sharing weights"
        if hasattr(self.decoder, 'dec_pretrained_model') and self.decoder.dec_pretrained_model in ["bert", "roberta"]:
            self.generator[0].linear.weight = self.decoder.embeddings.word_embeddings.weight
        elif hasattr(self.decoder, 'dec_pretrained_model') and self.decoder.dec_pretrained_model \
                in ["mbart", "mbart50", "deltalm"]:
            self.generator[0].linear.weight = self.decoder.embed_tokens.weight
        else:
            self.generator[0].linear.weight = self.decoder.word_lut.weight

    def decode(self, batch):

        raise NotImplementedError