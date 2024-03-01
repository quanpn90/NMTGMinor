import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from onmt.models.transformers import Transformer, TransformerDecodingState, TransformerDecodingStateMemory
from typing import List, Optional, Union
from collections import defaultdict
import onmt
from onmt.modules.optimized.linear import Linear
import math
from .fairseq_wav2vec2.file_io import PathManager
from omegaconf import DictConfig, open_dict, OmegaConf
from .fairseq_wav2vec2.utils import overwrite_args_by_name

import copy
import numpy as np
from onmt.modules.loss import CrossEntropyLossBase
from onmt.modules.layer_norm import LayerNorm

from itertools import groupby


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


def load_checkpoint_to_cpu(path, arg_overrides=None, load_on_all_ranks=False):
    """Loads a checkpoint to CPU (with upgrading for backward compatibility).

    If doing single-GPU training or if the checkpoint is only being loaded by at
    most one process on each node (current default behavior is for only rank 0
    to read the checkpoint from disk), load_on_all_ranks should be False to
    avoid errors from torch.distributed not having been initialized or
    torch.distributed.barrier() hanging.

    If all processes on each node may be loading the checkpoint
    simultaneously, load_on_all_ranks should be set to True to avoid I/O
    conflicts.

    There's currently no support for > 1 but < all processes loading the
    checkpoint on each node.
    """
    local_path = PathManager.get_local_path(path)
    # The locally cached file returned by get_local_path() may be stale for
    # remote files that are periodically updated/overwritten (ex:
    # checkpoint_last.pt) - so we remove the local copy, sync across processes
    # (if needed), and then download a fresh copy.
    if local_path != path and PathManager.path_requires_pathmanager(path):
        try:
            os.remove(local_path)
        except FileNotFoundError:
            # With potentially multiple processes removing the same file, the
            # file being missing is benign (missing_ok isn't available until
            # Python 3.8).
            pass
        if load_on_all_ranks:
            torch.distributed.barrier()
        local_path = PathManager.get_local_path(path)

    with open(local_path, "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))

    if "args" in state and state["args"] is not None and arg_overrides is not None:
        args = state["args"]
        for arg_name, arg_val in arg_overrides.items():
            setattr(args, arg_name, arg_val)

    if "cfg" in state and state["cfg"] is not None:

        # hack to be able to set Namespace in dict config. this should be removed when we update to newer
        # omegaconf version that supports object flags, or when we migrate all existing models

        state["cfg"] = OmegaConf.create(state["cfg"])

        OmegaConf.set_struct(state["cfg"], True)

        if arg_overrides is not None:
            overwrite_args_by_name(state["cfg"], arg_overrides)

    # state = _upgrade_state_dict(state)
    return state


# defining a Wav2vec2 encoder wrapping the HuggingFace model here
class FairseqWav2VecExtractor(nn.Module):

    def __init__(self, model_path="wav2vec_vox_new.pt"):
        self.model_path = model_path
        import fairseq
        # from fairseq.checkpoint_utils import load_model_ensemble_and_task, load_checkpoint_to_cpu
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


class FairseqWav2VecQuantizer(nn.Module):

    def __init__(self, model_path="wav2vec_vox_new.pt"):
        self.model_path = model_path
        # import fairseq
        # from fairseq.checkpoint_utils import load_model_ensemble_and_task, load_checkpoint_to_cpu
        from .fairseq_wav2vec2.wav2vec2 import Wav2Vec2Model

        super().__init__()
        state = load_checkpoint_to_cpu(model_path)

        self.cfg = state['cfg']['model']
        self.wav2vec_encoder = Wav2Vec2Model(cfg=self.cfg)
        self.wav2vec_encoder.load_state_dict(state['model'])

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
        wav2vec_output = self.wav2vec_encoder(input, attn_mask, mask=False,
                                              quantize=True, quantize_only=True,
                                              )

        codes = wav2vec_output['quantized_target']
        padding_mask = wav2vec_output['padding_mask']

        return codes, padding_mask


class FairseqWav2Vec(nn.Module):

    def __init__(self, opt, model_path="wav2vec_vox_new.pt",
                 stacked_encoder=None, **kwargs):

        super().__init__()
        # do we need opt for this?
        self.opt = opt
        self.model_path = model_path
        # import fairseq
        # from fairseq.checkpoint_utils import load_model_ensemble_and_task, load_checkpoint_to_cpu
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

        self.wav2vec_encoder = Wav2Vec2Model(cfg=self.cfg, favor=opt.favor_attention,
                                             weight_drop=opt.weight_drop,
                                             predict_language=opt.predict_language,
                                             n_languages=opt.n_languages,
                                             branchformer=opt.branchformer)
        self.favor = opt.favor_attention
        if self.favor:
            from onmt.modules.performer import ProjectionUpdater
            self.proj_updater = ProjectionUpdater(self.wav2vec_encoder.encoder,
                                                  feature_redraw_interval=1000)
            self.auto_check_redraw = True

        # load wav2vec weights
        load = True

        if load:
            wav2vec_weights = state['model']
            existed_weights = self.wav2vec_encoder.state_dict()

            # if we add new weights/buffers to new model then put them into the state_dict
            keys = existed_weights.keys()
            for key in keys:
                if key not in wav2vec_weights:
                    wav2vec_weights[key] = existed_weights[key]

            self.wav2vec_encoder.load_state_dict(state['model'])
        else:
            print("Not loading pretrained wav2vec weights")

        removing_quantizer = not opt.wav2vec2_quantize
        # remove the quantization modules
        # print("removing quantization modules", removing_quantizer)
        self.wav2vec_encoder.remove_pretraining_modules(removing_quantizer=removing_quantizer)

        cfg = self.wav2vec_encoder.cfg
        assert self.opt.model_size == cfg.encoder_embed_dim, \
            "Expect self.opt.model_size (%d) and cfg.encoder_embed_dim (%d) to equal " \
            % (self.opt.model_size, cfg.encoder_embed_dim)
        self.input_type = self.opt.encoder_type
        self.model_size = cfg.encoder_embed_dim
        self.wav2vec_encoder.feature_grad_mult = 0.0
        self.time = None
        self.quantize = opt.wav2vec2_quantize
        self.dual_output = opt.wav2vec2_dual_output and self.quantize

        if stacked_encoder is not None:
            self.wav2vec_encoder.add_stacked_encoder(stacked_encoder)

        # freezing the parameters of the Convolutional feature extractors (by default)
        for param in self.wav2vec_encoder.feature_extractor.parameters():
            param.requires_grad = False

        # TODO:
        # add relative attention
        if (hasattr(opt, 'wav2vec2_relative_attention') and opt.wav2vec2_relative_attention) or \
                (hasattr(opt, 'add_relative_attention') and opt.add_relative_attention):
            print("[INFO] Add relative attention for wav2vec")
            self.wav2vec_encoder.add_relative_attention()

        self.rotary_position_encoding = opt.rotary_position_encoding
        if self.rotary_position_encoding:
            assert not (hasattr(opt, 'wav2vec2_relative_attention') and opt.wav2vec2_relative_attention)
            self.wav2vec_encoder.add_rotary_attention()

        # freeze the whole encoder. needs to do this first before adding customized parameters
        if opt.freeze_encoder:
            print("[INFO] Freezing encoder parameters")
            for p in self.wav2vec_encoder.parameters():
                p.requires_grad = False

        if opt.freeze_encoder_ffn:
            self.freeze_ffn_params()

        # then add factorize
        if opt.multilingual_factorized_weights:
            print("[INFO] Factorizing Wav2vec model into %d languages and %d factors"
                  % (opt.n_languages, opt.n_attributes))
            self.wav2vec_encoder.encoder.add_factorize(opt.n_languages, rank=opt.mfw_rank,
                                                       multiplicative=opt.mfw_multiplicative,
                                                       flexible=opt.flex_factorize,
                                                       fast=opt.fast_factorize)

        self.predict_language = self.wav2vec_encoder.predict_language

        # or adapter
        if opt.wav2vec_adapter > 0:
            print("[INFO] Adding adapters for Wav2vec model with %d languages" % opt.n_languages)
            self.wav2vec_encoder.encoder.add_adapters(opt.n_languages, adapter_location=opt.wav2vec_adapter)

        # can receive an mbart or deltalm encoder
        # self.stacked_encoder = stacked_encoder
        # TODO: length conversion layer

        # if stacked_encoder is not None:

        # self.stacked_encoder = stacked_encoder
        # self.conv_downsampler =  nn.ModuleList()
        #
        # from .fairseq_wav2vec2.fairseq_modules import TransposeLast
        # from onmt.modules.layer_norm import LayerNorm
        # for i in range(3):
        #
        #     def make_conv(n_in, n_out, k, stride=2, padding=1):
        #         conv = nn.Conv1d(n_in, n_out, k, stride=stride, padding=padding, bias=False)
        #         torch.nn.init.kaiming_normal_(conv.weight)
        #         return conv
        #
        #     conv = nn.Sequential(
        #         make_conv(self.model_size, self.model_size, 4, stride=2, padding=1),
        #         nn.Sequential(
        #             TransposeLast(),
        #             LayerNorm(self.model_size),
        #             TransposeLast(),
        #         ),
        #         nn.GELU(),
        #     )
        #
        #     self.conv_downsampler.append(conv)

        else:
            self.stacked_encoder = None
            self.conv_downsampler = None

        # discrete encoder that works on top of the wav quantized output
        # self.discrete_encoder = None # discrete_encoder

        # if self.quantize:
        #     var_dim = self.wav2vec_encoder.quantizer.vars.size(-1) * self.wav2vec_encoder.quantizer.groups
        #     model_dim = self.model_size
        #     self.discrete_encoder = nn.Linear(var_dim, model_dim)
        # if discrete_encoder is not None:
        #     assert self.quantize is True
        #
        #     codebook_size = self.wav2vec_encoder.quantizer.num_vars ** self.wav2vec_encoder.quantizer.groups
        #     embed_dim = self.discrete_encoder.embed_dim
        #     var_dim = self.wav2vec_encoder.quantizer.vars.size(-1) * self.wav2vec_encoder.quantizer.groups
        #     # new embedding layer
        #     # self.discrete_encoder.embed_tokens =  nn.Linear(var_dim, embed_dim) #nn.Embedding(codebook_size, embed_dim)
        #     self.discrete_encoder.embed_tokens = nn.Embedding(codebook_size, embed_dim)
        #     nn.init.normal_(self.discrete_encoder.embed_tokens.weight, 0.0, 0.02)
        #
        #     # freeze the quantizer
        #     for param in self.wav2vec_encoder.quantizer.parameters():
        #         param.requires_grad = False
        #
        #     for param in self.wav2vec_encoder.layer_norm.parameters():
        #         param.requires_grad = False

    def fix_projection_matrices_(self):
        if self.favor:
            self.proj_updater.fix_projections_()

    def convert_fast_attention(self):
        self.wav2vec_encoder.convert_fast_attention()

    def freeze_ffn_params(self):
        for layer in self.wav2vec_encoder.encoder.layers:
            for p in layer.fc1.parameters():
                p.requires_grad = False
            for p in layer.fc2.parameters():
                p.requires_grad = False

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
                lang=None, atb=None,
                checkpointing_ffn=False, checkpointing_self_attn=False, **kwargs):
        """
        :param checkpointing_self_attn:
        :param checkpointing_ffn:
        :param atb:
        :param lang:
        :param input_ptb: perturbation added to the input itself
        :param adv_ptb_grad: adversarial perturbation step which we need the gradients w.r.t the input (wavs)
        :param batch_first_output: [bsz, seq_len, hidden_size] as output size, else transpose(0, 1)
        :param input: torch.Tensor [batch_size, sequence_length, 2]
        :param kwargs:
        :return:
        """

        # The data has been constructed that the first dimension is padding mask
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

        quantize_only = False  # self.quantize and not self.dual_output
        # don't mask when precomputed tdnn is used, because spec augmentation is used in the dataset

        wav2vec_output = self.wav2vec_encoder(input, attn_mask,
                                              mask=self.training, features_only=True, layer=None,
                                              precomputed_tdnn=precomputed_tdnn, quantize=self.quantize,
                                              quantize_only=quantize_only,
                                              lang=lang, atb=atb,
                                              checkpointing_ffn=checkpointing_ffn,
                                              checkpointing_self_attn=checkpointing_self_attn)

        # output size is always T x B x C
        continuous_output = wav2vec_output['x']
        time, batch_size = continuous_output.size(0), continuous_output.size(1)

        # mask size is B x T (1 for padded positions, 0 for unpadded)
        dec_attn_mask = wav2vec_output['padding_mask']

        if self.quantize:
            quantized_output = wav2vec_output['quantized_x']
            discrete_output = self.discrete_encoder(quantized_output)
            discrete_output = discrete_output.transpose(0, 1).contiguous()

            context = continuous_output + discrete_output
        else:
            context = continuous_output

        if dec_attn_mask is None:
            dec_attn_mask = context.new_zeros(batch_size, time).byte()
        else:
            dec_attn_mask = dec_attn_mask.byte()

        wav2vec_context = context
        wav2vec_padding_mask = dec_attn_mask

        # # TODO: make the stacked encoder run here
        # if self.stacked_encoder is not None:
        #     # assert self.conv_downsampler is not None
        #     #
        #     # # T x B x C -> B x C x T
        #     # context = context.transpose(0, 1).transpose(1, 2).contiguous()
        #     #
        #     # # apply convolutions to downsample the size
        #     # for conv in self.conv_downsampler:
        #     #     context = conv(context)
        #     #
        #     # # B x C x T -> B x T x C
        #     # context = context.transpose(1, 2).contiguous()
        #     #
        #     # padding_mask = dec_attn_mask
        #     #
        #     # # TODO: recompute the padding_mask from length
        #     # with torch.no_grad():
        #     #     input_lengths = (1 - padding_mask.long()).sum(-1)
        #     #
        #     #     def _conv_out_length(input_length, conv):
        #     #         kernel_size = conv.kernel_size[0]
        #     #         stride = conv.kernel_size[0]
        #     #         padding = conv.padding[0]
        #     #
        #     #         return torch.floor((input_length - kernel_size + 2 * padding) / stride + 1)
        #     #
        #     #     for conv_block in self.conv_downsampler:
        #     #         input_lengths = _conv_out_length(
        #     #             input_lengths, conv_block[0]
        #     #         )
        #     #
        #     #     input_lengths = input_lengths.to(torch.long)
        #     #
        #     #     padding_mask = torch.zeros(
        #     #         context.shape[:2], dtype=context.dtype, device=context.device
        #     #     )
        #     #
        #     #     padding_mask[
        #     #         (
        #     #             torch.arange(padding_mask.shape[0], device=padding_mask.device),
        #     #             input_lengths - 1,
        #     #         )
        #     #     ] = 1
        #     #
        #     #     padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        #     #
        #     # dec_attn_mask = padding_mask
        #     context = context.transpose(0, 1).contiguous()
        #
        #     # run the output through the stacked encoder
        #     stacked_encoder_output = self.stacked_encoder(inputs_embeds=context, attention_mask=dec_attn_mask,
        #                                                   checkpointing_ffn=checkpointing_ffn)
        #     context = stacked_encoder_output[0]

        # how to get the correct attention mask?
        output_dict = defaultdict(lambda: None, {'source': input, 'context': context, 'src_mask': dec_attn_mask,
                                                 'src': dec_attn_mask, 'pos_emb': None,
                                                 'wav2vec_context': wav2vec_context,
                                                 'wav2vec_padding_mask': wav2vec_padding_mask,
                                                 'enc_pred_lang': wav2vec_output['pred_lang']})

        return output_dict


class Wav2vecTransformer(Transformer):
    """Main model in 'Attention is all you need' """

    def __init__(self, encoder, decoder, generator=None,
                 mirror=False, ctc=False, **kwargs):
        super().__init__(encoder, decoder, generator, None, None, ctc=ctc)
        self.model_size = self.decoder.model_size
        self.switchout = self.decoder.switchout
        self.has_decoder = True

        if mirror:
            self.mirror_decoder = copy.deepcopy(self.decoder)
            self.mirror_g = nn.Linear(decoder.model_size, decoder.model_size)
            self.mirror_generator = copy.deepcopy(self.generator)
            self.mirror_generator[0].linear.weight = self.decoder.word_lut.weight

        if self.ctc:
            self.ctc_linear = Linear(encoder.model_size, self.tgt_vocab_size)

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

    def create_decoder_state(self, batch, beam_size=1, type=2, buffering=True,
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
        tgt_atb = batch.get('target_atbs')
        src_atb = batch.get('source_atbs')
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
        super().__init__(encoder, decoder, generator, mirror=mirror, ctc=False)

        self.src_vocab_size = 0
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.sub_encoder = sub_encoder
        self.model_size = encoder.model_size

        # this model cannot use CTC
        self.ctc = False
        self.ctc_compress = False

        self.has_decoder = True
        if hasattr(decoder, 'dec_pretrained_model') and decoder.dec_pretrained_model == "none":
            self.has_decoder = False

        if self.has_decoder:
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
        else:
            self.ctc = True
            # we need to make a ctc linear over here
            self.tgt_vocab_size = self.generator[0].linear.weight.size(0)
            self.model_size = self.generator[0].hidden_size
            self.ctc_linear = self.generator[0].linear
            self.ctc_compress = None

            # delete the generator
            self.generator = nn.ModuleList()

            # if ctc_compress != "None":
            #     from .ctc_compressor import CTCCompressStrategy
            #     self.ctc_compress = getattr(CTCCompressStrategy, ctc_compress)
            # else:
            #     self.ctc_compress = None

        if mirror:
            self.mirror_decoder = copy.deepcopy(self.decoder)
            self.mirror_g = nn.Linear(decoder.model_size, decoder.model_size)
            self.mirror_generator = copy.deepcopy(self.generator)
            self.mirror_generator[0].linear.weight = self.decoder.word_lut.weight

    # def create_ctc_char(self, char_data, ctc_compress="None"):
    #
    #     id2char = char_data['id2char']
    #     char_vocab_size = len(id2char)
    #     self.char_vocab_size = char_vocab_size
    #     self.char_ctc_linear = nn.Linear(self.model_size, char_vocab_size)
    #     print(self.char_vocab_size)
    #
    #     if ctc_compress != "None":
    #         from .ctc_compressor import CTCCompressStrategy
    #         self.ctc_compress = getattr(CTCCompressStrategy, ctc_compress)
    #     else:
    #         self.ctc_compress = None
    #
    #     self.ctc_char = True

    def forward(self, batch, zero_encoder=False, factorize=False, target_mask=None, mirror=False,
                checkpointing_ffn=False,
                checkpointing_cross_attn=False,
                checkpointing_self_attn=False,
                ctc_loss_function=None,
                ctc_labels=None,
                grad_scaler=None,
                ctc_coeff=None,
                **kwargs):
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
        :param ctc_coeff:
        :param grad_scaler:
        :param ctc_labels:
        :param ctc_loss_function:
        :return:
        """
        if self.switchout > 0 and self.training:
            batch.switchout(self.switchout, self.src_vocab_size, self.tgt_vocab_size)

        src = batch.get('source')
        tgt = batch.get('target_input')
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

        # during training mixture is always None
        encoder_output = self.encoder(src, batch_first_output=batch_first_output,
                                      lang=src_lang, atb=src_atb,
                                      checkpointing_ffn=checkpointing_ffn,
                                      checkpointing_self_attn=checkpointing_self_attn)

        encoder_output = defaultdict(lambda: None, encoder_output)

        context = encoder_output['context']
        src_attention_mask = encoder_output['src']
        contrastive_loss = 0

        if ctc_coeff > 0 or not self.has_decoder:
            # what is the ctc_labels here?
            ctc_labels = batch.get("target_output")
            assert (ctc_loss_function.padding_idx == onmt.constants.TGT_PAD)

            # we have to perform CTC first

            # compute the logits for each encoder step
            # run the ctcoutput via the wav2vec context (not context)
            # ctc output should have the mbart vocabulary
            # encoder_hidden = output_dict['wav2vec_context'].
            # output_dict['encoder_logits'] = self.ctc_linear(output_dict['wav2vec_context'])
            # how should we proceed from this?

            encoder_logits = self.ctc_linear(context)

            ctc_loss_inputs = dict()
            ctc_loss_inputs['encoder_logits'] = encoder_logits
            ctc_loss_inputs['wav2vec_padding_mask'] = encoder_output['wav2vec_padding_mask']
            ctc_loss_inputs['src_mask'] = encoder_output['src']

            ctc_loss, n_ctc_targets = ctc_loss_function(ctc_loss_inputs, ctc_labels)
            ctc_loss = ctc_loss * ctc_coeff

            # backward immediately and accumulate gradients into the context.grad

            ctc_loss_data = ctc_loss.item()

            if self.ctc_compress is not None:
                #     # TODO: Ctc compression
                with torch.no_grad():
                    x_ctc = encoder_logits
                    batch_predicted = []
                    prob_ctc = F.softmax(x_ctc, dim=-1).transpose(0, 1)  # from T x B x D to B x T x D
                    for b in range(prob_ctc.shape[0]):
                        predicted = prob_ctc[b][: src_lengths[b]].argmax(-1).tolist()
                        batch_predicted.append([(p[0], len(list(p[1]))) for p in groupby(predicted)])

                    new_lengths = [len(p) for p in batch_predicted]

                    # TODO: compress_method
                    weights_matrix = self.ctc_compress(prob_ctc, batch_predicted, new_lengths, x_ctc.dtype,
                                                       x_ctc.device)

                context = context_detached.permute(1, 2, 0).bmm(weights_matrix).permute(2, 0, 1)

                # creating a new padding mask
                max_len = max(new_lengths)
                _src_mask = context.new_zeros(len(new_lengths), max_len).bool()
                for i, l in enumerate(new_lengths):
                    _src_mask[i, l:] = 1

                src_attention_mask = _src_mask

        else:
            ctc_loss = None
            ctc_loss_data = None
            n_ctc_targets = 0

        # TODO2: sub_encoder (MBART Encoder)
        if self.sub_encoder is not None:
            sub_encoder_outputs = self.sub_encoder(inputs_embeds=context.transpose(0, 1).contiguous(),
                                                   attention_mask=src_attention_mask)

            context = sub_encoder_outputs[0]

        if not self.has_decoder:
            output_dict = defaultdict(lambda: None)

            output_dict['hidden'] = None
            output_dict['context'] = context
            output_dict['context_origin'] = context
            output_dict['src_mask'] = src_attention_mask
            output_dict['src'] = src
            output_dict['target_mask'] = target_mask
            output_dict['target'] = batch.get('target_output')
            output_dict['ctc_loss'] = ctc_loss
            output_dict['n_ctc_targets'] = n_ctc_targets

            output_dict['wav2vec_context'] = encoder_output['wav2vec_context']
            output_dict['wav2vec_padding_mask'] = encoder_output['wav2vec_padding_mask']
            output_dict['enc_pred_lang'] = encoder_output['enc_pred_lang']

            if output_dict['enc_pred_lang'] is not None:
                output_dict['dec_pred_lang'] = decoder_outputs[-1]

            return output_dict


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
                in ["deltalm", "mbart", "mbart50"]:  # TODO: NLLB

            src_attention_mask = src_attention_mask  # new version
            # tgt_attention_mask = tgt.ne(onmt.constants.TGT_PAD).long()  # [bsz, len]
            # tgt_attention_mask = tgt.new(*tgt.size()).fill_(1)
            tgt_attention_mask = batch.get('target_input_selfattn_mask')

            if encoder_output['enc_pred_lang'] is not None:
                _src_lang = torch.nn.functional.softmax(encoder_output['enc_pred_lang'], dim=-1, dtype=torch.float32)
            else:
                _src_lang = src_lang

            decoder_outputs = self.decoder(input_ids=tgt,
                                           attention_mask=tgt_attention_mask,
                                           encoder_hidden_states=context,
                                           encoder_attention_mask=src_attention_mask,
                                           sub_encoder_hidden_states=None,
                                           sub_encoder_attention_mask=None,
                                           lang=tgt_lang, atb=tgt_atb,
                                           src_lang=_src_lang,
                                           checkpointing_ffn=checkpointing_ffn,
                                           checkpointing_cross_attn=checkpointing_cross_attn,
                                           checkpointing_self_attn=checkpointing_self_attn)
            decoder_output = decoder_outputs[0]
            # contrastive_loss = decoder_outputs[-1]

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
        output_dict['src_mask'] = src_attention_mask
        output_dict['src'] = src
        output_dict['target_mask'] = target_mask
        output_dict['target'] = batch.get('target_output')
        output_dict['ctc_loss'] = ctc_loss_data
        output_dict['n_ctc_targets'] = n_ctc_targets

        output_dict['wav2vec_context'] = encoder_output['wav2vec_context']
        output_dict['wav2vec_padding_mask'] = encoder_output['wav2vec_padding_mask']
        output_dict['enc_pred_lang'] = encoder_output['enc_pred_lang']

        if output_dict['enc_pred_lang'] is not None:
            output_dict['dec_pred_lang'] = decoder_outputs[-1]

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

        return output_dict

    def post_backward(self, output_dict=None, grad_scaler=None, *args, **kwargs):

        # if self.ctc_char:
        #     org_context = output_dict['context_origin']
        #     context_grad = output_dict['context'].grad
        #
        #     org_context.backward(gradient=context_grad)
        #     del output_dict['context']
        #     del output_dict

        return

    def create_decoder_state(self, batch, beam_size=1, type=1,
                             buffering=True, dicts=None, **kwargs):
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

        encoder_context = encoder_output['context']
        src_attention_mask = encoder_output['src']

        context = encoder_output['context']

        if self.ctc_char:
            ctc_logits = self.char_ctc_linear(encoder_context)
            ctc_outputs = torch.nn.functional.log_softmax(ctc_logits, dim=-1)

            prob_ctc = ctc_outputs.transpose(0, 1).contiguous()
            bsz = prob_ctc.size(0)

            src_mask = encoder_output['src']
            src_lengths = (1 - src_mask.long()).sum(dim=1)
            print("Input Lengths", src_lengths)

            for b in range(bsz):
                predicted = prob_ctc[b][: src_lengths[b]].argmax(-1).tolist()

                if dicts is not None:
                    id2char = dicts['char_data']['id2char']
                    char_predicted = [id2char[_id] for _id in predicted]

                #     print(char_predicted)
                # else:
                #     print(predicted)

                # ctc compression

            if self.ctc_compress is not None:
                with torch.no_grad():
                    x_ctc = ctc_logits
                    batch_predicted = []
                    prob_ctc = F.softmax(x_ctc, dim=-1).transpose(0, 1)  # from T x B x D to B x T x D
                    for b in range(prob_ctc.shape[0]):
                        predicted = prob_ctc[b][: src_lengths[b]].argmax(-1).tolist()
                        batch_predicted.append([(p[0], len(list(p[1]))) for p in groupby(predicted)])

                    new_lengths = [len(p) for p in batch_predicted]

                    # TODO: compress_method
                    weights_matrix = self.ctc_compress(prob_ctc, batch_predicted, new_lengths, x_ctc.dtype,
                                                       x_ctc.device)

                context = context.permute(1, 2, 0).bmm(weights_matrix).permute(2, 0, 1)

                # creating a new padding mask
                max_len = max(new_lengths)
                _src_mask = context.new_zeros(len(new_lengths), max_len).bool()
                for i, l in enumerate(new_lengths):
                    _src_mask[i, l:] = 1

                src_attention_mask = _src_mask

        if self.sub_encoder is not None:
            sub_encoder_outputs = self.sub_encoder(inputs_embeds=context.transpose(0, 1).contiguous(),
                                                   attention_mask=src_attention_mask)

            context = sub_encoder_outputs[0]

        if hasattr(self.encoder, 'predict_language') and self.encoder.predict_language > 0:
            print("CREATING DECODER STATE with predictive source language...")
            pred_lang = encoder_output['enc_pred_lang']  # needs to indicate that this is only logits
            src_lang = torch.nn.functional.softmax(pred_lang, dim=-1, dtype=torch.float32).transpose(0, 1).contiguous()
            src_lang = torch.argmax(src_lang, dim=-1)
            src_lang = torch.zeros_like(pred_lang).transpose(0, 1).contiguous().scatter_(2, src_lang.unsqueeze(2), 1.)

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

        decoder_state = TransformerDecodingState(src, tgt_lang, context, src_lang,
                                                 beam_size=beam_size, model_size=self.model_size,
                                                 type=type, buffering=buffering, src_mask=mask_src,
                                                 dec_pretrained_model=self.decoder.dec_pretrained_model,
                                                 tgt_atb=tgt_atb)

        return decoder_state

    def tie_weights(self):
        if not self.has_decoder:
            return

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


def my_loss(lprobs, label, mask_no_mem, mask_mem):
    loss = -lprobs.gather(2, label.unsqueeze(-1))[:, :, 0]
    loss1 = loss[mask_no_mem].sum()
    loss2 = loss[mask_mem].sum()
    correct = lprobs.argmax(-1).eq(label)
    correct1 = correct[mask_no_mem].sum().item()
    correct2 = correct[mask_mem].sum().item()
    anz1 = mask_no_mem.sum().item()
    anz2 = mask_mem.sum().item()
    return loss1, correct1, anz1, loss2, correct2, anz2


class Wav2vecBERTMemory(Wav2vecBERT):
    def __init__(self, *args, opt=None, **kwargs):
        super().__init__(*args, **kwargs)

        self._no_entry_found = nn.Parameter(torch.randn(1, 1024))

        from pretrain_module.configuration_mbart import MBartConfig
        from pretrain_module.modeling_mbart import MBartEncoder

        enc_mbart_config = MBartConfig.from_json_file(opt.enc_config_file)
        self.memory_encoder = MBartEncoder(enc_mbart_config, opt)

        del self.memory_encoder.embed_tokens

        if opt.enc_state_dict:
            enc_model_state_dict = torch.load(opt.enc_state_dict, map_location="cpu")
            self.memory_encoder.load_state_dict(enc_model_state_dict, strict=False)
        else:
            print("Not loading pretrained mbart encoder weights for memory decoder")

        layers = []
        for layer_id in np.linspace(0, len(self.memory_encoder.layers) - 1, num=opt.encoder_layers_memory,
                                    dtype=np.int64):
            layers.append(copy.deepcopy(self.memory_encoder.layers[layer_id]))
        self.memory_encoder.layers = nn.ModuleList(layers)

        print("Using", len(self.memory_encoder.layers), "memory encoder layers")

        self.memory_cache = []  # distractors to memory from last batch(es)
        self.memory_cache_length = 0
        self.memory_size_max = opt.memory_size_max
        self.memory_cache_device = "cpu"

        self.memory_stats = torch.zeros(len(self.decoder.memory_decoder.layers) + 1, 6, device="cuda")
        self.counter_print = -1

        self.is_main = False
        self.memory_tokens_weighted_equally = True
        self.memory_loss_coeff = opt.memory_loss_coeff

    def train(self, mode=True):
        super().train(mode)
        if mode:  # after evaluation and before next training
            s = self.memory_stats
            loss_no_mem = s[-1, 0].sum()
            loss_mem = s[-1, 3].sum()
            anz_no_mem = s[-1, 2].sum()
            anz_mem = s[-1, 5].sum()
            if not self.memory_tokens_weighted_equally:
                ppl = math.exp((loss_no_mem + loss_mem) / (anz_no_mem + anz_mem))
            else:
                ppl_no_mem = math.exp(loss_no_mem / anz_no_mem)
                ppl_mem = math.exp(loss_mem / anz_mem)
                ppl = (ppl_no_mem + ppl_mem) / 2
            self.choose_best_epoch_by = ppl

            self.print()
            self.memory_cache = []

    def eval(self):
        self.memory_cache = []
        self.memory_stats.zero_()
        self.counter_print = -1
        super().eval()

    def print(self):
        if self.is_main:
            # loss_no_mem,correct_no_mem,anz_no_mem,loss_mem,correct_mem,anz_mem
            for i in range(self.memory_stats.shape[0]):
                s = self.memory_stats[i]
                ppl_no_mem = math.exp(s[0] / s[2])
                ppl_mem = math.exp(s[3] / s[5])
                if not self.memory_tokens_weighted_equally:
                    ppl_all = math.exp((s[0] + s[3]) / (s[2] + s[5]))
                else:
                    ppl_all = (ppl_no_mem + ppl_mem) / 2
                acc_no_mem = 100 * s[1] / s[2]
                acc_mem = 100 * s[4] / s[5]
                if not self.memory_tokens_weighted_equally:
                    acc_all = 100 * (s[1] + s[4]) / (s[2] + s[5])
                else:
                    acc_all = (acc_no_mem + acc_mem) / 2
                if i < self.memory_stats.shape[0] - 1:
                    print(
                        "    Layer %2d: ppl no_mem,mem,all: %6.2f,%6.2f,%6.2f, acc no_mem,mem,all: %6.2f,%6.2f,%6.2f" % (
                        i,
                        ppl_no_mem, ppl_mem, ppl_all, acc_no_mem, acc_mem, acc_all))
                else:
                    print("    Ntp:      ppl no_mem,mem,all: %6.2f,%6.2f,%6.2f, acc no_mem,mem,all: %6.2f,%6.2f,%6.2f" %
                          (ppl_no_mem, ppl_mem, ppl_all, acc_no_mem, acc_mem, acc_all))

        self.memory_stats.zero_()

    def add_memory_stats(self, i, stats):
        for j, s in enumerate(stats):
            self.memory_stats[i, j] += s
        if self.training and i == self.memory_stats.shape[0] - 1:
            self.counter_print += 1
            if self.counter_print % 500 == 0:
                self.print()

    @property
    def no_entry_found(self):
        nef = self._no_entry_found.to(torch.float32)
        m, s = nef.mean(), nef.std()
        return (nef - m) / s

    def encode_memory(self, memory_text_embeds, memory_text_mask):
        if memory_text_embeds is None:
            return self.no_entry_found, None

        lengths = memory_text_mask.eq(0).sum(1).unsqueeze(1)  # n_mem x 1

        memory_text_enc = self.memory_encoder(inputs_embeds=memory_text_embeds, attention_mask=memory_text_mask)[0]
        if memory_text_enc is not None:
            memory_text_enc[memory_text_mask.transpose(1, 0)] = 0

        encoder_output_memory_wonef = memory_text_enc.sum(0) / lengths  # n_mem x d_model

        while len(self.memory_cache) > 0 and self.memory_cache_length + encoder_output_memory_wonef.shape[
            0] + 1 > self.memory_size_max > 0:
            self.memory_cache_length -= self.memory_cache[0].shape[0]
            self.memory_cache = self.memory_cache[1:]

        encoder_output_memory = torch.cat([self.no_entry_found, encoder_output_memory_wonef]
                                          + [c.cuda() for c in self.memory_cache], 0)  # (n_mem+1) x d_model

        if self.training and self.memory_size_max > 0:
            self.memory_cache.append(encoder_output_memory_wonef.detach().to(self.memory_cache_device))
            self.memory_cache_length += self.memory_cache[-1].shape[0]

        return encoder_output_memory, memory_text_enc

    def run_encoder(self, src, batch_first_output=False,  # src: l_src1 x b x d_model
                    src_lang=None, src_atb=None,
                    checkpointing_ffn=False,
                    checkpointing_self_attn=False):
        # dict with keys context: l_src2 x b x d_model, src_mask: b x l_src2
        return self.encoder(src.transpose(0, 1), batch_first_output=batch_first_output,
                            lang=src_lang, atb=src_atb,
                            checkpointing_ffn=checkpointing_ffn,
                            checkpointing_self_attn=checkpointing_self_attn)

    def forward(self, batch, zero_encoder=False, factorize=False, target_mask=None, mirror=False,
                checkpointing_ffn=False,
                checkpointing_cross_attn=False,
                checkpointing_self_attn=False,
                **kwargs):
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
        tgt_pos = batch.get('target_pos')
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')
        src_atb = batch.get('source_atbs')
        tgt_atb = batch.get('target_atbs')
        src_lengths = batch.src_lengths
        tgt_lengths = batch.tgt_lengths

        org_src = src
        org_tgt = tgt
        tgt = tgt.transpose(0, 1)  # transpose to have batch first

        batch_first_output = False
        if hasattr(self.decoder, 'dec_pretrained_model') and self.decoder.dec_pretrained_model in ["bart"]:
            batch_first_output = True

        # print(src_lang, src_atb, tgt_lang, tgt_atb)

        if src is not None:
            encoder_output = self.run_encoder(src, batch_first_output, src_lang, src_atb,
                                              checkpointing_ffn, checkpointing_self_attn)
            context, src_attention_mask = encoder_output['context'], encoder_output['src_mask']
        else:
            context, src_attention_mask = batch.get("src_features"), batch.get("src_features_mask")
            encoder_output = {"context": context, "src": src_attention_mask}
        encoder_output = defaultdict(lambda: None, encoder_output)

        contrastive_loss = 0

        memory_text_ids = batch.get('memory_text_ids')
        memory_text_embeds, memory_text_mask = self.decoder.calc_token_embedding(memory_text_ids)

        encoder_output_memory, memory_text_enc = self.encode_memory(memory_text_embeds, memory_text_mask)

        if hasattr(self.decoder, 'dec_pretrained_model') and self.decoder.dec_pretrained_model in ["bert", "roberta"]:
            raise NotImplementedError
        elif hasattr(self.decoder, 'dec_pretrained_model') and self.decoder.dec_pretrained_model in ["bart"]:
            raise NotImplementedError
        elif hasattr(self.decoder, 'dec_pretrained_model') and self.decoder.dec_pretrained_model \
                in ["deltalm", "mbart", "mbart50"]:

            sub_context = None
            sub_context_mask = None

            src_attention_mask = src_attention_mask  # new version
            # tgt_attention_mask = tgt.ne(onmt.constants.TGT_PAD).long()  # [bsz, len]
            # tgt_attention_mask = tgt.new(*tgt.size()).fill_(1)
            tgt_attention_mask = batch.get('target_input_selfattn_mask')

            if encoder_output['enc_pred_lang'] is not None:
                _src_lang = torch.nn.functional.softmax(encoder_output['enc_pred_lang'], dim=-1, dtype=torch.float32)
            else:
                _src_lang = src_lang

            decoder_outputs = self.decoder(input_ids=tgt,
                                           attention_mask=tgt_attention_mask,
                                           encoder_hidden_states=context,
                                           encoder_attention_mask=src_attention_mask,
                                           sub_encoder_hidden_states=sub_context,
                                           sub_encoder_attention_mask=sub_context_mask,
                                           lang=tgt_lang, atb=tgt_atb,
                                           src_lang=_src_lang,
                                           checkpointing_ffn=checkpointing_ffn,
                                           checkpointing_cross_attn=checkpointing_cross_attn,
                                           checkpointing_self_attn=checkpointing_self_attn,
                                           memory_text_enc=memory_text_enc,
                                           memory_text_mask=memory_text_mask,
                                           encoder_output_memory=encoder_output_memory)
            decoder_output = decoder_outputs[0]
            decoder_output_memory = decoder_outputs[1]
            all_cross_attn_weights = decoder_outputs[2]

            # contrastive_loss = decoder_outputs[-1]

            output = decoder_output
            output_dict = defaultdict(lambda: None)

        else:
            raise NotImplementedError

        output_dict['hidden'] = output
        output_dict['context'] = context
        output_dict['src_mask'] = encoder_output['src']
        output_dict['src'] = src
        output_dict['target_mask'] = target_mask
        output_dict['target'] = batch.get('target_output')

        output_dict['wav2vec_context'] = encoder_output['wav2vec_context']
        output_dict['wav2vec_padding_mask'] = encoder_output['wav2vec_padding_mask']
        output_dict['enc_pred_lang'] = encoder_output['enc_pred_lang']

        if output_dict['enc_pred_lang'] is not None:
            output_dict['dec_pred_lang'] = decoder_outputs[-1]

        # final layer: computing softmax
        logits = self.generator[0](output_dict)['logits']

        if decoder_output_memory is not None:
            output_dict2 = {'hidden': decoder_output_memory, 'target_mask': target_mask}
            logits_memory = self.generator[0](output_dict2)['logits']

            all_weights = [F.softmax(cross_attn_weights.detach(), -1)[:, :, 0:1] for cross_attn_weights in
                           all_cross_attn_weights]
            weights = torch.cat(all_weights, -1).mean(-1, keepdim=True)  # L x B x 1

            probs = F.softmax(logits, -1)
            probs_memory = F.softmax(logits_memory, -1)

            probs = weights * probs + (1 - weights) * probs_memory  # L x B x n_vocab
            logits = torch.log(probs)

            label_ntp = batch.get("target_output")
            label_mem = batch.get("label_mem")

            if label_ntp is not None and label_mem is not None:
                mask_no_mem = label_mem.eq(0)
                mask_mem = label_mem.gt(0)
                label_mem.clamp_(min=0)

                loss_memory = 0
                for i, cross_attn_weights in enumerate(all_cross_attn_weights):
                    ca_lprobs = F.log_softmax(cross_attn_weights, -1)

                    loss_no_mem, correct_no_mem, anz_no_mem, loss_mem, correct_mem, anz_mem = my_loss(ca_lprobs,
                                                                                                      label_mem,
                                                                                                      mask_no_mem,
                                                                                                      mask_mem)

                    if not self.memory_tokens_weighted_equally:
                        loss = loss_no_mem + loss_mem
                    else:
                        loss = (loss_no_mem / anz_no_mem + loss_mem / anz_mem) * (anz_no_mem + anz_mem)

                    self.add_memory_stats(i,
                                          [loss_no_mem.item(), correct_no_mem, anz_no_mem, loss_mem.item(), correct_mem,
                                           anz_mem])

                    loss_memory += loss

                loss_no_mem, correct_no_mem, anz_no_mem, loss_mem, correct_mem, anz_mem = my_loss(logits, label_ntp,
                                                                                                  mask_no_mem, mask_mem)
                self.add_memory_stats(len(all_cross_attn_weights),
                                      [loss_no_mem.item(), correct_no_mem, anz_no_mem, loss_mem.item(), correct_mem,
                                       anz_mem])

                if not self.memory_tokens_weighted_equally:
                    loss_ntp = loss_no_mem + loss_mem
                else:
                    loss_ntp = (loss_no_mem / anz_no_mem + loss_mem / anz_mem) * (anz_no_mem + anz_mem)

                # print(self.memory_loss_coeff * loss_memory, loss_ntp)
                loss_memory = self.memory_loss_coeff * loss_memory + loss_ntp

                output_dict['loss_memory'] = loss_memory

        output_dict['logprobs'] = logits

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

        if hasattr(self.encoder, 'predict_language') and self.encoder.predict_language > 0:
            print("predicting language for the source ...")
            pred_lang = encoder_output['pred_lang']
            src_lang = torch.nn.functional.softmax(pred_lang, dim=-1, dtype=torch.float32)

        src_attention_mask = encoder_output['src']

        memory_text_ids = batch.get('memory_text_ids')
        memory_text_embeds, memory_text_mask = self.decoder.calc_token_embedding(memory_text_ids)

        encoder_output_memory, memory_text_enc = self.encode_memory(memory_text_embeds, memory_text_mask)

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

        decoder_state = TransformerDecodingStateMemory(src, tgt_lang, encoder_output['context'], src_lang,
                                                       beam_size=beam_size, model_size=self.model_size,
                                                       type=type, buffering=False, src_mask=mask_src,
                                                       dec_pretrained_model=self.decoder.dec_pretrained_model,
                                                       tgt_atb=tgt_atb,
                                                       encoder_output_memory=encoder_output_memory,
                                                       memory_text_enc=memory_text_enc,
                                                       memory_text_mask=memory_text_mask)

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

        output_dict, output_dict_memory, all_cross_attn_weights = self.decoder.step(input_t, decoder_state,
                                                                                    streaming=streaming)
        output_dict['src'] = decoder_state.src.transpose(0, 1)

        logits = self.generator[0](output_dict)['logits']  # 1 x B x n_vocab
        logits_memory = self.generator[0](output_dict_memory)['logits']  # 1 x B x n_vocab

        all_weights = [F.softmax(cross_attn_weights.detach(), -1)[-1:, :, 0:1] for cross_attn_weights in
                       all_cross_attn_weights]
        weights = torch.cat(all_weights, -1).mean(-1, keepdim=True)  # 1 x B x 1

        probs = F.softmax(logits, -1)  # 1 x B x n_vocab
        probs_memory = F.softmax(logits_memory, -1)  # 1 x B x n_vocab

        probs = weights * probs + (1 - weights) * probs_memory  # 1 x B x n_vocab
        log_prob = torch.log(probs).squeeze(0)  # B x n_vocab

        coverage = output_dict['coverage']
        last_coverage = coverage[:, -1, :].squeeze(1)

        output_dict['log_prob'] = log_prob
        output_dict['coverage'] = last_coverage

        return output_dict
