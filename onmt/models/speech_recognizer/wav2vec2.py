import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from onmt.models.transformers import Transformer, TransformerDecodingState
from typing import List, Optional, Union
from collections import defaultdict
import onmt
from onmt.modules.optimized.linear import Linear
import math
from .fairseq_wav2vec2.file_io import PathManager
from omegaconf import DictConfig, open_dict, OmegaConf
from .fairseq_wav2vec2.utils import overwrite_args_by_name

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
        from .fairseq_wav2vec2.wav2vec2 import Wav2Vec2Model
        state = load_checkpoint_to_cpu(model_path)
        self.cfg = state['cfg']['model']

        # don't override the options for wav2vec yet (some of them can create NaN)
        self.cfg.dropout = self.opt.enc_pretrain_emb_dropout
        self.cfg.activation_dropout = self.opt.ffn_dropout
        # self.cfg.attention_dropout = self.opt.enc_pretrain_hidden_dropout
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
                                                       fast=opt.fast_factorize,
                                                       sub_factors=opt.n_attributes,
                                                       sub_factor_rank=math.floor(
                                                           opt.mfw_rank * opt.mfw_atb_rank_scale))

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

        # if self.quantize:
        #     quantized_codebooks = wav2vec_output['quantized_target']
        #     encoder_input = quantized_codebooks.prod(dim=-1, keepdim=False)  # .transpose(0, 1)  # -> t x b x groups
        #     dec_attn_mask = wav2vec_output['padding_mask'] # b x t
        #
        #     # 44204 = magic number
        #     additional_mask = encoder_input.eq(44204)
        #
        #     if dec_attn_mask is not None:
        #         dec_attn_mask = torch.logical_or(dec_attn_mask.bool(), additional_mask)
        #     else:
        #         dec_attn_mask = additional_mask
        #
        #     discrete_encoder_output = self.discrete_encoder(input_ids=encoder_input, attention_mask=dec_attn_mask)
        #     discrete_output = discrete_encoder_output[0]
        #     batch_size, time = discrete_output.size(1), discrete_output.size(0)
        #     if batch_first_output:
        #         discrete_output = discrete_output.transpose(0, 1).contiguous()
        #         batch_size, time = discrete_output.size(0), discrete_output.size(1)
        # else:
        #     discrete_output = None

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
                                                 'wav2vec_padding_mask': wav2vec_padding_mask})

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
        super().__init__(encoder, decoder, generator, mirror=mirror, ctc=ctc)

        self.src_vocab_size = 0
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.sub_encoder = sub_encoder

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

        if self.ctc:
            self.ctc_linear = nn.Linear(encoder.model_size, self.tgt_vocab_size)

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
