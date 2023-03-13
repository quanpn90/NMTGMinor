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

    return state




class WavLMEncoder(nn.Module):

    def __init__(self, opt, model_path="wav2vec_vox_new.pt",
                 **kwargs):

        super().__init__()
        # do we need opt for this?
        self.opt = opt
        self.model_path = model_path
        # import fairseq
        # from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
        from .fairseq_wav2vec2.wavlm import WavLM, WavLMConfig
        state = load_checkpoint_to_cpu(model_path)
        self.cfg = WavLMConfig(checkpoint['cfg'])

        self.cfg.dropout = self.opt.enc_pretrain_emb_dropout
        self.cfg.activation_dropout = self.opt.ffn_dropout
        self.cfg.encoder_layerdrop = self.opt.death_rate
        # self.cfg.dropout_features = self.opt.emb_dropout
        # self.cfg.mask_channel_before = True
        self.cfg.mask_channel_prob = 0.2 if self.opt.wav2vec_spec_augment else 0.0
        self.cfg.mask_channel_length = 64
        self.cfg.mask_prob = 0.0

        self.wavlm_encoder = WavLM(self.cfg)
        # load wav2vec weights
        wav2vec_weights = state['model']
        existed_weights = self.wav2vec_encoder.state_dict()

        # if we add new weights/buffers to new model then put them into the state_dict
        keys = existed_weights.keys()
        for key in keys:
            if key not in wav2vec_weights:
                wav2vec_weights[key] = existed_weights[key]

        self.wav2vec_encoder.load_state_dict(wav2vec_weights)

        cfg = self.cfg
        assert self.opt.model_size == cfg.encoder_embed_dim, \
            "Expect self.opt.model_size (%d) and cfg.encoder_embed_dim (%d) to equal " \
            % (self.opt.model_size, cfg.encoder_embed_dim)
        self.input_type = self.opt.encoder_type
        self.model_size = cfg.encoder_embed_dim
        self.time = None

        # freezing the parameters of the Convolutional feature extractors (by default)
        for param in self.wav2vec_encoder.feature_extractor.parameters():
            param.requires_grad = False

        # freeze the whole encoder. needs to do this first before adding customized parameters
        if opt.freeze_encoder:
            print("[INFO] Freezing encoder parameters")
            for p in self.wav2vec_encoder.parameters():
                p.requires_grad = False

        if opt.freeze_encoder_ffn:
            self.freeze_ffn_params()

    def freeze_ffn_params(self):
        for layer in self.wav2vec_encoder.encoder.layers:
            for p in layer.fc1.parameters():
                p.requires_grad = False
            for p in layer.fc2.parameters():
                p.requires_grad = False

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


        # output size is always T x B x C
        continuous_output = wav2vec_output['x']
        time, batch_size = continuous_output.size(0), continuous_output.size(1)

        # mask size is B x T (1 for padded positions, 0 for unpadded)
        dec_attn_mask = wav2vec_output['padding_mask']

        context = continuous_output

        if dec_attn_mask is None:
            dec_attn_mask = context.new_zeros(batch_size, time).byte()
        else:
            dec_attn_mask = dec_attn_mask.byte()

        wav2vec_context = context
        wav2vec_padding_mask = dec_attn_mask

        output_dict = defaultdict(lambda: None, {'source': input, 'context': context, 'src_mask': dec_attn_mask,
                                                 'src': dec_attn_mask, 'pos_emb': None,
                                                 'wav2vec_context': wav2vec_context,
                                                 'wav2vec_padding_mask': wav2vec_padding_mask})

        return output_dict



