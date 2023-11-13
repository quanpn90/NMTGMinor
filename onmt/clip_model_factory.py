from __future__ import division

import datetime
import gc
import math
import os
import re
import time
import torch
import copy
import sys
import contextlib

import onmt
import onmt.markdown
import onmt.modules
from onmt.data.data_iterator import DataIterator
from onmt.data.multidata_iterator import MultiDataIterator
from onmt.data.dataset import rewrap
from onmt.model_factory import build_model, build_language_model, optimize_model
from onmt.model_factory import init_model_parameters
from onmt.modules.loss import NMTLossFunc, NMTAndCTCLossFunc
from onmt.train_utils.stats import Logger
from onmt.utils import checkpoint_paths, normalize_gradients, clip_grad_norm
from onmt.model_factory import build_model, optimize_model, init_model_parameters
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP_model
from torch.cuda.amp import autocast
import warnings
from onmt.constants import add_tokenidx
import dill

# ignore the pytorch -> numpy conversion warnings
warnings.filterwarnings("ignore", category=UserWarning)
from .model_factory import json_to_namespace


def build_clip_model(opt, dicts, constants=None):

    print("Building CLIP Wav2vec + MBART50 model ...")

    from onmt.models.speech_recognizer.wav2vec2 import FairseqWav2Vec
    from pretrain_module.modeling_mbart import MBartEncoder
    from pretrain_module.configuration_mbart import MBartConfig

    acoustic_encoder = FairseqWav2Vec(opt, model_path=opt.wav2vec2_pretrained_model)

    for parameter in acoustic_encoder.parameters():
        parameter.requires_grad = False  # don't update these guys

    if "s4" in opt.enc_pretrained_model:  # wav2vec_s4
        s4_config = json_to_namespace(opt.s4_config_file)
        print("[INFO] Replacing self attn in encoder with s4")
        acoustic_encoder.wav2vec_encoder.replace_attn_with_s4(s4_config)

    # add extra layers for the wav2vec model
    if opt.extra_layers > 0:
        print("[INFO] Adding extra layers on top of wav2vec")
        acoustic_encoder.wav2vec_encoder.add_extra_layers(opt.extra_layers)

    if opt.enc_config_file is not None and len(opt.enc_config_file) > 1:
        print("creating MBART sub-encoders")
        enc_mbart_config = MBartConfig.from_json_file(opt.enc_config_file)
        sub_encoder = MBartEncoder(enc_mbart_config, opt)

        if opt.enc_state_dict is not None and len(opt.enc_state_dict) > 1:
            print("[INFO] Loading weights for (sub) mBART encoder from: %s ..." % opt.enc_state_dict)
            enc_model_state_dict = torch.load(opt.enc_state_dict, map_location="cpu")
            sub_encoder.load_state_dict(enc_model_state_dict)
        for parameter in sub_encoder.parameters():
            parameter.requires_grad = False  # don't update these guys

    else:
        raise NotImplementedError("No configuration available for the MBART encoder!!!")

    from onmt.models.speech_recognizer.wav2vec_clip import Wav2VecCLIP

    model = Wav2VecCLIP(acoustic_encoder, sub_encoder)

    # add ctc output layer
    if opt.char_ctc and opt.ctc_loss > 0.0:
        print("creating CTC output layer")
        model.create_ctc_char(dicts['char_data'], ctc_compress=opt.ctc_compress)

    return model
