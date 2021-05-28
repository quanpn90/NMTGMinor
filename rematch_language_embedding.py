#!/usr/bin/env python3
from __future__ import division

import onmt
import onmt.markdown
import torch
import argparse
import math
import numpy
import sys
import copy
from onmt.model_factory import build_model, build_language_model, optimize_model
from onmt.constants import add_tokenidx
from onmt.options import backward_compatible


parser = argparse.ArgumentParser(description='translate.py')
onmt.markdown.add_md_help_argument(parser)

parser.add_argument('-model_src', required=True,
                    help='Path to model .pt file')
parser.add_argument('-model_tgt', required=True,
                    help='Path to model .pt file')
parser.add_argument('-model_out', required=True,
                    help='Path to model .pt file')

opt = parser.parse_args()
# first, we load the model src
print(opt.model_src)
checkpoint = torch.load(opt.model_src, map_location=lambda storage, loc: storage)

model_opt = checkpoint['opt']
model_opt = backward_compatible(model_opt)

src_dicts = checkpoint['dicts']
# update special tokens
onmt.constants = add_tokenidx(model_opt, onmt.constants, src_dicts)

model = build_model(model_opt, checkpoint['dicts'])
model.load_state_dict(checkpoint['model'])

# now load the 2nd model
print(opt.model_tgt)
checkpoint = torch.load(opt.model_tgt, map_location=lambda storage, loc: storage)
# model_opt = checkpoint['opt']
# model_opt = backward_compatible(model_opt)
tgt_dicts = checkpoint['dicts']

# tgt_model = build_model(model_opt, checkpoint['dicts'])

# check the embedding
lang_emb = copy.deepcopy(model.encoder.language_embedding.weight.data)
new_emb = copy.deepcopy(lang_emb)

for key in src_dicts['langs']:

    old_idx = src_dicts['langs'][key]
    new_idx = tgt_dicts['langs'][key]
    print(key, old_idx, "->", new_idx)

    new_emb[new_idx].copy_(lang_emb[old_idx])

model.encoder.language_embedding.weight.data.copy_(new_emb)

model_state_dict = model.state_dict()

save_checkpoint = {
    'model': model_state_dict,
    'dicts': tgt_dicts,
    'opt': model_opt,
    'epoch': -1,
    'iteration': -1,
    'batchOrder': None,
    'optim': None
}

print("Saving converted model to %s" % opt.model_out)

torch.save(save_checkpoint, opt.model_out)

