from __future__ import division

import onmt
import onmt.markdown
import torch
import argparse
import math
import numpy
import os, sys
from onmt.model_factory import build_model, build_language_model, build_classifier, optimize_model
from copy import deepcopy
from onmt.utils import checkpoint_paths, normalize_gradients
import glob
import torch.nn as nn


parser = argparse.ArgumentParser(description='translate.py')
onmt.markdown.add_md_help_argument(parser)

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')

parser.add_argument('-output', default='model.averaged',
                    help="""Path to output averaged model""")
parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")
parser.add_argument('-n_languages', type=int, default=10,
                    help="Device to run on")



def custom_build_model(opt, dict, lm=False, type='seq2seq'):

    if type == 'seq2seq':
        if not lm:
            model = build_model(opt, dict)
        else:
            model = build_language_model(opt, dict)
    elif type == 'classifier':
        model = build_classifier(opt, dict)

    optimize_model(model)

    return model


def main():
    
    opt = parser.parse_args()
    
    opt.cuda = opt.gpu > -1

    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    # checkpoint for main model
    checkpoint = torch.load(opt.model, map_location=lambda storage, loc: storage)

    if 'optim' in checkpoint:
        del checkpoint['optim']

    model_opt = checkpoint['opt']

    dicts = checkpoint['dicts']

    # extending the weights
    def is_factorize_params(p_name):

        # feed forward neural net
        if p_name.endswith(".r_i") or p_name.endswith(".s_i") \
                or p_name.endswith(".r_o") or p_name.endswith(".s_o") \
                or p_name.endswith(".r_p") or p_name.endswith(".s_p"):
            return True

        # if p_name.endswith(".sub_r_i") or p_name.endswith(".sub_s_i") \
        #         or p_name.endswith(".sub_r_o") or p_name.endswith(".sub_s_o") \
        #         or p_name.endswith(".sub_r_p") or p_name.endswith(".sub_s_p"):
        #     return True

        if p_name.endswith(".rm_i") or p_name.endswith(".sm_i") or \
                p_name.endswith(".rm_o") or p_name.endswith(".sm_o") or \
                p_name.endswith(".rm_p") or p_name.endswith(".sm_p"):
            return True

        if p_name.endswith(".r_q") or p_name.endswith(".s_q") \
                or p_name.endswith(".r_o") or p_name.endswith(".s_o") \
                or p_name.endswith(".r_kv") or p_name.endswith(".s_kv"):
            return True

        if p_name.endswith(".rm_q") or p_name.endswith(".sm_q") \
                or p_name.endswith(".rm_o") or p_name.endswith(".sm_o") \
                or p_name.endswith(".rm_kv") or p_name.endswith(".sm_kv"):
            return True

        return False

    # Saving
    model_state_dict = checkpoint['model']

    for name in model_state_dict:
        if is_factorize_params(name):
            param = model_state_dict[name]
            sizes = list(param.size())
            print(name)

            # initialize it
            if name.endswith("r_i") or name.endswith("r_o") or name.endswith("r_kv") or name.endswith("r_q") or name.endswith("r_p") or \
                    name.endswith("s_i") or name.endswith("s_o") or name.endswith("s_kv") or name.endswith("s_q") or name.endswith(
                "s_p"):
                std = 0.02
                prev_n_languages = sizes[0]
                sizes[0] = max(opt.n_languages, sizes[0])
                # new parameter
                p = param.new_zeros(sizes)

                nn.init.normal_(p, 0.0, std)
                p[0:prev_n_languages].copy_(param)

            elif name.endswith("rm_i") or name.endswith("rm_o") or name.endswith("rm_kv") or name.endswith("rm_q") or name.endswith("rm_p") or \
                    name.endswith("sm_i") or name.endswith("sm_o") or name.endswith("sm_kv") or name.endswith("sm_q") or name.endswith(
                "sm_p"):
                rank = sizes[1]
                fast = (sizes[0] > 1)
                prev_n_languages = sizes[0]
                if fast:
                    # new parameter
                    sizes[0] = max(opt.n_languages, sizes[0])
                    p = param.new_zeros(sizes)
                else:
                    sizes[0] = 1
                    p = param.new_zeros(sizes)
                sizes[0] = 1

                constant = math.sqrt(1.0 / rank) if fast else 1
                nn.init.constant_(p, constant)

                if fast:
                    p[0:prev_n_languages].copy_(param)
                else:
                    p.copy_(param)

            model_state_dict[name] = p

    save_checkpoint = {
        'model': model_state_dict,
        'dicts': dicts,
        'opt': model_opt,
        'epoch': -1,
        'iteration': -1,
        'batchOrder': None,
        'optim': None
    }

    output = opt.model + ".extend" + str(opt.n_languages)
    print("Saving averaged model to %s" % output)

    torch.save(save_checkpoint, output)

if __name__ == "__main__":
    main()
