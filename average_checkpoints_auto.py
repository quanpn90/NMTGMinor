from __future__ import division

import onmt
import onmt.markdown
import torch
import argparse
import math
import numpy
import os, sys
from onmt.model_factory import build_model, build_language_model
from copy import deepcopy
from onmt.utils import checkpoint_paths, normalize_gradients
import glob


parser = argparse.ArgumentParser(description='translate.py')
onmt.markdown.add_md_help_argument(parser)

parser.add_argument('-models', required=True,
                    help='Path to model .pt file')
parser.add_argument('-lm', action='store_true',
                    help='Language model (default is seq2seq model')
parser.add_argument('-sort_by_date', action='store_true',
                    help='Sort the model files by date')
parser.add_argument('-output', default='model.averaged',
                    help="""Path to output averaged model""")
parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")
parser.add_argument('-top', type=int, default=10,
                    help="Device to run on")
parser.add_argument('-method', default='mean',
                    help="method to average: mean|gmean")


def custom_build_model(opt, dict, lm=False):

    if not lm:
        model = build_model(opt, dict)
    else:
        model = build_language_model(opt, dict)

    return model


def main():
    
    opt = parser.parse_args()
    
    opt.cuda = opt.gpu > -1

    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    path = opt.models

    if not opt.sort_by_date:
        existed_save_files = checkpoint_paths(path)
    else:
        existed_save_files = glob.glob(path + "/" + "*.pt")
        existed_save_files.sort(key=os.path.getmtime)
        print("\n".join(existed_save_files))

    # print(existed_save_files)
    models = existed_save_files
    # # opt.model should be a string of models, split by |
    # models = list()
    #

    # take the top
    models = models[:opt.top]

    # print(models)
    #
    n_models = len(models)
    #
    print("Loading main model from %s ..." % models[0])
    checkpoint = torch.load(models[0], map_location=lambda storage, loc: storage)

    if 'optim' in checkpoint:
        del checkpoint['optim']

    main_checkpoint = checkpoint

    # best_checkpoint = {
    #     'model': deepcopy(main_checkpoint['model']),
    #     'dicts': main_checkpoint['dicts'],
    #     'opt': main_checkpoint['opt'],
    #     'epoch': -1,
    #     'iteration': -1,
    #     'batchOrder': None,
    #     'optim': None
    # }
    best_checkpoint = main_checkpoint

    # print("Saving best model to %s" % opt.output + ".top")

    # torch.save(best_checkpoint, opt.output + ".top")

    model_opt = checkpoint['opt']
    dicts = checkpoint['dicts']

    print(model_opt.layers)

    main_model = custom_build_model(model_opt, checkpoint['dicts'], lm=opt.lm)

    main_model.load_state_dict(checkpoint['model'])

    if opt.cuda:
        main_model = main_model.cuda()

    for i in range(1, len(models)):

        model = models[i]
        print("Loading model from %s ..." % models[i])
        checkpoint = torch.load(model, map_location=lambda storage, loc: storage)

        model_opt = checkpoint['opt']

        # delete optim information to save GPU memory
        if 'optim' in checkpoint:
            del checkpoint['optim']

        current_model = custom_build_model(model_opt, checkpoint['dicts'], lm=opt.lm)

        current_model.load_state_dict(checkpoint['model'])

        if opt.cuda:
            current_model = current_model.cuda()

        if opt.method == 'mean':
            # Sum the parameter values
            for (main_param, param) in zip(main_model.parameters(), current_model.parameters()):
                main_param.data.add_(param.data)
        elif opt.method == 'gmean':
            # Take the geometric mean of parameter values
            for (main_param, param) in zip(main_model.parameters(), current_model.parameters()):
                main_param.data.mul_(param.data)
        else:
            raise NotImplementedError

    # Normalizing
    if opt.method == 'mean':
        for main_param in main_model.parameters():
            main_param.data.div_(n_models)
    elif opt.method == 'gmean':
        for main_param in main_model.parameters():
            main_param.data.pow_(1. / n_models)

    # Saving
    model_state_dict = main_model.state_dict()

    save_checkpoint = {
        'model': model_state_dict,
        'dicts': dicts,
        'opt': model_opt,
        'epoch': -1,
        'iteration': -1,
        'batchOrder': None,
        'optim': None
    }

    print("Saving averaged model to %s" % opt.output)

    torch.save(save_checkpoint, opt.output)




if __name__ == "__main__":
    main()
