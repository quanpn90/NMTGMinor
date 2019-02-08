import logging
import sys

import torch

from nmtg.data import Dictionary
from nmtg.models.nmt_model import NMTModel


class Dict:
    pass


sys.modules['onmt.Dict'] = sys.modules[__name__]

logger = logging.getLogger(__name__)


def load_checkpoint(filename):
    checkpoint = torch.load(filename, map_location='cpu')

    if 'batchOrder' in checkpoint:
        checkpoint = convert_checkpoint(checkpoint)

    return checkpoint


def convert_checkpoint(checkpoint):
    logger.info('Converting old checkpoint...')
    num_updates = checkpoint['optim']['_step']
    del checkpoint['optim']['_step']
    new_checkpoint = {
        'args': checkpoint['opt'],
        'train_data': {
            'model': flatten_state_dict(
                NMTModel.convert_state_dict(checkpoint['opt'],
                                            unflatten_state_dict(checkpoint['model']))),
            'epoch': checkpoint['epoch'],
            'sampler': {'index': checkpoint['iteration'], 'batch_order': checkpoint['batchOrder']},
            'num_updates': num_updates,
            'optimizer': checkpoint['optim'],
            'lr_scheduler': {'best': None}
        }
    }

    # Dictionaries
    src_state_dict = Dictionary.convert(checkpoint['dicts']['src']).state_dict()
    join_vocab = checkpoint['dicts']['src'].labelToIdx == checkpoint['dicts']['tgt'].labelToIdx
    if join_vocab:
        new_checkpoint['dict'] = src_state_dict
    else:
        new_checkpoint['src_dict'] = src_state_dict
        tgt_state_dict = Dictionary.convert(checkpoint['dicts']['tgt']).state_dict()
        new_checkpoint['tgt_dict'] = tgt_state_dict

    return new_checkpoint


def unflatten_state_dict(state_dict):
    res = {}
    for key, value in state_dict.items():
        key = key.split('.')
        current = res
        for part in key[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[key[-1]] = value
    return res


def flatten_state_dict(state_dict, prefix=''):
    res = {}
    for k, v in state_dict.items():
        key = prefix + k
        value = v
        if isinstance(value, dict):
            res.update(flatten_state_dict(value, key + '.'))
        else:
            res[key] = value
    return res
