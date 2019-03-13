import logging
import sys

import torch

from nmtg.data import Dictionary


class Dict:
    pass


sys.modules['onmt.Dict'] = sys.modules[__name__]

logger = logging.getLogger(__name__)


def load_checkpoint(filename):
    checkpoint = torch.load(filename, map_location='cpu')

    if 'batchOrder' in checkpoint:
        checkpoint = convert_checkpoint(checkpoint)
    if 'train_data' in checkpoint:
        checkpoint.update(checkpoint.pop('train_data'))

    return checkpoint


def convert_checkpoint(checkpoint):
    logger.info('Converting old checkpoint...')
    train_data = {
        'model': flatten_state_dict(
            convert_nmt_model(checkpoint['opt'],
                              unflatten_state_dict(checkpoint['model']))),
        'lr_scheduler': {'best': None},
        'training_time': 0.0
    }
    if 'optim' in checkpoint:
        num_updates = checkpoint['optim']['_step']
        del checkpoint['optim']['_step']
        train_data['optimizer'] = checkpoint['optim']
        train_data['num_updates'] = num_updates
    if 'epoch' in checkpoint:
        train_data['epoch'] = checkpoint['epoch']
    if 'iteration' in checkpoint:
        train_data['sampler'] = {'index': checkpoint['iteration'], 'batch_order': checkpoint['batchOrder']}

    new_checkpoint = {'train_data': train_data}

    # Dictionaries
    src_state_dict = Dictionary.convert(checkpoint['dicts']['src']).state_dict()
    join_vocab = checkpoint['dicts']['src'].labelToIdx == checkpoint['dicts']['tgt'].labelToIdx
    if join_vocab:
        new_checkpoint['dict'] = src_state_dict
    else:
        new_checkpoint['src_dict'] = src_state_dict
        tgt_state_dict = Dictionary.convert(checkpoint['dicts']['tgt']).state_dict()
        new_checkpoint['tgt_dict'] = tgt_state_dict
    args = checkpoint['opt']
    args.join_vocab = join_vocab
    args.word_vec_size = None
    input_chars = all(len(x[0]) == 1 for x in src_state_dict['dict'])
    args.input_type = 'char' if input_chars else 'word'
    new_checkpoint['args'] = args

    return new_checkpoint


def convert_nmt_model(opt, state_dict):
    res = {
        'encoder': {
            'embedded_dropout': {'embedding': {'weight': state_dict['encoder']['word_lut']['weight']}},
        },
        'decoder': {
            'embedded_dropout': {'embedding': {'weight': state_dict['decoder']['word_lut']['weight']}},
            'linear': {'weight': state_dict['generator']['linear']['weight'],
                       'bias': state_dict['generator']['linear']['bias']}
        }
    }
    model_type = opt.model
    if model_type != 'transformer':
        raise NotImplementedError('Currently, only Transformer models can be converted')
    model_state_dict = convert_transformer(opt, state_dict)
    res['encoder']['encoder'] = model_state_dict['encoder']
    res['decoder']['decoder'] = model_state_dict['decoder']
    return res


def convert_transformer(opt, state_dict):
    res = {'decoder': {'future_mask': state_dict['decoder']['mask']},
           'encoder': {}}

    if opt.time in ('lstm', 'gru'):
        res['encoder']['positional_encoding'] = {'rnn': state_dict['encoder']['time_transformer']}
        res['decoder']['positional_encoding'] = {'rnn': state_dict['decoder']['time_transformer']}
    else:
        res['encoder']['positional_encoding'] = state_dict['encoder']['time_transformer']
        res['decoder']['positional_encoding'] = state_dict['decoder']['time_transformer']
    res['encoder']['postprocess'] = state_dict['encoder']['postprocess_layer']
    res['decoder']['postprocess'] = state_dict['decoder']['postprocess_layer']

    def convert_linear_relu_linear(ffn_dict):
        return {'layer_1': ffn_dict['fc_1']['linear'], 'layer_2': ffn_dict['fc_2']['linear']}

    def convert_maxout(ffn_dict):
        return {'linear': ffn_dict['lin']}

    convert_ffn = convert_linear_relu_linear if opt.activation_layer == 'linear_relu_linear' else convert_maxout

    res['encoder']['layers'] = {}
    res['decoder']['layers'] = {}
    for i in range(opt.layers):
        layer_in = state_dict['encoder']['layer_modules'][str(i)]
        layer_dict = {
            'preprocess_attn': layer_in['preprocess_attn'],
            'preprocess_ffn': layer_in['preprocess_ffn'],
            'attention': {
                'query_projection': {'function': layer_in['multihead']['fc_query']['function']['linear']},
                'key_projection': {'function': layer_in['multihead']['fc_key']['function']['linear']},
                'value_projection': {'function': layer_in['multihead']['fc_value']['function']['linear']},
                'out_projection': {'function': layer_in['multihead']['fc_concat']['function']['linear']}
            },
            'feed_forward': {'function': convert_ffn(layer_in['feedforward']['function'])}
        }
        res['encoder']['layers'][str(i)] = layer_dict

        layer_in = state_dict['decoder']['layer_modules'][str(i)]
        layer_dict = {
            'preprocess_attn': layer_in['preprocess_attn'],
            'preprocess_ffn': layer_in['preprocess_ffn'],
            'attention_tgt': {
                'query_projection': {'function': layer_in['multihead_tgt']['fc_query']['function']['linear']},
                'key_projection': {'function': layer_in['multihead_tgt']['fc_key']['function']['linear']},
                'value_projection': {'function': layer_in['multihead_tgt']['fc_value']['function']['linear']},
                'out_projection': {'function': layer_in['multihead_tgt']['fc_concat']['function']['linear']}
            },
            'feed_forward': {'function': convert_ffn(layer_in['feedforward']['function'])},
            'preprocess_src_attn': layer_in['preprocess_src_attn'],
            'attention_src': {
                'query_projection': {'function': layer_in['multihead_src']['fc_query']['function']['linear']},
                'key_projection': {'function': layer_in['multihead_src']['fc_key']['function']['linear']},
                'value_projection': {'function': layer_in['multihead_src']['fc_value']['function']['linear']},
                'out_projection': {'function': layer_in['multihead_src']['fc_concat']['function']['linear']}
            }
        }
        res['decoder']['layers'][str(i)] = layer_dict

    opt.batch_first = False
    opt.ignore_context = False
    opt.freeze_embeddings = False
    opt.mask_layers = False
    opt.no_future_masking = False

    return res


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
