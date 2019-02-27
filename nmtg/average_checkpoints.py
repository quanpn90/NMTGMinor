import logging

import torch

logger = logging.getLogger(__name__)


def average_checkpoints(filenames, method='mean'):
    if len(filenames) == 0:
        raise ValueError('No filenames specified')

    logger.info('Loading {}'.format(filenames[0]))
    checkpoint = torch.load(filenames[0], map_location='cpu')
    state_dict = checkpoint['train_data']['model']

    # If some parameters are shared, don't consider them twice
    keys = list({value.data_ptr(): key for key, value in state_dict.items()}.values())

    for key in list(checkpoint['train_data'].keys()):
        if key not in ['model', 'epoch', 'num_updates', 'training_time']:
            del checkpoint['train_data'][key]

    for filename in filenames[1:]:
        logger.info('Loading {}'.format(filename))
        new_checkpoint = torch.load(filename, map_location='cpu')
        new_dict = new_checkpoint['train_data']['model']

        if method == 'mean':
            # Arithmetic mean
            for key in keys:
                state_dict[key].add_(new_dict[key])
        elif method == 'gmean':
            # Geometric mean
            for key in keys:
                state_dict[key].mul_(new_dict[key])

        for key in ('epoch', 'num_updates', 'training_time'):
            checkpoint['train_data'][key] = max(checkpoint['train_data'][key],
                                                new_checkpoint['train_data'][key])

        del new_dict
        del new_checkpoint

    if method == 'mean':
        for key in keys:
            state_dict[key].div_(float(len(filenames)))
    elif method == 'gmean':
        for key in keys:
            state_dict[key].pow_(1.0 / len(filenames))

    return checkpoint
