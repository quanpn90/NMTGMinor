import logging

from nmtg import convert

logger = logging.getLogger(__name__)


def average_checkpoints(filenames, method='mean'):
    if len(filenames) == 0:
        raise ValueError('No filenames specified')

    logger.info('Loading {}'.format(filenames[0]))
    checkpoint = convert.load_checkpoint(filenames[0])
    state_dict = checkpoint['model']

    # If some parameters are shared, don't consider them twice
    keys = list({value.data_ptr(): key for key, value in state_dict.items()}.values())

    del checkpoint['optimizer'], checkpoint['lr_scheduler']

    for filename in filenames[1:]:
        logger.info('Loading {}'.format(filename))
        new_checkpoint = convert.load_checkpoint(filenames[0])
        new_dict = new_checkpoint['model']

        if method == 'mean':
            # Arithmetic mean
            for key in keys:
                state_dict[key].add_(new_dict[key])
        elif method == 'gmean':
            # Geometric mean
            for key in keys:
                state_dict[key].mul_(new_dict[key])

        for key in ('epoch', 'num_updates', 'training_time'):
            checkpoint[key] = max(checkpoint[key], new_checkpoint[key])

        del new_dict
        del new_checkpoint

    if method == 'mean':
        for key in keys:
            state_dict[key].div_(float(len(filenames)))
    elif method == 'gmean':
        for key in keys:
            state_dict[key].pow_(1.0 / len(filenames))

    return checkpoint
