import argparse

import torch

from nmtg.custom_logging import add_log_options, setup_logging
from nmtg.options import add_general_options, add_trainer_option

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="preprocess.py")
    add_general_options(parser)
    trainer_class = add_trainer_option(parser)
    trainer_class.add_preprocess_options(parser)
    add_log_options(parser)

    args = parser.parse_args()

    logger = setup_logging(args.log_dir, 'preprocess', args.log_level_file, args.log_level_console)

    logger.debug('Torch version: {}'.format(torch.__version__))
    logger.debug(args)

    torch.manual_seed(args.seed)

    trainer_class.preprocess(args)
