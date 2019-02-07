import argparse

import torch

from nmtg.logging import add_log_options, setup_logging
from nmtg.options import add_general_options, add_solution_option
from nmtg.solutions import get_solution_type


def main(args):
    logger = setup_logging(args.log_dir, 'preprocess', args.log_level_file, args.log_level_console)

    logger.debug('Torch version: {}'.format(torch.__version__))
    logger.debug(args)

    torch.manual_seed(args.seed)

    solution_class = get_solution_type(args.solution)
    solution_class.preprocess(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="preprocess.py")
    add_general_options(parser)
    solution_class = add_solution_option(parser)
    solution_class.add_preprocess_options(parser)
    add_log_options(parser)

    args = parser.parse_args()

    main(args)
