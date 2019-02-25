import argparse
import os

import torch

from nmtg import convert, custom_logging
from nmtg.custom_logging import add_log_options
from nmtg.options import add_general_options, add_task_option, add_trainer_option
from nmtg.trainers import Trainer
from nmtg.tasks import Task


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train.py")
    add_general_options(parser)
    task_class = add_task_option(parser)
    task_class.add_options(parser)
    add_log_options(parser)
    parser.add_argument('results',
                        help='Filename with the results.')

    args = parser.parse_args()

    logger = custom_logging.setup_logging_from_args(args, 'score')

    logger.debug('Torch version: {}'.format(torch.__version__))
    logger.debug(args)

    torch.manual_seed(args.seed)

    task = task_class.setup_task(args)  # type: Task

    results = task.load_results(args.results)

    logger.info(' | '.join(task.score_results(results)))
