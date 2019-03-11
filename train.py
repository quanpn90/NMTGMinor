import argparse

import numpy as np
import torch

from nmtg.convert import load_checkpoint
from nmtg.custom_logging import add_log_options, setup_logging_from_args
from nmtg.options import add_general_options, add_task_option, add_trainer_option
from nmtg.trainers import Trainer
from nmtg.tasks import Task


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train.py", allow_abbrev=False)
    add_general_options(parser)
    task_class = add_task_option(parser)
    task_class.add_options(parser)
    trainer_class = add_trainer_option(parser)
    trainer_class.add_training_options(parser)
    add_log_options(parser)

    parser.add_argument('-load_from', type=str,
                        help='If training from a checkpoint then this is the'
                             'path to the pretrained model.')
    parser.add_argument('-reset_optim', action='store_true')

    args = parser.parse_args()

    logger = setup_logging_from_args(args, 'train')

    logger.debug('Torch version: {}'.format(torch.__version__))
    logger.debug(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    task = task_class.setup_task(args)  # type: Task

    if args.load_from is not None:
        logger.info("Loading checkpoint {}".format(args.load_from))
        checkpoint = load_checkpoint(args.load_from)
    else:
        checkpoint = None

    trainer = trainer_class(args, checkpoint=checkpoint, for_training=True)  # type: Trainer

    trainer.train(task, checkpoint=checkpoint if not args.reset_optim else None)
