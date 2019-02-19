import argparse

import torch

from nmtg import convert, custom_logging
from nmtg.custom_logging import add_log_options
from nmtg.options import add_general_options, add_task_option, add_trainer_option
from nmtg.tasks import Task
from nmtg.trainers import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train.py")
    add_general_options(parser)
    task_class = add_task_option(parser)
    task_class.add_options(parser)
    trainer_class = add_trainer_option(parser)
    trainer_class.add_eval_options(parser)
    add_log_options(parser)

    parser.add_argument('-load_from', type=str, required=True,
                        help='Path to a pretrained model.')
    parser.add_argument('-output',
                        help="Path to output the predictions")

    args = parser.parse_args()

    logger = custom_logging.setup_logging_from_args(args, 'validate')

    logger.debug('Torch version: {}'.format(torch.__version__))
    logger.debug(args)

    torch.manual_seed(args.seed)

    task = task_class.setup_task(args)  # type: Task

    trainer = trainer_class(args)  # type: Trainer

    model = trainer.load_checkpoint(convert.load_checkpoint(args.load_from))

    val_loss = trainer.evaluate(model, task)
