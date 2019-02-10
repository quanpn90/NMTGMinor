import argparse

import torch

from nmtg import convert
from nmtg.logging import add_log_options, setup_logging
from nmtg.options import add_general_options, add_task_option, add_trainer_option
from nmtg.trainers import Trainer
from nmtg.tasks import Task


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train.py")
    add_general_options(parser)
    task_class = add_task_option(parser)
    task_class.add_options(parser)
    trainer_class = add_trainer_option(parser)
    trainer_class.add_eval_options(parser)
    add_log_options(parser)

    parser.add_argument('-load_from', type=str, nargs='+', required=True,
                        help='Path to one or more pretrained models.')
    parser.add_argument('-output',
                        help="Path to output the predictions")

    args = parser.parse_args()

    logger = setup_logging(args.log_dir, 'train', args.log_level_file, args.log_level_console)

    logger.debug('Torch version: {}'.format(torch.__version__))
    logger.debug(args)

    torch.manual_seed(args.seed)

    task = task_class.setup_task(args)  # type: Task

    trainer = trainer_class(args)  # type: Trainer

    models = [trainer.load_checkpoint(convert.load_checkpoint(filename)) for filename in args.load_from]

    if args.cuda:
        for model in models:
            model.cuda()

    results = trainer.solve(models, task)

    logger.info(' | '.join(task.score_results(results)))

    if args.output is not None:
        task.save_results(results, args.output)
