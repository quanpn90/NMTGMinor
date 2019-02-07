import argparse

import torch

from nmtg.logging import add_log_options, setup_logging
from nmtg.options import add_general_options, add_task_option, add_solution_option
from nmtg.solutions import Solution
from nmtg.tasks import Task


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train.py")
    add_general_options(parser)
    task_class = add_task_option(parser)
    task_class.add_options(parser)
    solution_class = add_solution_option(parser)
    solution_class.add_options(parser)
    add_log_options(parser)

    args, _ = parser.parse_known_args()

    logger = setup_logging(args.log_dir, 'train', args.log_level_file, args.log_level_console)

    logger.debug('Torch version: {}'.format(torch.__version__))
    logger.debug(args)

    torch.manual_seed(args.seed)

    task = task_class.setup_task(args)  # type: Task

    solution = solution_class.setup_solution_eval(parser, task)  # type: Solution

    results = solution.solve()

    logger.info(' | '.join(task.score_results(results)))

    if args.output is not None:
        task.save_results(results)
