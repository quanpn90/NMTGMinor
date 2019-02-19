import argparse
import torch

from nmtg import custom_logging
from nmtg.average_checkpoints import average_checkpoints

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints', nargs='+',
                        help='Which checkpoints to average')
    parser.add_argument('-output', type=str, required=True,
                        help='Output filename')
    parser.add_argument('-method', choices=['mean', 'gmean'], default='mean',
                        help='Method of averaging')
    custom_logging.add_log_options(parser)
    args = parser.parse_args()
    logger = custom_logging.setup_logging_from_args(args, 'average_checkpoints.py')

    checkpoint = average_checkpoints(args.checkpoints, args.method)

    logger.info('Saving checkpoint to {}'.format(args.output))
    torch.save(checkpoint, args.output)
