# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from . import LRScheduler, register_lr_scheduler


@register_lr_scheduler('fixed')
@register_lr_scheduler('regular')
class FixedSchedule(LRScheduler):
    """Decay the LR on a fixed schedule."""

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)

        # set defaults
        args.warmup_steps = getattr(args, 'warmup_steps', 0) or 0

        self.lr = args.learning_rate
        if args.warmup_steps > 0:
            self.warmup_factor = 1. / args.warmup_steps
        else:
            self.warmup_factor = 1

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('-learning_rate', type=float, default=1.0,
                            help='Learning rate')
        parser.add_argument('-force_anneal', type=int,
                            help='force annealing at specified epoch')
        parser.add_argument('-warmup_steps', type=int, default=4096,
                            help='Number of steps to increase the lr in noam')
        parser.add_argument('--lr-shrink', default=0.1, type=float,
                            help='learning rate shrink factor for annealing, lr_new = (lr * lr_shrink)')
        # fmt: on

    def get_next_lr(self, epoch):
        lrs = self.args.lr
        if self.args.force_anneal is None or epoch < self.args.force_anneal:
            # use fixed LR schedule
            next_lr = lrs[min(epoch, len(lrs) - 1)]
        else:
            # annneal based on lr_shrink
            next_lr = lrs[-1] * self.args.lr_shrink ** (epoch + 1 - self.args.force_anneal)
        return next_lr

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        self.lr = self.get_next_lr(epoch)
        self.optimizer.set_lr(self.warmup_factor * self.lr)
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if self.args.warmup_steps > 0 and num_updates <= self.args.warmup_steps:
            self.warmup_factor = num_updates / float(self.args.warmup_steps)
            self.optimizer.set_lr(self.warmup_factor * self.lr)
        return self.optimizer.get_lr()
