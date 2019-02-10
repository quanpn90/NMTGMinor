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

    def __init__(self, optimizer, learning_rate, warmup_steps=0, force_anneal=None, lr_shrink=0.1):
        self.warmup_steps = warmup_steps
        self.force_anneal = force_anneal
        self.lr_shring = lr_shrink

        self.lr = learning_rate
        if warmup_steps > 0:
            self.warmup_factor = 1. / warmup_steps
        else:
            self.warmup_factor = 1

        super().__init__(optimizer)

    @staticmethod
    def add_options(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('-learning_rate', type=float, default=1.0,
                            help='Learning rate')
        parser.add_argument('-force_anneal', type=int,
                            help='force annealing at specified epoch')
        parser.add_argument('-warmup_steps', type=int, default=4096,
                            help='Number of steps to increase the lr in noam')
        parser.add_argument('-lr-shrink', default=0.1, type=float,
                            help='learning rate shrink factor for annealing, lr_new = (lr * lr_shrink)')
        # fmt: on

    @classmethod
    def build_lr_scheduler(cls, args, optimizer):
        return cls(optimizer, args.learning_rate, args.warmup_steps, args.force_anneal, args.lr_srhink)

    def get_next_lr(self, epoch):
        lrs = self.lr
        if self.force_anneal is None or epoch < self.force_anneal:
            # use fixed LR schedule
            next_lr = lrs[min(epoch, len(lrs) - 1)]
        else:
            # annneal based on lr_shrink
            next_lr = lrs[-1] * self.lr_shrink ** (epoch + 1 - self.force_anneal)
        return next_lr

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        self.lr = self.get_next_lr(epoch)
        self.optimizer.set_lr(self.warmup_factor * self.lr)
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if self.warmup_steps > 0 and num_updates <= self.warmup_steps:
            self.warmup_factor = num_updates / float(self.warmup_steps)
            self.optimizer.set_lr(self.warmup_factor * self.lr)
        return self.optimizer.get_lr()
