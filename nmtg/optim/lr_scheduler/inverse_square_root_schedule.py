# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from . import LRScheduler, register_lr_scheduler


@register_lr_scheduler('inverse_sqrt')
@register_lr_scheduler('noam')
class InverseSquareRootSchedule(LRScheduler):
    """Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup::

      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_steps)
      lr = lrs[update_num]

    After warmup::

      decay_factor = args.lr * sqrt(args.warmup_steps)
      lr = decay_factor / sqrt(update_num)
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)

        # TODO: Find a better way to do this, ideally still keeping backwards compatibility
        self.init_lr = args.model_size ** (-0.5) * args.learning_rate
        self.lr = self.init_lr
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('-learning_rate', type=float, default=1.0,
                            help='Learning rate multiplier')
        parser.add_argument('-warmup_steps', type=int, default=4096,
                            help='Number of steps to increase the lr in noam')
        # fmt: on

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.args.warmup_steps:
            self.lr = self.init_lr * num_updates * self.args.warmup_steps ** (-1.5)
        else:
            self.lr = self.init_lr * num_updates**(-0.5)
        self.optimizer.set_lr(self.lr)
        return self.lr
