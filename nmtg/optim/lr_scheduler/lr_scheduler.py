# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from .. import Optimizer


class LRScheduler:
    def __init__(self, optimizer):
        super().__init__()
        if not isinstance(optimizer, Optimizer):
            raise ValueError('optimizer must be an instance of Optimizer')
        self.optimizer = optimizer
        self.best = None
        self.step_update(0)

    @classmethod
    def build_lr_scheduler(cls, args, optimizer):
        return cls(optimizer)

    @staticmethod
    def add_options(parser):
        """Add arguments to the parser for this LR scheduler."""
        raise NotImplementedError

    def state_dict(self):
        """Return the LR scheduler state dict."""
        return {'best': self.best}

    def load_state_dict(self, state_dict):
        """Load an LR scheduler state dict."""
        self.best = state_dict['best']

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        if val_loss is not None:
            if self.best is None:
                self.best = val_loss
            else:
                self.best = min(self.best, val_loss)

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        return self.optimizer.get_lr()
