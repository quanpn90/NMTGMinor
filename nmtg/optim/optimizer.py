# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
import itertools
import math
import torch
from typing import Sequence


class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    @classmethod
    def build_optimizer(cls, args, params):
        return cls(params)

    @property
    def optimizer(self):
        """Return a torch.optim.optimizer.Optimizer instance."""
        if not hasattr(self, '_optimizer'):
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError('_optimizer must be an instance of torch.optim.Optimizer')
        return self._optimizer

    @staticmethod
    def add_options(parser):
        """Add optimizer-specific arguments to the parser."""
        raise NotImplementedError

    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        """Return the optimizer's state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load an optimizer state dict."""
        self.optimizer.load_state_dict(state_dict)

    def backward(self, tensor, gradient=None, retain_graph=None, create_graph=False):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves.
        """
        tensor.backward(gradient, retain_graph, create_graph)

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        if c == 1:
            return

        for p in self.params:
            if p.grad is not None:
                p.grad.mul_(c)

    def clip_grad_norm(self, max_norm):
        """Clips gradient norm."""
        if max_norm > 0:
            return torch.nn.utils.clip_grad_norm_(self.params, max_norm)
        else:
            return math.sqrt(sum(p.grad.norm()**2 for p in self.params if p.grad is not None))

    def step(self, closure=None):
        """Performs a single optimization step."""
        self.optimizer.step(closure)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        self.optimizer.zero_grad()
