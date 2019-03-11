# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import importlib
import os

from .lr_scheduler import LRScheduler


LR_SCHEDULER_REGISTRY = {}


def get_lr_scheduler_type(name):
    return LR_SCHEDULER_REGISTRY[name]


def get_lr_scheduler_names():
    return list(LR_SCHEDULER_REGISTRY.keys())


def build_lr_scheduler(name, args, optimizer):
    return get_lr_scheduler_type(name).build_lr_scheduler(args, optimizer)


def register_lr_scheduler(name):
    """Decorator to register a new LR scheduler."""

    def register_lr_scheduler_cls(cls):
        if name in LR_SCHEDULER_REGISTRY:
            raise ValueError('Cannot register duplicate LR scheduler ({})'.format(name))
        if not issubclass(cls, LRScheduler):
            raise ValueError('LR Scheduler ({}: {}) must extend LRScheduler'.format(name, cls.__name__))
        LR_SCHEDULER_REGISTRY[name] = cls
        return cls

    return register_lr_scheduler_cls


# automatically import any Python files in the optim/lr_scheduler/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('nmtg.optim.lr_scheduler.' + module)
